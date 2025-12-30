# core/stage2_engine.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import time


class Zone(str, Enum):
    SAFE = "SAFE"
    CLOSE = "CLOSE"
    DANGER = "DANGER"
    UNKNOWN = "UNKNOWN"


class State(str, Enum):
    INIT = "INIT"                 # collecting baseline
    ACTIVE = "ACTIVE"             # baseline locked + tracking ok
    PAUSED_NO_FACE = "PAUSED_NO_FACE"
    PAUSED_UNSTABLE = "PAUSED_UNSTABLE"


@dataclass
class EngineOutput:
    state: State
    baseline_locked: bool
    baseline_width: Optional[float]

    ratio_raw: Optional[float]
    ratio_smooth: Optional[float]

    zone: Zone
    zone_changed: bool
    message_key: str  # for UI: CALIBRATING / SAFE / CLOSE / DANGER / NO_FACE / UNSTABLE


class DistanceEngine:
    """
    Stage-2 engine:
    - Baseline capture (median) for first N seconds (stable frames only)
    - Ratio = baseline / current width
    - EMA smoothing (hold when unstable)
    - Zone classification
    - State machine outputs for UI
    """

    def __init__(
        self,
        baseline_seconds: float = 15.0,
        min_samples: int = 60,             # ~ 4s @ 15fps stable frames
        ema_alpha: float = 0.2,
        

        safe_low: float = 0.90,
        safe_high: float = 1.10,
        close_high: float = 1.30,

        zone_hold_seconds: float = 0.6     # prevent rapid flip-flop on boundary
    ):
        self.baseline_seconds = baseline_seconds
        self.min_samples = min_samples
        self.ema_alpha = ema_alpha

        self.safe_low = safe_low
        self.safe_high = safe_high
        self.close_high = close_high

        self.zone_hold_seconds = zone_hold_seconds

        self._start_time = time.time()

        self._baseline_samples: List[float] = []
        self._baseline_width: Optional[float] = None
        self._baseline_locked = False

        self._ratio_ema: Optional[float] = None

        self._last_zone: Zone = Zone.UNKNOWN
        self._last_zone_change_t = time.time()
        self._min_width = None
        self._max_width = None


    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    @property
    def baseline_width(self) -> Optional[float]:
        return self._baseline_width

    def _classify_zone(self, ratio: float) -> Zone:
        if self.safe_low <= ratio <= self.safe_high:
            return Zone.SAFE
        elif self.safe_high < ratio <= self.close_high:
            return Zone.CLOSE
        elif ratio > self.close_high:
            return Zone.DANGER
        return Zone.UNKNOWN

    def _zone_with_hold(self, proposed: Zone) -> Tuple[Zone, bool]:
        """
        Only allow zone change if held long enough, to reduce boundary oscillation.
        """
        now = time.time()
        if proposed == self._last_zone:
            return proposed, False

        if (now - self._last_zone_change_t) < self.zone_hold_seconds:
            # too soon to change; keep old
            return self._last_zone, False

        # accept change
        prev = self._last_zone
        self._last_zone = proposed
        self._last_zone_change_t = now
        return proposed, True

    def update(
        self,
        face_detected: bool,
        tracking_stable: bool,
        width_px: Optional[float],
        frame_rejected: bool
    ) -> EngineOutput:
        """
        Call once per frame.
        tracking_stable should be True only when:
        - face_detected True
        - width available
        - not rejected (no hard jump)
        """
        # Priority: no face
        if not face_detected:
            return EngineOutput(
                state=State.PAUSED_NO_FACE,
                baseline_locked=self._baseline_locked,
                baseline_width=self._baseline_width,
                ratio_raw=None,
                ratio_smooth=self._ratio_ema,
                zone=self._last_zone,
                zone_changed=False,
                message_key="NO_FACE"
            )

        # Face present but unstable (or rejected)
        if (not tracking_stable) or frame_rejected or width_px is None:
            return EngineOutput(
                state=State.PAUSED_UNSTABLE,
                baseline_locked=self._baseline_locked,
                baseline_width=self._baseline_width,
                ratio_raw=None,
                ratio_smooth=self._ratio_ema,  # hold last EMA
                zone=self._last_zone,
                zone_changed=False,
                message_key="UNSTABLE"
            )

        # INIT: baseline learning window
        if not self._baseline_locked:
            elapsed = time.time() - self._start_time
            # collect samples only in stable frames
            w = float(width_px)
            self._baseline_samples.append(w)

            self._min_width = w if self._min_width is None else min(self._min_width, w)
            self._max_width = w if self._max_width is None else max(self._max_width, w)
            # lock baseline if either time window done AND enough samples
            if (elapsed >= self.baseline_seconds) and (len(self._baseline_samples) >= self.min_samples):

                median_w = float(np.median(np.array(self._baseline_samples, dtype=np.float32)))
                motion_ratio = (self._max_width - self._min_width) / max(median_w, 1e-6)

                # Validation gate
                if motion_ratio < 0.06:
                    # baseline invalid → restart calibration
                    self._baseline_samples.clear()
                    self._min_width = None
                    self._max_width = None
                    self._start_time = time.time()

                    return EngineOutput(
                        state=State.INIT,
                        baseline_locked=False,
                        baseline_width=None,
                        ratio_raw=None,
                        ratio_smooth=None,
                        zone=Zone.UNKNOWN,
                        zone_changed=False,
                        message_key="ADJUST_POSTURE"
                    )

                # Accept baseline
                self._baseline_width = median_w
                self._baseline_locked = True
                self._ratio_ema = 1.0
                self._last_zone = Zone.SAFE
                self._last_zone_change_t = time.time()


        # ACTIVE: baseline locked → compute ratio
        if (not self._baseline_locked) or (self._baseline_width is None):
            # Safety net: should never compute ratio without baseline
            return EngineOutput(
                state=State.INIT,
                baseline_locked=False,
                baseline_width=None,
                ratio_raw=None,
                ratio_smooth=None,
                zone=Zone.UNKNOWN,
                zone_changed=False,
                message_key="CALIBRATING"
            )

        ratio_raw = float(width_px / self._baseline_width) if width_px > 1e-6 else None

        if ratio_raw is None:
            return EngineOutput(
                state=State.PAUSED_UNSTABLE,
                baseline_locked=True,
                baseline_width=self._baseline_width,
                ratio_raw=None,
                ratio_smooth=self._ratio_ema,
                zone=self._last_zone,
                zone_changed=False,
                message_key="UNSTABLE"
            )

        # EMA update (only when stable)
        if self._ratio_ema is None:
            self._ratio_ema = ratio_raw
        else:
            a = self.ema_alpha
            self._ratio_ema = a * ratio_raw + (1 - a) * self._ratio_ema

        proposed_zone = self._classify_zone(self._ratio_ema)
        zone, changed = self._zone_with_hold(proposed_zone)

        key = "SAFE" if zone == Zone.SAFE else ("CLOSE" if zone == Zone.CLOSE else ("DANGER" if zone == Zone.DANGER else "UNSTABLE"))

        return EngineOutput(
            state=State.ACTIVE,
            baseline_locked=True,
            baseline_width=self._baseline_width,
            ratio_raw=ratio_raw,
            ratio_smooth=self._ratio_ema,
            zone=zone,
            zone_changed=changed,
            message_key=key
        )
