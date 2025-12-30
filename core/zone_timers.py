# core/zone_timers.py
from dataclasses import dataclass
from typing import Optional
import time

from core.stage2_engine import Zone


@dataclass
class TimerOutput:
    danger_continuous_s: float
    danger_cumulative_s_window: float
    danger_triggered: bool


class ZoneTimers:
    """
    Tracks:
    - continuous time in current zone (for DANGER)
    - cumulative DANGER time in a rolling window (e.g., last 5 minutes)

    Rules:
    - pause when no face or unstable (don't advance timers)
    - reset continuous on zone exit
    """

    def __init__(self, danger_continuous_trigger_s: float = 6.0, cumulative_trigger_s: float = 30.0, window_s: float = 300.0):
        self.danger_continuous_trigger_s = danger_continuous_trigger_s
        self.cumulative_trigger_s = cumulative_trigger_s
        self.window_s = window_s

        self._last_t: Optional[float] = None
        self._current_zone: Zone = Zone.UNKNOWN

        self._danger_continuous = 0.0

        # rolling window accumulator: keep (timestamp, dt) contributions
        self._danger_events = []  # list of (t, dt)

    def reset(self):
        self._last_t = None
        self._current_zone = Zone.UNKNOWN
        self._danger_continuous = 0.0
        self._danger_events.clear()

    def update(self, zone: Zone, tracking_active: bool) -> TimerOutput:
        now = time.time()

        if not tracking_active:
            # Pause timers by not updating dt
            self._last_t = now
            return TimerOutput(self._danger_continuous, self._cumulative(now), False)

        if self._last_t is None:
            self._last_t = now
            self._current_zone = zone
            return TimerOutput(self._danger_continuous, self._cumulative(now), False)

        dt = now - self._last_t
        self._last_t = now

        # continuous danger
        if zone == Zone.DANGER:
            self._danger_continuous += dt
            self._danger_events.append((now, dt))
        else:
            self._danger_continuous = 0.0

        # update current zone
        self._current_zone = zone

        cum = self._cumulative(now)
        triggered = (self._danger_continuous >= self.danger_continuous_trigger_s) or (cum >= self.cumulative_trigger_s)

        return TimerOutput(self._danger_continuous, cum, triggered)

    def _cumulative(self, now: float) -> float:
        # drop old contributions beyond window
        cutoff = now - self.window_s
        self._danger_events = [(t, dt) for (t, dt) in self._danger_events if t >= cutoff]
        return sum(dt for (_, dt) in self._danger_events)
