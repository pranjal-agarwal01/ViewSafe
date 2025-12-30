# utils/timing.py
import time

class FPSTracker:
    def __init__(self, smoothing: float = 0.9):
        self.prev_t = time.time()
        self.fps_ema = 0.0
        self.smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        dt = now - self.prev_t
        self.prev_t = now
        fps = 1.0 / dt if dt > 0 else 0.0

        if self.fps_ema == 0.0:
            self.fps_ema = fps
        else:
            self.fps_ema = self.smoothing * self.fps_ema + (1 - self.smoothing) * fps

        return self.fps_ema
