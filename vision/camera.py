# vision/camera.py
import cv2
from typing import Optional, Tuple

class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)  # DSHOW = more stable on Windows
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Try changing index (0/1) or close other apps using camera.")

        # Set desired properties (not always guaranteed by webcam driver)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # Reduce buffering to lower latency (some backends ignore this)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[any]]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
