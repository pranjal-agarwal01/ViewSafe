# vision/face_tracker.py
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class FaceResult:
    detected: bool
    face_index: Optional[int]
    landmarks_px: Optional[np.ndarray]  # shape (468, 2) int
    bbox: Optional[Tuple[int, int, int, int]]  # (x1,y1,x2,y2)
    width_px: Optional[float]
    reason: str

class FaceTracker:
    """
    Real-world stage-1 tracker:
    - Runs FaceMesh each frame (offline)
    - Handles multiple faces (largest + stability)
    - Provides a stable 'width_px' signal using outer eye corners
    - Rejects flicker/jumps
    """

    # Outer eye corners (FaceMesh indices)
    # These are widely used stable eye-corner points in MediaPipe FaceMesh.
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(
        self,
        max_num_faces: int = 2,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        jump_ratio_threshold: float = 0.18,   # reject sudden changes > 18% vs last accepted
        center_lock_weight: float = 0.7       # how strongly we stick to previous face center
    ):
        self.mp_face = mp.solutions.face_mesh
        self.mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.last_center: Optional[Tuple[float, float]] = None
        self.last_width: Optional[float] = None
        self.jump_ratio_threshold = jump_ratio_threshold
        self.center_lock_weight = center_lock_weight

    def _landmarks_to_px(self, landmarks, w: int, h: int) -> np.ndarray:
        pts = np.zeros((468, 2), dtype=np.int32)
        for i, lm in enumerate(landmarks.landmark[:468]):
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts[i] = (x, y)
        return pts

    def _bbox_from_pts(self, pts: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = int(np.clip(np.min(pts[:, 0]), 0, w - 1))
        y1 = int(np.clip(np.min(pts[:, 1]), 0, h - 1))
        x2 = int(np.clip(np.max(pts[:, 0]), 0, w - 1))
        y2 = int(np.clip(np.max(pts[:, 1]), 0, h - 1))
        return (x1, y1, x2, y2)

    def _face_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _bbox_area(self, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _choose_face_index(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        frame_center: Tuple[float, float]
    ) -> int:
        """
        Choose a face consistently:
        Score = (area normalized) - (distance to last_center / frame diag) * lock_weight
        If no last_center, choose largest (or closest to center as tie-break).
        """
        areas = np.array([self._bbox_area(b) for b in bboxes], dtype=np.float32)
        if np.max(areas) <= 0:
            return 0

        areas_norm = areas / (np.max(areas) + 1e-6)

        centers = np.array([self._face_center(b) for b in bboxes], dtype=np.float32)
        fc = np.array(frame_center, dtype=np.float32)
        dist_to_frame_center = np.linalg.norm(centers - fc, axis=1)

        if self.last_center is None:
            # Largest, tie-breaker: closer to frame center
            best = int(np.argmax(areas_norm - 0.05 * (dist_to_frame_center / (np.max(dist_to_frame_center) + 1e-6))))
            return best

        last = np.array(self.last_center, dtype=np.float32)
        dist_to_last = np.linalg.norm(centers - last, axis=1)

        # Normalize distance by frame diagonal-ish magnitude
        dist_norm = dist_to_last / (np.max(dist_to_last) + 1e-6)

        score = areas_norm - (self.center_lock_weight * dist_norm)
        return int(np.argmax(score))

    def _compute_width(self, pts: np.ndarray) -> float:
        a = pts[self.LEFT_EYE_OUTER].astype(np.float32)
        b = pts[self.RIGHT_EYE_OUTER].astype(np.float32)
        return float(np.linalg.norm(a - b))

    def process(self, frame_bgr) -> FaceResult:
        h, w = frame_bgr.shape[:2]
        frame_center = (w / 2.0, h / 2.0)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            # Reset lightly (don’t nuke last_center; helps reacquire without wild jumps)
            return FaceResult(False, None, None, None, None, "No face detected")

        # Convert all faces to pixel landmarks + bbox
        faces_pts = []
        faces_bbox = []
        for fl in result.multi_face_landmarks:
            pts = self._landmarks_to_px(fl, w, h)
            bbox = self._bbox_from_pts(pts, w, h)
            faces_pts.append(pts)
            faces_bbox.append(bbox)

        # Choose one face consistently
        chosen_idx = self._choose_face_index(faces_bbox, frame_center)
        pts = faces_pts[chosen_idx]
        bbox = faces_bbox[chosen_idx]
        center = self._face_center(bbox)

        # Compute stable width signal (eye-corner distance)
        width_px = self._compute_width(pts)

        # Jump rejection (flicker / face switch / glitch)
        if self.last_width is not None and self.last_width > 1e-3:
            ratio = abs(width_px - self.last_width) / self.last_width
            if ratio > self.jump_ratio_threshold:
                # Ignore this frame, keep last state
                return FaceResult(True, chosen_idx, None, bbox, None, f"Rejected jump ({ratio*100:.1f}%)")

        # Accept this frame
        self.last_center = center
        self.last_width = width_px

        return FaceResult(True, chosen_idx, pts, bbox, width_px, "OK")
