# app.py
import cv2
import numpy as np

from vision.camera import Camera
from vision.face_tracker import FaceTracker
from ui.overlay import draw_top_right_box
from utils.timing import FPSTracker

def draw_landmarks(frame, pts, draw_full_mesh: bool = False):
    if pts is None:
        return

    if draw_full_mesh:
        for (x, y) in pts:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    else:
        # Draw only key points (eye corners) for performance
        cv2.circle(frame, tuple(pts[FaceTracker.LEFT_EYE_OUTER]), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(pts[FaceTracker.RIGHT_EYE_OUTER]), 3, (0, 255, 0), -1)
        cv2.line(frame, tuple(pts[FaceTracker.LEFT_EYE_OUTER]), tuple(pts[FaceTracker.RIGHT_EYE_OUTER]), (0, 255, 0), 2)

def draw_bbox(frame, bbox):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

def main():
    cam = Camera(index=0, width=640, height=480, fps=30)
    tracker = FaceTracker(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        jump_ratio_threshold=0.18
    )
    fps = FPSTracker()

    draw_full_mesh = False
    mirror = True

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            draw_top_right_box(frame, ["Camera read failed ❌"], box_color=(60, 0, 0))
            cv2.imshow("ViewSafe Dev", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        face = tracker.process(frame)

        # Debug visuals
        if face.detected and face.landmarks_px is not None:
            draw_bbox(frame, face.bbox)
            draw_landmarks(frame, face.landmarks_px, draw_full_mesh=draw_full_mesh)

        # FPS
        fps_val = fps.tick()

        # UI overlay text
        if not face.detected:
            lines = [
                "Face: ❌ not detected",
                f"FPS: {fps_val:.1f}"
            ]
            draw_top_right_box(frame, lines, box_color=(60, 0, 0))
        else:
            # detected but may be rejected jump
            width_text = f"Width(px): {face.width_px:.1f}" if face.width_px is not None else "Width(px): --"
            lines = [
                "Face: ✅ detected",
                width_text,
                f"State: {face.reason}",
                f"FPS: {fps_val:.1f}"
            ]
            draw_top_right_box(frame, lines, box_color=(0, 60, 0))

        cv2.imshow("ViewSafe Dev", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('m'), ord('M')):
            draw_full_mesh = not draw_full_mesh
        elif key in (ord('f'), ord('F')):
            mirror = not mirror

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
