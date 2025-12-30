# app.py
import cv2

from vision.camera import Camera
from vision.face_tracker import FaceTracker
from utils.timing import FPSTracker

from ui.overlay import draw_top_right_box, overlay_lines_from_key
from core.stage2_engine import DistanceEngine, State, Zone
from core.zone_timers import ZoneTimers


def draw_landmarks(frame, pts, draw_full_mesh: bool = False):
    if pts is None:
        return
    if draw_full_mesh:
        for (x, y) in pts:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    else:
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

    # Stage-2 engines
    engine = DistanceEngine(
        baseline_seconds=15.0,
        min_samples=60,
        ema_alpha=0.2,
        safe_low=0.90,
        safe_high=1.10,
        close_high=1.30,
        zone_hold_seconds=0.6
    )
    timers = ZoneTimers(
        danger_continuous_trigger_s=6.0,
        cumulative_trigger_s=30.0,
        window_s=300.0
    )

    fps = FPSTracker()
    draw_full_mesh = False
    mirror = True

    # calm overlay: update only when key changes
    last_message_key = None

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        face = tracker.process(frame)

        # Stage-1 debug render
        if face.detected and face.landmarks_px is not None:
            draw_bbox(frame, face.bbox)
            draw_landmarks(frame, face.landmarks_px, draw_full_mesh=draw_full_mesh)

        # tracking_stable means: we have usable width AND not rejected
        frame_rejected = (face.reason.startswith("Rejected"))
        tracking_stable = face.detected and (face.width_px is not None) and (not frame_rejected)

        out = engine.update(
            face_detected=face.detected,
            tracking_stable=tracking_stable,
            width_px=face.width_px,
            frame_rejected=frame_rejected
        )

        # timers update only when ACTIVE (baseline locked + stable tracking)
        tracking_active = (out.state == State.ACTIVE) and tracking_stable
        t_out = timers.update(out.zone, tracking_active=tracking_active)

        # If timers trigger danger, force danger message (still calm: only changes on trigger)
        message_key = out.message_key
        if out.baseline_locked and t_out.danger_triggered:
            message_key = "DANGER"

        # overlay updates only if key changed OR during init to show calibrating
        if (message_key != last_message_key):
            last_message_key = message_key

        lines = overlay_lines_from_key(last_message_key)
        draw_top_right_box(frame, lines)

        # FPS (dev-only small text bottom-left)
        fps_val = fps.tick()
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("ViewSafe Dev", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('m'), ord('M')):
            draw_full_mesh = not draw_full_mesh
        elif key in (ord('f'), ord('F')):
            mirror = not mirror

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
