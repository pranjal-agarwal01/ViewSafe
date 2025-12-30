# ui/overlay.py
import cv2
from typing import Tuple

def draw_top_right_box(
    frame,
    lines,
    pad: int = 10,
    line_gap: int = 6,
    font_scale: float = 0.55,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Tuple[int, int, int] = (30, 30, 30),
    alpha: float = 0.6
):
    """
    Draws a semi-transparent overlay box with multiple lines in the top-right.
    This simulates the final UX (top-right overlay) even in dev window mode.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compute text block size
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_w = max((s[0] for s in sizes), default=0)
    text_h = sum((s[1] for s in sizes), 0) + line_gap * (len(lines) - 1)

    box_w = text_w + 2 * pad
    box_h = text_h + 2 * pad

    x2 = w - 10
    y1 = 10
    x1 = x2 - box_w
    y2 = y1 + box_h

    # Clamp
    x1 = max(0, x1)
    y2 = min(h, y2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw text
    y = y1 + pad
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.putText(frame, line, (x1 + pad, y + th), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += th + line_gap


def overlay_lines_from_key(message_key: str):
    if message_key == "CALIBRATING":
        return ["ViewSafe", "Calibrating posture…"]
    if message_key == "SAFE":
        return ["🟢 ViewSafe — Safe"]
    if message_key == "CLOSE":
        return ["🟡 ViewSafe — Too Close"]
    if message_key == "DANGER":
        return ["🔴 ViewSafe — Move Back"]
    if message_key == "NO_FACE":
        return ["ViewSafe", "Face not detected"]
    if message_key == "UNSTABLE":
        return ["ViewSafe", "Hold still…"]
    if message_key == "ADJUST_POSTURE":
        return ["ViewSafe", "Adjust posture slightly…"]
    return ["ViewSafe"]
    

