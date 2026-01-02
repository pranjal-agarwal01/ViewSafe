import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
import time

# Load pre-trained depth model (downloads ~95MB first time, then cached)
depth_estimator = pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf")  # Small for speed

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def estimate_depth_from_face(frame):
    # Detect face
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if not results.detections:
        return None
    
    # Get face bounding box (simplified; use landmarks for eyes if needed)
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    face_crop = frame[y:y+bh, x:x+bw]
    
    if face_crop.size == 0:
        return None
    
    # Predict depth map (outputs PIL image with depths)
    pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    depth = depth_estimator(pil_image)["depth"]
    
    # Convert to numpy, resize to match crop, get median depth (in relative units)
    depth_np = np.array(depth)
    depth_np = cv2.resize(depth_np, (bw, bh))
    
    # Scale to metric (Depth Anything is pre-scaled; median face depth ≈ distance to camera)
    # For absolute cm, use model's built-in affine transform (simplified here; tune if needed)
    median_depth = np.median(depth_np) * 100  # Rough scale to cm (calibrate once dev-side for your avg webcam)
    
    return median_depth

# Main loop
cap = cv2.VideoCapture(0)
warn_time = 0
warning_threshold = 50  # cm
hysteresis = 5  # seconds too close before warning

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    depth = estimate_depth_from_face(frame)
    
    if depth:
        cv2.putText(frame, f"Distance: {depth:.0f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if depth < warning_threshold:
            if time.time() - warn_time > hysteresis:
                # Warning: Full-screen overlay or notification
                cv2.putText(frame, "TOO CLOSE! Move back 20cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Optional: playsound or plyer notification
                warn_time = time.time()
    
    cv2.imshow('Eye Distance Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()