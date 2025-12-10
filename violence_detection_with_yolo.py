import cv2
import time
import argparse
import os
import numpy as np
from ultralytics import YOLO
from tracker import CentroidTracker

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="0", help="0 for webcam or video file path")
args = parser.parse_args()
SOURCE = 0 if args.source == "0" else args.source

# -----------------------------
# PARAMETERS
# -----------------------------
RESIZE_WIDTH = 640
THRESHOLD_SCORE = 0.15         # motion threshold
CONSEC_FRAMES = 6              # frames required to trigger alert
ALERT_CLIP_LENGTH = 48         # frames to save during alert

# -----------------------------
# LOAD YOLO
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO loaded.")

# -----------------------------
# TRACKER
# -----------------------------
tracker = CentroidTracker()

# -----------------------------
# VIDEO CAPTURE
# -----------------------------
cap = cv2.VideoCapture(SOURCE)

# -----------------------------
# MOTION DETECTION INIT
# -----------------------------
prev_gray = None
alert_counter = 0
recording = False
alert_frames = []
alerts_saved = 0

# -----------------------------
# STREAM & ALERT FOLDERS
# -----------------------------
os.makedirs("alerts", exist_ok=True)
os.makedirs("stream", exist_ok=True)

# -----------------------------
# FPS COUNTER
# -----------------------------
prev_time = time.time()

print("Starting main loop. Press 'q' to exit.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))
    display = frame.copy()

    # -----------------------------
    # MOTION DETECTION
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh) / thresh.size

    prev_gray = gray

    if motion_score > THRESHOLD_SCORE:
        alert_counter += 1
    else:
        alert_counter = 0

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(frame, verbose=False)[0]
    person_bboxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls_id] == "person" and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display, f"person {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -----------------------------
    # ✅ TRACKING (DAY 5 STEP 2)
    # -----------------------------
    tracked_objects = tracker.update(person_bboxes)

    for objectID, centroid in tracked_objects.items():
        text = f"Person-{objectID}"
        cv2.putText(display, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(display, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

    # -----------------------------
    # ✅ ALERT TRIGGER
    # -----------------------------
    if alert_counter >= CONSEC_FRAMES:
        if not recording:
            recording = True
            alert_frames = []
            print(f"ALERT: triggered (motion={motion_score:.3f})")

    if recording:
        alert_frames.append(display.copy())
        if len(alert_frames) >= ALERT_CLIP_LENGTH:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"alerts/alert_{ts}.avi"
            h, w, _ = alert_frames[0].shape
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))
            for f in alert_frames:
                out.write(f)
            out.release()

            print(f"Saved alert clip: {out_path}")
            alerts_saved += 1
            recording = False
            alert_counter = 0

    # -----------------------------
    # ✅ STATUS + FPS
    # -----------------------------
    status = "ALERT" if recording else "NORMAL"
    color = (0, 0, 255) if recording else (0, 255, 0)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    cv2.putText(display, f"Status: {status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f"Score: {motion_score:.3f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"FPS: {fps}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -----------------------------
    # ✅ SAVE STREAM FRAME FOR DASHBOARD
    # -----------------------------
    try:
        cv2.imwrite("stream/stream.jpg",
                    cv2.resize(display, (800, int(display.shape[0] * 800 / display.shape[1]))))
    except:
        pass

    cv2.imshow("YOLOv8 Live", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Done. Alerts saved:", alerts_saved)
