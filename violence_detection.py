import cv2
import numpy as np
import os
import time
import argparse

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="0",
                    help="0 for webcam or path to video file")
args = parser.parse_args()

# Convert webcam string to int
if args.source == "0":
    SOURCE = 0
else:
    SOURCE = args.source

# ---------------------------
# Create alerts folder
# ---------------------------
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# ---------------------------
# Parameters
# ---------------------------
RESIZE_WIDTH = 640
THRESHOLD_SCORE = 0.5          # Score for abnormal detection
CONSEC_FRAMES = 10             # Sustained frames threshold
MIN_AREA = 500                 # Minimum movement area
DURATION_SAVE = 2              # seconds of alert recording

# ---------------------------
# Video Capture
# ---------------------------
cap = cv2.VideoCapture(SOURCE)
time.sleep(1)

ret, frame = cap.read()
if not ret:
    print("❌ Error: Cannot open video source.")
    exit()

# Resize frame
h, w = frame.shape[:2]
ratio = RESIZE_WIDTH / w
frame_size = (RESIZE_WIDTH, int(h * ratio))

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)

alert_counter = 0
alert_active = False
alert_start_time = None

# ---------------------------
# Helper to save alert clip
# ---------------------------
def save_alert_clip(frames_list):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outname = f"alerts/alert_{timestamp}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outname, fourcc, 20.0,
                          (frame_size[0], frame_size[1]))

    for f in frames_list:
        out.write(f)
    out.release()

    print(f"⚠ Alert clip saved: {outname}")


# ---------------------------
# Main Loop
# ---------------------------
alert_frames = []
alert_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, frame_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # Threshold to remove noise
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find movement regions
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    motion_score = 0
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        motion_score += cv2.contourArea(c)

    # Normalize score
    motion_score = motion_score / (frame_size[0] * frame_size[1])
    motion_score = min(motion_score, 1.0)  # Clamp

    # Check for abnormal activity
    if motion_score > THRESHOLD_SCORE:
        alert_counter += 1
    else:
        alert_counter = 0

    if alert_counter >= CONSEC_FRAMES:
        alert_active = True
        if alert_start_time is None:
            alert_start_time = time.time()

    # Record frames during alert
    if alert_active:
        alert_frames.append(frame)
        alert_frame_count += 1

        # Stop after duration
        if time.time() - alert_start_time >= DURATION_SAVE:
            save_alert_clip(alert_frames)
            alert_frames = []
            alert_active = False
            alert_start_time = None
            alert_frame_count = 0

    # ---------------------------
    # Display Video
    # ---------------------------
    status_text = "ALERT - SUSPICIOUS ACTIVITY!" if alert_active else "NORMAL"
    color = (0,0,255) if alert_active else (0,255,0)

    cv2.putText(frame, f"Status: {status_text}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Score: {motion_score:.4f}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Abnormal Activity Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
