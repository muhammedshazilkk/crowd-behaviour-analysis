# violence_detection_with_yolo.py
import cv2, os, time, argparse
import numpy as np
from ultralytics import YOLO

# -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0', help='0 for webcam or path to video file')
args = parser.parse_args()
SOURCE = 0 if args.source == '0' else args.source

# -------------------
# Params (tune these)
RESIZE_WIDTH = 640
THRESHOLD_SCORE = 0.15
CONSEC_FRAMES = 6
MIN_AREA = 300
DURATION_SAVE = 2
RUN_YOLO_EVERY = 4     # run YOLO every 4 frames
YOLO_CONF = 0.35

# Weapon labels to check (example). Update if your model uses different names.
WEAPON_LABELS = {'knife', 'scissors', 'fork'}  # COCO doesn't have gun; update later with custom model

# -------------------
if not os.path.exists('alerts'):
    os.makedirs('alerts')

# init YOLO
print("Loading YOLO model...")
yolo = YOLO('yolov8n.pt')
print("YOLO loaded. Class names:", yolo.model.names)

# Video capture
cap = cv2.VideoCapture(SOURCE)
time.sleep(0.3)
if not cap.isOpened():
    print("ERROR: cannot open source", SOURCE)
    exit(1)

ret, frame = cap.read()
if not ret:
    print("ERROR: empty input.")
    cap.release()
    exit(1)

h, w = frame.shape[:2]
ratio = RESIZE_WIDTH / w
frame_size = (RESIZE_WIDTH, int(h * ratio))

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)

alert_counter = 0
alert_active = False
alert_start = None
alert_frames = []
frame_idx = 0
alerts_saved = 0

print("Starting main loop. Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame = cv2.resize(frame, frame_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_score = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        motion_score += area
    motion_score = min(1.0, motion_score / (frame_size[0] * frame_size[1]))

    # detect via YOLO every RUN_YOLO_EVERY frames
    weapon_detected = False
    yolo_annotations = None
    if frame_idx % RUN_YOLO_EVERY == 0:
        results = yolo(frame, conf=YOLO_CONF, verbose=False)
        # results[0] contains detection; use .boxes to iterate
        if len(results) > 0:
            boxes = results[0].boxes
            # draw and check for weapon labels
            annotated = results[0].plot()  # annotated image
            yolo_annotations = annotated
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = yolo.model.names[cls]
                # check if label in weapon list
                if name.lower() in WEAPON_LABELS and conf >= YOLO_CONF:
                    weapon_detected = True
        else:
            yolo_annotations = frame.copy()

    # simple fusion: treat high motion as suspicious
    if motion_score > THRESHOLD_SCORE:
        alert_counter += 1
    else:
        alert_counter = 0

    if alert_counter >= CONSEC_FRAMES or weapon_detected:
        if not alert_active:
            alert_active = True
            alert_start = time.time()
            print(f"ALERT: triggered at frame {frame_idx} (motion={motion_score:.3f}, weapon={weapon_detected})")

    # record frames during alert
    if alert_active:
        # prefer annotated frame for saving
        if yolo_annotations is not None:
            alert_frames.append(yolo_annotations.copy())
        else:
            alert_frames.append(frame.copy())

        if time.time() - alert_start >= DURATION_SAVE:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            outname = f"alerts/alert_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(outname, fourcc, 20.0, frame_size)
            for f in alert_frames:
                out.write(f)
            out.release()
            alerts_saved += 1
            print("Saved alert clip:", outname)
            alert_frames = []
            alert_active = False
            alert_counter = 0
            weapon_detected = False

    # display annotated (if available) else original with overlay
    display_frame = yolo_annotations if yolo_annotations is not None else frame.copy()
    status = "ALERT" if alert_active else "NORMAL"
    color = (0,0,255) if alert_active else (0,255,0)
    cv2.putText(display_frame, f"Status: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display_frame, f"Score: {motion_score:.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if weapon_detected:
        cv2.putText(display_frame, "WEAPON DETECTED", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Crowd Monitor (Motion + YOLO)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Alerts saved:", alerts_saved)
