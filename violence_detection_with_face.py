# violence_detection_with_face.py
# Integrated motion + YOLOv8 + FaceID (facenet-pytorch)
# Saves annotated latest frame to stream/stream.jpg for dashboard
# Appends structured JSON logs to alerts/log.jsonl

import os
import time
import argparse
import pickle
import json
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# ---------------- args
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0', help='0 for webcam or path to video')
args = parser.parse_args()
SOURCE = 0 if args.source == '0' else args.source

# ---------------- params (tune as needed)
RESIZE_WIDTH = 640
THRESHOLD_SCORE = 0.15
CONSEC_FRAMES = 6
MIN_AREA = 300
DURATION_SAVE = 2         # seconds for saved alert clip
RUN_YOLO_EVERY = 4        # run object detection every N frames
YOLO_CONF = 0.35
ID_TOLERANCE = 0.95       # facenet euclidean distance threshold (tune 0.8-1.1)
RECOG_REPEAT = 3          # require same ID for N frames to confirm
STREAM_WIDTH = 800        # width of saved stream image

# ---------------- folders
if not os.path.exists('alerts'):
    os.makedirs('alerts')
if not os.path.exists('stream'):
    os.makedirs('stream')

LOG_FILE = os.path.join('alerts', 'log.jsonl')

# ---------------- load YOLO
print("Loading YOLO...")
yolo = YOLO('yolov8n.pt')   # model will download if not present
print("YOLO loaded. Classes:", yolo.model.names)

# ---------------- load facenet models & known embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device for facenet:", device)
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

enc_file = "faces/known_facenet.pkl"
if not os.path.exists(enc_file):
    print("ERROR: embeddings file not found. Run encode_known_faces_facenet.py first.")
    exit(1)

with open(enc_file, 'rb') as f:
    data = pickle.load(f)
known_embs = data['embeddings']  # numpy array (N,512)
known_names = data['names']

print("Loaded known embeddings:", len(known_names))

# ---------------- Video capture
cap = cv2.VideoCapture(SOURCE)
time.sleep(0.3)
if not cap.isOpened():
    print("ERROR: cannot open source", SOURCE); exit(1)

ret, frame = cap.read()
if not ret:
    print("ERROR: empty source"); cap.release(); exit(1)

h, w = frame.shape[:2]
ratio = RESIZE_WIDTH / w
frame_size = (RESIZE_WIDTH, int(h * ratio))

fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)

# recognition state
recog_counts = {}
alert_counter = 0
alert_active = False
alert_start = None
alert_frames = []
frame_idx = 0
alerts_saved = 0

def append_log(entry: dict):
    """Append a JSON line to alerts/log.jsonl"""
    try:
        with open(LOG_FILE, "a") as lf:
            lf.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("Log write error:", e)

print("Starting main loop. Press 'q' to quit.")
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

    # motion score
    motion_score = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA: continue
        motion_score += area
    motion_score = min(1.0, motion_score / (frame_size[0] * frame_size[1]))

    # YOLO inference every N frames
    weapon_detected = False
    yolo_annotations = None
    person_bboxes = []

    if frame_idx % RUN_YOLO_EVERY == 0:
        results = yolo(frame, conf=YOLO_CONF, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            yolo_annotations = results[0].plot()
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = yolo.model.names[cls].lower()
                xy = box.xyxy[0].tolist()
                x1,y1,x2,y2 = map(int, xy)
                if name == 'person':
                    person_bboxes.append((x1,y1,x2,y2))
                if name in {'knife'} and conf >= YOLO_CONF:
                    weapon_detected = True
        else:
            yolo_annotations = frame.copy()
    else:
        yolo_annotations = frame.copy()

    # update alert counter from motion
    if motion_score > THRESHOLD_SCORE:
        alert_counter += 1
    else:
        alert_counter = 0

    # face recognition on detected person crops
    recognized_this_frame = []
    for (x1,y1,x2,y2) in person_bboxes:
        pad = 8
        x1c = max(0, x1-pad); y1c = max(0, y1-pad)
        x2c = min(frame.shape[1], x2+pad); y2c = min(frame.shape[0], y2+pad)
        crop = frame[y1c:y2c, x1c:x2c]
        try:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            face_tensor = mtcnn(pil)
            if face_tensor is None:
                continue
            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
            dists = np.linalg.norm(known_embs - emb, axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist < ID_TOLERANCE:
                name = known_names[best_idx]
                recognized_this_frame.append((name, best_dist, (x1c,y1c,x2c,y2c)))
        except Exception as e:
            # keep running even if recognition fails for one crop
            # print("Face recognition error:", e)
            continue

    # aggregate recognition counts
    confirmed_name = None
    if recognized_this_frame:
        top = sorted(recognized_this_frame, key=lambda x: x[1])[0]
        nm = top[0]
        recog_counts[nm] = recog_counts.get(nm, 0) + 1
        for k in list(recog_counts.keys()):
            if k != nm:
                recog_counts[k] = 0
        if recog_counts[nm] >= RECOG_REPEAT:
            confirmed_name = nm
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap = f"alerts/{nm}_{ts}.jpg"
            try:
                cv2.imwrite(snap, frame)
                log_entry = {"time": ts, "event": "person_identified", "name": nm, "image": snap}
                append_log(log_entry)
                print("Identified:", nm, "saved snapshot", snap)
            except Exception as e:
                print("Snapshot save error:", e)
            recog_counts[nm] = 0

    # decide alert
    if alert_counter >= CONSEC_FRAMES or weapon_detected:
        if not alert_active:
            alert_active = True
            alert_start = time.time()
            print(f"ALERT: triggered at frame {frame_idx} (motion={motion_score:.3f}, weapon={weapon_detected})")

    # record frames during alert
    if alert_active:
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
            # log video alert
            log_entry = {"time": timestamp, "event": "motion_alert", "motion_score": round(float(motion_score),3), "video": outname}
            append_log(log_entry)
            print("Saved alert clip:", outname)
            alert_frames = []
            alert_active = False
            alert_counter = 0

    # prepare display
    display = yolo_annotations if yolo_annotations is not None else frame.copy()
    status = "ALERT" if alert_active else "NORMAL"
    color = (0,0,255) if alert_active else (0,255,0)
    cv2.putText(display, f"Status: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f"Score: {motion_score:.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if confirmed_name:
        cv2.putText(display, f"ID: {confirmed_name}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ------------------ SAVE LATEST STREAM FRAME FOR DASHBOARD ------------------
    try:
        # write resized annotated frame for dashboard
        h_disp, w_disp = display.shape[:2]
        new_h = int(h_disp * (STREAM_WIDTH / w_disp))
        resized = cv2.resize(display, (STREAM_WIDTH, new_h))
        stream_path = os.path.join('stream', 'stream.jpg')
        cv2.imwrite(stream_path, resized)
    except Exception:
        pass

    # show window locally
    cv2.imshow("Crowd Monitor - Face ID", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
print("Done. Alerts saved:", alerts_saved)
