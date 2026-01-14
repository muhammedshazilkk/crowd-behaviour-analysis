# violence_detection_with_face.py
# Motion + YOLOv8 + FaceID (facenet-pytorch)
# Smooth video + Face alert + Telegram

import os
import time
import argparse
import pickle
import json
import cv2
import numpy as np
import torch
import requests
import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# =========================================================
# TELEGRAM CONFIG
# =========================================================
TELEGRAM_BOT_TOKEN = "8393591405:AAGs6mj4KDaxefBVxv7d6qKegy86WEWAOgk"
TELEGRAM_CHAT_ID = "1548646865"

def send_telegram_alert(message, image_path):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=5
        )
        with open(image_path, "rb") as img:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                files={"photo": img},
                data={"chat_id": TELEGRAM_CHAT_ID},
                timeout=5
            )
    except Exception as e:
        print("Telegram error:", e)

def async_face_alert(name, frame):
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = f"alerts/{name}_{ts}.jpg"
    cv2.imwrite(img_path, frame)
    msg = f"🚨 PERSON IDENTIFIED 🚨\nName: {name}\nTime: {ts}"
    threading.Thread(
        target=send_telegram_alert,
        args=(msg, img_path),
        daemon=True
    ).start()

# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0')
args = parser.parse_args()
SOURCE = 0 if args.source == '0' else args.source

# =========================================================
# PERFORMANCE
# =========================================================
FRAME_SKIP = 4
RESIZE_WIDTH = 640
STREAM_WIDTH = 800

# =========================================================
# FACE PARAMETERS
# =========================================================
ID_TOLERANCE = 0.95
RECOG_REPEAT = 3
FACE_ALERT_COOLDOWN = 10  # seconds

# =========================================================
# FOLDERS
# =========================================================
os.makedirs('alerts', exist_ok=True)
os.makedirs('stream', exist_ok=True)

# =========================================================
# LOAD MODELS
# =========================================================
print("Loading YOLO...")
yolo = YOLO('yolov8n.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("FaceNet device:", device)

mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# =========================================================
# LOAD KNOWN FACES
# =========================================================
with open("faces/known_facenet.pkl", "rb") as f:
    data = pickle.load(f)

known_embs = data['embeddings']
known_names = data['names']
print("Loaded known faces:", known_names)

# =========================================================
# VIDEO
# =========================================================
cap = cv2.VideoCapture(SOURCE)
ret, frame = cap.read()
if not ret:
    print("❌ Cannot open source")
    exit()

h, w = frame.shape[:2]
ratio = RESIZE_WIDTH / w
frame_size = (RESIZE_WIDTH, int(h * ratio))

# =========================================================
# STATE
# =========================================================
frame_idx = 0
recog_counts = {}
last_alert_time = {}

cached_boxes = []
cached_name = None
cached_status = "NORMAL"
cached_color = (0,255,0)

print("System running. Press 'q' to exit.")

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame = cv2.resize(frame, frame_size)
    display = frame.copy()

    # ---------- DRAW CACHED FRAMES
    if frame_idx % FRAME_SKIP != 0:
        for (x1,y1,x2,y2,label,color) in cached_boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            cv2.putText(display,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        cv2.putText(display,f"STATUS: {cached_status}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,cached_color,2)

        if cached_name:
            cv2.putText(display,f"ID: {cached_name}",(10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Crowd Monitor - Face ID", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    cached_boxes.clear()
    cached_name = None
    cached_status = "NORMAL"
    cached_color = (0,255,0)

    # ---------- PERSON DETECTION
    results = yolo(frame, conf=0.35, verbose=False)

    if results:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if yolo.model.names[cls] != "person":
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cached_boxes.append((x1,y1,x2,y2,"PERSON",(255,0,0)))

            crop = frame[y1:y2, x1:x2]
            try:
                from PIL import Image
                face = mtcnn(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if face is None:
                    continue

                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]

                dists = np.linalg.norm(known_embs - emb, axis=1)
                idx = int(np.argmin(dists))
                dist = float(dists[idx])

                if dist < ID_TOLERANCE:
                    name = known_names[idx]
                    recog_counts[name] = recog_counts.get(name, 0) + 1

                    if recog_counts[name] >= RECOG_REPEAT:
                        now = time.time()
                        if now - last_alert_time.get(name, 0) > FACE_ALERT_COOLDOWN:
                            last_alert_time[name] = now
                            cached_name = name
                            cached_status = "PERSON IDENTIFIED"
                            cached_color = (0,0,255)

                            print(f"Identified: {name}")
                            async_face_alert(name, display.copy())

                        recog_counts[name] = 0
            except:
                pass

    # ---------- STREAM IMAGE
    cv2.imwrite("stream/stream.jpg", display)

    # ---------- DISPLAY
    for (x1,y1,x2,y2,label,color) in cached_boxes:
        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
        cv2.putText(display,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.putText(display,f"STATUS: {cached_status}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,cached_color,2)

    if cached_name:
        cv2.putText(display,f"ID: {cached_name}",(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Crowd Monitor - Face ID", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped.")
