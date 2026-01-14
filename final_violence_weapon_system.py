import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
from collections import defaultdict, deque
import argparse
import requests

# =========================================================
# TELEGRAM CONFIG
# =========================================================
TELEGRAM_BOT_TOKEN = "8393591405:AAGs6mj4KDaxefBVxv7d6qKegy86WEWAOgk"
TELEGRAM_CHAT_ID = "1548646865"

def send_telegram_alert(message, image_path=None):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )

        if image_path:
            with open(image_path, "rb") as img:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    files={"photo": img},
                    data={"chat_id": TELEGRAM_CHAT_ID}
                )
    except Exception as e:
        print("Telegram error:", e)

# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="0",
                    help="0 for webcam OR path to video")
args = parser.parse_args()

SOURCE = 0 if args.source == "0" else args.source

# =========================================================
# PERFORMANCE
# =========================================================
FRAME_SKIP = 3
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# =========================================================
# ALERT PARAMETERS
# =========================================================
MOTION_THRESHOLD = 0.15
SPEED_THRESHOLD = 25
ALERT_FRAMES = 6
ALERT_COOLDOWN = 5   # seconds

# =========================================================
# LOAD MODELS
# =========================================================
print("Loading models...")
person_model = YOLO("yolov8n.pt")
weapon_model = YOLO("best_weapon.pt")
print("Models loaded ✅")

# =========================================================
# VIDEO
# =========================================================
cap = cv2.VideoCapture(SOURCE)
ret, prev = cap.read()
if not ret:
    print("❌ Cannot read source")
    exit()

prev = cv2.resize(prev, (TARGET_WIDTH, TARGET_HEIGHT))
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

os.makedirs("alerts", exist_ok=True)

# =========================================================
# STATE VARIABLES
# =========================================================
track_history = defaultdict(lambda: deque(maxlen=10))
alert_counter = 0
alert_active = False
last_alert_time = 0
alerts_saved = 0

# =========================================================
# HELPERS
# =========================================================
def compute_speed(points):
    if len(points) < 2:
        return 0
    (x1, y1), (x2, y2) = points[-2], points[-1]
    return ((x2-x1)**2 + (y2-y1)**2) ** 0.5

cached_boxes = []
cached_status = "NORMAL"
cached_color = (0,255,0)

frame_count = 0
fps, fps_counter = 0.0, 0
fps_start = time.time()

print("System running. Press 'q' to exit.")

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    display = frame.copy()
    frame_count += 1

    # ================= FRAME SKIP =================
    if frame_count % FRAME_SKIP != 0:
        for (x1,y1,x2,y2,label,color) in cached_boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            cv2.putText(display,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        cv2.putText(display,f"STATUS: {cached_status}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,cached_color,3)
        cv2.putText(display,f"FPS: {fps:.1f}",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.imshow("Crowd Behaviour Analysis", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    cached_boxes.clear()

    # ================= MOTION =================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
    motion_score = np.sum(thresh)/thresh.size
    prev_gray = gray.copy()

    # ================= PERSON TRACK =================
    fast_movement = False
    person_results = person_model.track(frame, conf=0.4,
                                        persist=True, verbose=False)

    if person_results:
        for box in person_results[0].boxes:
            if box.id is None: continue
            cls = int(box.cls[0])
            if person_model.names[cls] != "person": continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            track_history[track_id].append((cx,cy))

            speed = compute_speed(track_history[track_id])
            if speed > SPEED_THRESHOLD:
                fast_movement = True

            cached_boxes.append(
                (x1,y1,x2,y2,f"ID {track_id} {conf:.2f}",(255,0,0))
            )

    # ================= WEAPON =================
    weapon_found = False
    weapon_results = weapon_model(frame, conf=0.15, verbose=False)
    if weapon_results:
        for box in weapon_results[0].boxes:
            cls = int(box.cls[0])
            name = weapon_model.names[cls].lower()
            if name in ["weapon","gun","knife"]:
                weapon_found = True
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cached_boxes.append(
                    (x1,y1,x2,y2,f"{name.upper()}",(0,0,255))
                )

    # ================= ALERT LOGIC =================
    alert_reasons = []
    if motion_score > MOTION_THRESHOLD: alert_reasons.append("High motion")
    if fast_movement: alert_reasons.append("Fast movement")
    if weapon_found: alert_reasons.append("Weapon detected")

    if alert_reasons:
        alert_counter += 1
    else:
        alert_counter = max(0, alert_counter - 1)

    # ================= ALERT TRIGGER =================
    if (alert_counter >= ALERT_FRAMES and not alert_active and
        time.time() - last_alert_time > ALERT_COOLDOWN):

        alert_active = True
        last_alert_time = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = f"alerts/alert_{ts}.jpg"
        cv2.imwrite(img_path, display)

        msg = (
            "🚨 SECURITY ALERT 🚨\n"
            f"Time: {ts}\n"
            f"Reasons: {', '.join(alert_reasons)}"
        )

        print(msg)
        send_telegram_alert(msg, img_path)
        alerts_saved += 1
        alert_counter = 0

    if alert_active and time.time() - last_alert_time > ALERT_COOLDOWN:
        alert_active = False

    # ================= STATUS =================
    cached_status = "ALERT" if alert_active else "NORMAL"
    cached_color = (0,0,255) if alert_active else (0,255,0)

    # ================= FPS =================
    fps_counter += 1
    if fps_counter >= 10:
        fps = fps_counter / (time.time() - fps_start)
        fps_start = time.time()
        fps_counter = 0

    # ================= DISPLAY =================
    for (x1,y1,x2,y2,label,color) in cached_boxes:
        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
        cv2.putText(display,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.putText(display,f"STATUS: {cached_status}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,cached_color,3)
    cv2.putText(display,f"FPS: {fps:.1f}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("Crowd Behaviour Analysis", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# =========================================================
# CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()
print("✅ System stopped. Alerts sent:", alerts_saved)
