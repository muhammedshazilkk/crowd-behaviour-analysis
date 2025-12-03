# yolo_video_demo.py
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')   # use yolov8n for speed
cap = cv2.VideoCapture(0)    # webcam index 0

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference (on the current frame)
    results = model(frame, verbose=False)   # verbose False to reduce logs

    # Annotated frame
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
