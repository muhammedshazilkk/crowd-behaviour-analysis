# debug_cam_test.py
import cv2, time, sys
print("DEBUG: script started")
print("Python Version:", sys.version.split()[0])
try:
    # try several camera indexes (0..3)
    found = False
    for cam_idx in range(4):
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)  # use DirectShow on Windows
        time.sleep(0.3)
        opened = cap.isOpened()
        print(f"DEBUG: camera index {cam_idx} isOpened -> {opened}")
        if opened:
            found = True
            ret, frame = cap.read()
            print(f"DEBUG: first read ret -> {ret}")
            if ret:
                print("DEBUG: frame shape ->", frame.shape)
                cv2.imshow("DEBUG_FRAME", frame)
                print("DEBUG: If a window opened showing the camera frame, click on it and press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cap.release()
            break
        cap.release()

    if not found:
        print("DEBUG: No camera index (0-3) opened. Check camera drivers and privacy settings.")
    print("DEBUG: script finished")
except Exception as e:
    print("DEBUG: exception:", e)
