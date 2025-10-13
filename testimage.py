import cv2

cap = cv2.VideoCapture(0)  # try 0 first
if not cap.isOpened():
    print("❌ Cannot open camera with index 0")
else:
    print("✅ Camera opened successfully")
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
    else:
        print("✅ Frame captured:", frame.shape)
cap.release()
