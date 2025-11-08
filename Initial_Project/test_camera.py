import cv2

# Try forcing DirectShow instead of MSMF
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot access camera")
else:
    print("✅ Camera opened successfully with DirectShow")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
