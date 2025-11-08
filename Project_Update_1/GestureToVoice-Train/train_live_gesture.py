# train_live_gesture.py — Two-hand gesture live trainer (robust version)

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ===== CONFIG =====
MODEL_PATH = "gesture_model.pkl"
DATA_PATH = "gesture_data.npz"
MAX_NUM_HANDS = 2
FIXED_FEATURE_LEN = 21 * 3 * MAX_NUM_HANDS  # 126 for two hands
# ==================

# ===== Load existing data =====
if os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH, allow_pickle=True)
    X, y = list(data["X"]), list(data["y"])
    print(f"✅ Loaded previous training data with {len(X)} samples.")
else:
    X, y = [], []
    print("🧠 Starting fresh training dataset...")

# ===== Load or create model =====
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Loaded existing model.")
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    print("🧩 Created new RandomForest model.")

# ===== Initialize MediaPipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

print("\n🎥 Live Gesture Trainer (Two-Hand Mode)")
print("Press 't' to start training a new gesture")
print("Press 's' to stop and save model")
print("Press 'q' to quit\n")

current_label = None
collecting = False
sample_count = 0


# ===== Helper: build features for 1–2 hands =====
def extract_features(result):
    """Return fixed-length (126,) vector for 1–2 hands."""
    all_hands_data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = raw[0]
            normalized = [(x - base_x, y - base_y, z - base_z) for x, y, z in raw]
            all_hands_data.extend([v for pt in normalized for v in pt])

    # pad if fewer than 2 hands or empty
    detected = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    pad_needed = (MAX_NUM_HANDS - detected) * 21 * 3
    all_hands_data.extend([0.0] * pad_needed)

    # enforce consistent fixed length
    if len(all_hands_data) > FIXED_FEATURE_LEN:
        all_hands_data = all_hands_data[:FIXED_FEATURE_LEN]
    elif len(all_hands_data) < FIXED_FEATURE_LEN:
        all_hands_data.extend([0.0] * (FIXED_FEATURE_LEN - len(all_hands_data)))

    return np.array(all_hands_data, dtype=np.float32)


# ===== Main loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw and collect
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Always create a valid feature vector (even if no hand)
    features = extract_features(result)

    if collecting and current_label:
        X.append(features.tolist())
        y.append(current_label)
        sample_count += 1
        cv2.putText(frame, f"Collecting '{current_label}' [{sample_count}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ===== Display =====
    hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    color = (0, 255, 0) if hand_count == 2 else (0, 255, 255) if hand_count == 1 else (0, 0, 255)
    cv2.putText(frame, f"Hands detected: {hand_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Gesture: {current_label or 'None'} | Samples: {sample_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("✋ Live Gesture Trainer (Two-Hand Mode)", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===== Controls =====
    if key == ord('t') and not collecting:
        current_label = input("Enter gesture name (e.g., HELLO): ").strip().upper()
        if not current_label:
            print("⚠️ Invalid label, try again.")
            continue
        collecting = True
        sample_count = 0
        print(f"🟢 Collecting samples for '{current_label}'... Press 's' to stop and train.")

    elif key == ord('s') and collecting:
        collecting = False
        print("⏳ Training model with new samples...")

        if len(X) > 0:
            # ensure all feature vectors are same shape
            fixed_X = []
            for x in X:
                arr = np.array(x, dtype=np.float32).flatten()
                if arr.shape[0] != FIXED_FEATURE_LEN:
                    arr = np.pad(arr, (0, max(0, FIXED_FEATURE_LEN - arr.shape[0])))[:FIXED_FEATURE_LEN]
                fixed_X.append(arr)
            X_np = np.stack(fixed_X)
            y_np = np.array(y)

            model.fit(X_np, y_np)
            joblib.dump(model, MODEL_PATH)
            np.savez(DATA_PATH, X=X_np, y=y_np)
            print(f"✅ Model trained & saved. Total gestures learned: {len(set(y))}")

        sample_count = 0
        print("Ready for next gesture (press 't').")

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Training session ended.")
