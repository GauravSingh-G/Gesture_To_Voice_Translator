# GestureToVoice.py — Real-time synced + speaking display version
# Works smoothly and shows current spoken word on-screen

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import pyttsx3
import time
import os
import traceback
from queue import Queue, Empty
from threading import Thread, Lock

# ---------- CONFIG ----------
MODEL_PATH = "gesture_model.pkl"
MAX_NUM_HANDS = 2
SPEAK_RATE = 150
HAND_RESET_DELAY = 1.2
COOLDOWN_AFTER_SPEAK = 1.0   # seconds pause after each speech
DEBUG = True
# -----------------------------

FIXED_FEATURE_LEN = 21 * 3 * MAX_NUM_HANDS  # 126 for two hands

def debug(*a, **k):
    if DEBUG:
        print(*a, **k)

# ========== ASYNC TTS THREAD ==========
class TTSWorker(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = Queue()
        self._stop = False
        self.is_speaking = False
        self.current_text = ""
        self.lock = Lock()
        self.start()

    def run(self):
        while not self._stop:
            try:
                text = self.q.get(timeout=0.1)
            except Empty:
                continue
            if text is None:
                break
            try:
                with self.lock:
                    self.is_speaking = True
                    self.current_text = text
                debug("🗣️ Speaking:", text)

                # reinitialize each time (fix for COM freeze)
                engine = pyttsx3.init()
                engine.setProperty("rate", SPEAK_RATE)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine

            except Exception as e:
                print("⚠️ TTS speaking error:", e)
            finally:
                with self.lock:
                    self.is_speaking = False
                    self.current_text = ""
                self.q.task_done()

    def speak(self, text: str):
        if text:
            try:
                self.q.put_nowait(text)
            except Exception:
                self.q.put(text)

    def stop(self):
        self._stop = True
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

    def busy(self):
        with self.lock:
            return self.is_speaking

    def current(self):
        with self.lock:
            return self.current_text


# ========== LOAD MODEL SAFELY ==========
def load_model(path):
    if not os.path.exists(path):
        print(f"❌ Model file not found at '{path}'. Train first.")
        return None
    try:
        m = joblib.load(path)
        print("✅ Model loaded (joblib).")
        return m
    except Exception as e_job:
        debug("joblib load failed:", e_job)
        try:
            with open(path, "rb") as f:
                m = pickle.load(f)
            print("✅ Model loaded (pickle).")
            return m
        except Exception as e_pick:
            print("❌ Could not load model:", e_pick)
            traceback.print_exc()
            return None


model = load_model(MODEL_PATH)
if model is None:
    exit()

expected_n_features = getattr(model, "n_features_in_", FIXED_FEATURE_LEN)
debug("Model expects n_features_in_ =", expected_n_features)

# ========== INIT COMPONENTS ==========
tts = TTSWorker()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

print("\n🎥 Gesture-to-Voice Translator (Two-Hand Mode) started.")
print("👉 Show your trained gestures. Press 'q' to quit.\n")

# ========== HELPERS ==========
def prepare_feature(result):
    """Return (1, FIXED_FEATURE_LEN) vector."""
    all_hands_data = []
    for hand in result.multi_hand_landmarks:
        raw = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        base_x, base_y, base_z = raw[0]
        normalized = [(x - base_x, y - base_y, z - base_z) for x, y, z in raw]
        all_hands_data.extend([v for pt in normalized for v in pt])
    if len(result.multi_hand_landmarks) < MAX_NUM_HANDS:
        all_hands_data.extend([0.0] * ((MAX_NUM_HANDS - len(result.multi_hand_landmarks)) * 21 * 3))
    all_hands_data = all_hands_data[:FIXED_FEATURE_LEN]
    return np.array(all_hands_data).reshape(1, -1)

# ========== MAIN LOOP ==========
last_gesture = None
last_spoken_time = 0
hand_visible = False
last_hand_time = time.time()
cooldown_until = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        current_time = time.time()
        prediction_text = "No Hand Detected"

        # During speech or cooldown
        if tts.busy() or current_time < cooldown_until:
            speaking_word = tts.current()
            if speaking_word:
                cv2.putText(frame, f"🗣️ SPEAKING: {speaking_word}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"⏳ SPEAKING... please wait", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("🖐 Gesture To Voice Translator", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
            continue

        if result.multi_hand_landmarks:
            hand_visible = True
            last_hand_time = current_time
            features = prepare_feature(result)

            # fix dimension mismatch if needed
            if features.shape[1] != expected_n_features:
                diff = expected_n_features - features.shape[1]
                if diff > 0:
                    features = np.pad(features, ((0, 0), (0, diff)))
                else:
                    features = features[:, :expected_n_features]

            try:
                pred = str(model.predict(features)[0])
                prediction_text = pred

                # Speak only if gesture changed
                if pred != last_gesture:
                    print(f"🗣️ Speaking: {pred}")
                    tts.speak(pred)
                    last_gesture = pred
                    last_spoken_time = current_time
                    cooldown_until = current_time + COOLDOWN_AFTER_SPEAK

            except Exception as e:
                prediction_text = "Error"
                print("⚠️ Prediction error:", e)

            for hand in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        else:
            if hand_visible and (current_time - last_hand_time > HAND_RESET_DELAY):
                last_gesture = None
                hand_visible = False

        cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("🖐 Gesture To Voice Translator", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    print("👋 Exited Gesture-to-Voice Translator.")
