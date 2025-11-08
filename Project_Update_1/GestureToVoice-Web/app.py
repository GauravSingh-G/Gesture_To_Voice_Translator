# app.py — Flask Web UI for Gesture-To-Voice Translator (Synced version)

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import threading
import os
from queue import Queue, Empty
from threading import Lock

# ---------------- CONFIG ----------------
MODEL_PATH = "gesture_model.pkl"
MAX_NUM_HANDS = 2
SPEAK_RATE = 150
COOLDOWN_AFTER_SPEAK = 1.0   # seconds pause after speaking
HAND_RESET_DELAY = 1.2       # reset when hand leaves frame
# ----------------------------------------

# Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Train using train_live_gesture.py first.")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_NUM_HANDS,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ---- ASYNC TTS WORKER ----
class TTSWorker(threading.Thread):
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

                engine = pyttsx3.init()
                engine.setProperty("rate", SPEAK_RATE)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine

            except Exception as e:
                print("⚠️ TTS error:", e)
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

    def busy(self):
        with self.lock:
            return self.is_speaking

    def current(self):
        with self.lock:
            return self.current_text

    def stop(self):
        self._stop = True
        try:
            self.q.put_nowait(None)
        except Exception:
            pass


# Initialize Flask and Components
app = Flask(__name__)
cap = cv2.VideoCapture(0)
tts = TTSWorker()

if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam.")

last_gesture = None
last_spoken_time = 0
hand_visible = False
last_hand_time = time.time()
cooldown_until = 0


# ===== Helper Function: Extract Features =====
def extract_features(result):
    all_hands_data = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, base_z = raw[0]
            normalized = [(x - base_x, y - base_y, z - base_z) for x, y, z in raw]
            all_hands_data.extend([v for pt in normalized for v in pt])
    detected = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    pad_needed = (MAX_NUM_HANDS - detected) * 21 * 3
    all_hands_data.extend([0.0] * pad_needed)
    return np.array(all_hands_data[:126]).reshape(1, -1)


# ===== Main Generator (video + speech sync) =====
def gen_frames():
    global last_gesture, last_spoken_time, hand_visible, last_hand_time, cooldown_until

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        current_time = time.time()
        prediction_text = "No Hand Detected"

        # ---- Handle cooldown or speaking ----
        if tts.busy() or current_time < cooldown_until:
            speaking_word = tts.current()
            display_text = f"🗣️ Speaking: {speaking_word}" if speaking_word else "⏳ Please wait..."
            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if result.multi_hand_landmarks:
                hand_visible = True
                last_hand_time = current_time
                features = extract_features(result)

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

                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            else:
                if hand_visible and (current_time - last_hand_time > HAND_RESET_DELAY):
                    last_gesture = None
                    hand_visible = False

            cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- Encode frame for browser ----
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ===== Flask Routes =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ===== Main Entrypoint =====
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        tts.stop()
        cap.release()
        cv2.destroyAllWindows()
