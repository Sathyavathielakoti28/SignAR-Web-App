"""
AR Sign Language Translator — Flask Web App
"""
import os
import sys
import json
import pickle
import threading
import subprocess
import base64
import time
import numpy as np
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODEL_FILE = os.path.join(BASE_DIR, "hand_landmarker.task")
SENTENCE_DATA = os.path.join(BASE_DIR, "my_sentences.pkl")
SENTENCE_MODEL = os.path.join(BASE_DIR, "my_sentence_model.pth")

sys.path.insert(0, BACKEND_DIR)

# ── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/letter_translator")
def letter_translator():
    return render_template("letter_translator.html")

@app.route("/sentence_translator")
def sentence_translator():
    return render_template("sentence_translator.html")

# ── API: System Tests ─────────────────────────────────────

@app.route("/api/test/dependencies", methods=["GET"])
def test_dependencies():
    results = {}
    packages = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "pyttsx3": "pyttsx3",
        "torch": "torch",
    }
    for module, pkg in packages.items():
        try:
            __import__(module)
            results[pkg] = {"status": "ok", "label": pkg}
        except ImportError as e:
            results[pkg] = {"status": "fail", "label": pkg, "error": str(e)}
    
    all_ok = all(v["status"] == "ok" for v in results.values())
    return jsonify({"results": results, "all_ok": all_ok})

@app.route("/api/test/camera", methods=["GET"])
def test_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"status": "fail", "message": "Cannot open camera. Make sure it's connected and not in use."})
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"status": "fail", "message": "Camera opened but failed to read frame."})
        h, w = frame.shape[:2]
        return jsonify({"status": "ok", "message": f"Camera working! Resolution: {w}x{h}"})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

@app.route("/api/test/mediapipe", methods=["GET"])
def test_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        task_path = MODEL_FILE
        if not os.path.exists(task_path):
            task_path = os.path.join(BACKEND_DIR, "hand_landmarker.task")
        if not os.path.exists(task_path):
            return jsonify({"status": "fail", "message": f"hand_landmarker.task not found. Place it in: {BASE_DIR}"})
        
        base_options = mp_python.BaseOptions(model_asset_path=task_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        detector = vision.HandLandmarker.create_from_options(options)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        detector.detect(mp_img)
        return jsonify({"status": "ok", "message": "MediaPipe hand detection working!"})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

@app.route("/api/test/tts", methods=["GET"])
def test_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return jsonify({"status": "ok", "message": f"TTS engine ready. {len(voices)} voice(s) available."})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

# ── API: Sentence Training ────────────────────────────────

@app.route("/api/sentences/list", methods=["GET"])
def list_sentences():
    try:
        data_path = SENTENCE_DATA
        if not os.path.exists(data_path):
            data_path = os.path.join(BACKEND_DIR, "my_sentences.pkl")
        if not os.path.exists(data_path):
            return jsonify({"sentences": [], "count": 0})
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        sentences = list(data.get("sentences", []))
        labels = list(data.get("labels", []))
        counts = {s: labels.count(s) for s in sentences}
        return jsonify({"sentences": sentences, "sample_counts": counts, "total_samples": len(labels)})
    except Exception as e:
        return jsonify({"sentences": [], "error": str(e)})

@app.route("/api/model/status", methods=["GET"])
def model_status():
    model_path = SENTENCE_MODEL
    if not os.path.exists(model_path):
        model_path = os.path.join(BACKEND_DIR, "my_sentence_model.pth")
    
    exists = os.path.exists(model_path)
    if exists:
        try:
            import torch
            checkpoint = torch.load(model_path, map_location="cpu")
            sentences = checkpoint.get("sentences", [])
            acc = checkpoint.get("accuracy", 0)
            return jsonify({"trained": True, "sentences": sentences, "accuracy": round(acc, 2), "count": len(sentences)})
        except Exception as e:
            return jsonify({"trained": False, "error": str(e)})
    return jsonify({"trained": False})

# ── API: Launch Apps ──────────────────────────────────────

training_process = None
training_status = {"running": False, "log": [], "done": False, "success": False}

@app.route("/api/launch/letter_translator", methods=["POST"])
def launch_letter_translator():
    script = os.path.join(BACKEND_DIR, "sign_language_translator.py")
    try:
        subprocess.Popen([sys.executable, script], cwd=BACKEND_DIR)
        return jsonify({"status": "ok", "message": "Letter Translator launched in a new window!"})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

@app.route("/api/launch/sentence_collector", methods=["POST"])
def launch_sentence_collector():
    """Launch data collection — non-blocking"""
    data = request.json or {}
    sentences = data.get("sentences", [])
    if not sentences:
        return jsonify({"status": "fail", "message": "No sentences provided"})
    
    script = os.path.join(BACKEND_DIR, "sentence_engine.py")
    # Write a wrapper script
    wrapper = os.path.join(BASE_DIR, "_collect_wrapper.py")
    with open(wrapper, "w") as f:
        f.write(f"""import sys, os
sys.path.insert(0, r"{BACKEND_DIR}")
os.chdir(r"{BACKEND_DIR}")

# Patch the CUSTOM_SENTENCES before running
import sentence_engine as se
se.CUSTOM_SENTENCES = {json.dumps(sentences)}

# Rebuild model class with correct num_classes
se.SentenceLSTM.__init__.__defaults__ = None
import torch.nn as nn
class SentenceLSTM(nn.Module):
    def __init__(self, input_size=se.FEATURE_DIM, hidden_size=128, 
                 num_classes=len(se.CUSTOM_SENTENCES), dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        import torch
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.fc(self.dropout(context))
se.SentenceLSTM = SentenceLSTM

collector = se.DataCollector()
for sentence in se.CUSTOM_SENTENCES:
    print(f"Collecting: {{sentence}}")
    collector.collect_sentence(sentence, 8)
collector.save(r"{os.path.join(BASE_DIR, 'my_sentences.pkl')}")
""")
    try:
        subprocess.Popen([sys.executable, wrapper])
        return jsonify({"status": "ok", "message": f"Collection started for {len(sentences)} sentence(s). A camera window will open."})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

@app.route("/api/train/model", methods=["POST"])
def train_model():
    global training_process, training_status
    data_path = SENTENCE_DATA
    if not os.path.exists(data_path):
        data_path = os.path.join(BACKEND_DIR, "my_sentences.pkl")
    if not os.path.exists(data_path):
        return jsonify({"status": "fail", "message": "No training data found. Collect gesture samples first."})
    
    training_status = {"running": True, "log": ["Starting training..."], "done": False, "success": False}
    
    wrapper = os.path.join(BASE_DIR, "_train_wrapper.py")
    with open(wrapper, "w") as f:
        f.write(f"""import sys, os, pickle
sys.path.insert(0, r"{BACKEND_DIR}")
os.chdir(r"{BASE_DIR}")

with open(r"{data_path}", "rb") as f:
    d = pickle.load(f)
sentences = list(d.get("sentences", []))
if not sentences:
    labels = list(d.get("labels", []))
    sentences = list(set(labels))

import sentence_engine as se
se.CUSTOM_SENTENCES = sentences

import torch.nn as nn
class SentenceLSTM(nn.Module):
    def __init__(self, input_size=se.FEATURE_DIM, hidden_size=128, 
                 num_classes=len(se.CUSTOM_SENTENCES), dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        import torch
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.fc(self.dropout(context))
se.SentenceLSTM = SentenceLSTM

se.train_model(dataset_path=r"{data_path}", epochs=60)
print("TRAINING_COMPLETE")
""")
    
    def run_training():
        global training_process, training_status
        training_process = subprocess.Popen(
            [sys.executable, "-u", wrapper],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in training_process.stdout:
            line = line.strip()
            if line:
                training_status["log"].append(line)
                if len(training_status["log"]) > 200:
                    training_status["log"] = training_status["log"][-200:]
        training_process.wait()
        training_status["running"] = False
        training_status["done"] = True
        training_status["success"] = training_process.returncode == 0
    
    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"status": "ok", "message": "Training started!"})

@app.route("/api/train/status", methods=["GET"])
def training_status_api():
    return jsonify(training_status)

@app.route("/api/launch/sentence_translator", methods=["POST"])
def launch_sentence_translator():
    script = os.path.join(BACKEND_DIR, "sentence_translator.py")
    model_path = SENTENCE_MODEL
    if not os.path.exists(model_path):
        model_path = os.path.join(BACKEND_DIR, "my_sentence_model.pth")
    if not os.path.exists(model_path):
        return jsonify({"status": "fail", "message": "No trained model found. Please train the model first."})
    
    wrapper = os.path.join(BASE_DIR, "_translate_wrapper.py")
    with open(wrapper, "w") as f:
        f.write(f"""import sys, os
sys.path.insert(0, r"{BACKEND_DIR}")
os.chdir(r"{BACKEND_DIR}")
import sentence_translator as st
import torch.nn as nn

class SentenceLSTM(nn.Module):
    def __init__(self, input_size=st.FEATURE_DIM, hidden_size=128, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        import torch
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.fc(self.dropout(context))
st.SentenceLSTM = SentenceLSTM

class Recognizer(st.SentenceRecognizer):
    def __init__(self):
        super().__init__(model_path=r"{model_path}")

orig = st.LiveSentenceTranslator.__init__
def patched_init(self_inner, task_path=st.MODEL_PATH):
    self_inner.recognizer = Recognizer()
    import pyttsx3, threading, time
    from collections import deque
    self_inner.tts = st.TTSEngine()
    self_inner.latest_result, self_inner.last_ts = None, 0
    self_inner.frame_buffer = deque(maxlen=st.SEQUENCE_LENGTH)
    self_inner.current_sentence, self_inner.last_sentence = "", ""
    self_inner.last_detection_time, self_inner.total_sentences = time.time(), 0
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_path),
        running_mode=RunningMode.LIVE_STREAM, num_hands=1,
        min_hand_detection_confidence=0.7,
        result_callback=lambda r,i,t: setattr(self_inner, 'latest_result', r))
    self_inner.detector = HandLandmarker.create_from_options(opts)
st.LiveSentenceTranslator.__init__ = patched_init

st.LiveSentenceTranslator().run()
""")
    try:
        subprocess.Popen([sys.executable, wrapper])
        return jsonify({"status": "ok", "message": "Sentence Translator launched!"})
    except Exception as e:
        return jsonify({"status": "fail", "message": str(e)})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AR Sign Language Translator Web App")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
