# SignAR-Web-App
A web-based application that translates sign language gestures into text and speech using computer vision and machine learning. Built to improve communication accessibility between hearing-impaired individuals and others through real-time gesture recognition.
# SignAR — Translator Web App

A beautiful Flask web interface for your AR Sign Language Translator project.

## 📁 File Structure

```
sign_web/
├── app.py                    ← Flask web server (run this!)
├── requirements.txt          ← Python dependencies
├── hand_landmarker.task      ← MediaPipe model (copy here!)
├── templates/
│   ├── index.html            ← Page 1: System Test + Help
│   ├── dashboard.html        ← Page 2: Mode Selection
│   ├── letter_translator.html ← Letter A-Z Translator
│   └── sentence_translator.html ← Sentence Train + Test
├── backend/
│   ├── sign_language_translator.py
│   ├── sentence_engine.py
│   ├── sentence_translator.py
│   └── test_system.py
```

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place required files
Copy these files into the `sign_web/` folder (same level as app.py):
- `hand_landmarker.task` — download from MediaPipe or use your existing one

The Python backend scripts go in `backend/` (already done if you followed the setup).

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## 📖 How to Use

### Page 1 — System Tests
- Click **Run System Tests** to verify all dependencies
- Tests: Python packages, camera, MediaPipe, text-to-speech
- If all pass → click **Proceed to Translator**
- **HELP button** (top right) shows full instructions

### Page 2 — Dashboard
- Choose **Letter Translator** or **Sentence Translator**

### Letter Translator
- Click Launch → camera window opens
- Show ASL hand signs → letters appear
- `C` to clear, `Q` to quit

### Sentence Translator — Train
1. Add sentences in the web UI
2. Click **Start Recording Gestures** → camera opens
3. Press `R` to record, `SPACE` to save (8 times per sentence)
4. Click **Train Model** → watch live training log

### Sentence Translator — Test
- Click **Launch Sentence Recognizer** → camera opens
- Show your trained gesture → sentence is displayed and spoken
- Unknown gesture? It prompts you to teach it

## 🔧 Troubleshooting

**"hand_landmarker.task not found"**
- Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
- Place it in the `sign_web/` folder

**Camera not working**
- Make sure no other app is using the camera
- Check camera permissions in your OS settings

**TTS not working on Linux**
```bash
sudo apt-get install espeak
```
"# SIGN-LANGUAGE-TRANSLATOR"
