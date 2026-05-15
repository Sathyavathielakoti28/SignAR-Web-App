"""
Sentence Translator — Recognize & Speak YOUR Trained Sentences
Uses model trained in sentence_engine.py

Usage:
  python sentence_translator.py
"""

import cv2
import time
import numpy as np
import os
import mediapipe as mp
from collections import deque
import pyttsx3
import threading
import torch
import torch.nn as nn

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions


# ══════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
SEQUENCE_LENGTH = 40
FEATURE_DIM = 63
CONF_THRESHOLD = 0.55


# ══════════════════════════════════════════════════════════════
class FeatureExtractor:
    @staticmethod
    def extract_frame(landmarks):
        features = []
        wrist, mid_mcp = landmarks[0], landmarks[9]
        palm_size = np.sqrt((mid_mcp.x-wrist.x)**2+(mid_mcp.y-wrist.y)**2)+1e-6
        for lm in landmarks:
            features.extend([float((lm.x-wrist.x)/palm_size),
                           float((lm.y-wrist.y)/palm_size),
                           float((lm.z-wrist.z)/palm_size)])
        return np.array(features, dtype=np.float32)
    
    def extract_sequence(self, landmark_list):
        if len(landmark_list) < SEQUENCE_LENGTH:
            while len(landmark_list) < SEQUENCE_LENGTH:
                landmark_list.append(landmark_list[-1])
        else:
            landmark_list = landmark_list[-SEQUENCE_LENGTH:]
        return np.array([self.extract_frame(lm) for lm in landmark_list], dtype=np.float32)


class SentenceLSTM(nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=128, 
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.fc(self.dropout(context))


class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
    
    def speak(self, text):
        def _s():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except: pass
        threading.Thread(target=_s, daemon=True).start()


class SentenceRecognizer:
    def __init__(self, model_path="my_sentence_model.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found. Run sentence_engine.py first!")
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.sentences = checkpoint['sentences']
        self.model = SentenceLSTM(num_classes=len(self.sentences))
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.extractor = FeatureExtractor()
        
        print(f"✓ Model loaded: {len(self.sentences)} sentences, {checkpoint.get('accuracy', 0):.1f}% accuracy")
    
    def predict(self, landmark_sequence, threshold=CONF_THRESHOLD):
        try:
            features = self.extractor.extract_sequence(landmark_sequence)
            with torch.no_grad():
                output = self.model(torch.FloatTensor(features).unsqueeze(0))
                probs = torch.softmax(output, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                if float(conf) < threshold: return None, float(conf)
                return self.sentences[int(pred_idx)], float(conf)
        except: return None, 0.0


class LiveSentenceTranslator:
    def __init__(self, task_path=MODEL_PATH):
        self.recognizer = SentenceRecognizer()
        self.tts = TTSEngine()
        self.latest_result, self.last_ts = None, 0
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_sentence, self.last_sentence = "", ""
        self.last_detection_time, self.total_sentences = time.time(), 0
        
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=RunningMode.LIVE_STREAM, num_hands=1,
            min_hand_detection_confidence=0.7,
            result_callback=lambda r,i,t: setattr(self, 'latest_result', r))
        self.detector = HandLandmarker.create_from_options(opts)
    
    def speak_sentence(self, sentence):
        print(f"\n{'='*60}\n  {sentence}\n{'='*60}")
        self.tts.speak(sentence)
        self.last_sentence, self.total_sentences = sentence, self.total_sentences + 1
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("  Sentence Translator — YOUR Trained Sentences")
        print("="*60)
        print("\nAvailable:")
        for i, s in enumerate(self.recognizer.sentences, 1):
            print(f"  {i:2d}. {s}")
        print("\n" + "="*60)
        print("Sign a sentence and it will be recognized & spoken!")
        print("ENTER=speak now  R=reset  C=clear  Q=quit")
        print("="*60 + "\n")
        
        detected, confidence = None, 0.0  # track last detected across frames

        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            ts = time.monotonic_ns() // 1_000_000
            if ts <= self.last_ts: ts = self.last_ts + 1
            self.last_ts = ts
            self.detector.detect_async(mp_img, ts)
            
            detected, confidence = None, 0.0
            
            if self.latest_result and self.latest_result.hand_landmarks:
                hand = self.latest_result.hand_landmarks[0]
                h, w = frame.shape[:2]
                for lm in hand:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,80), -1)
                
                self.last_detection_time = time.time()
                self.frame_buffer.append(hand)
                
                if len(self.frame_buffer) == SEQUENCE_LENGTH:
                    sentence, conf = self.recognizer.predict(list(self.frame_buffer))
                    if sentence and sentence != self.current_sentence:
                        self.current_sentence = sentence
                        self.speak_sentence(sentence)
                        detected, confidence = sentence, conf
                        self.frame_buffer.clear()
                    elif sentence:
                        detected, confidence = sentence, conf
            else:
                if time.time() - self.last_detection_time > 2.0:
                    self.current_sentence = ""
            
            # UI
            h, w = frame.shape[:2]
            ov = frame.copy()
            
            # Top - detection
            cv2.rectangle(ov, (0,0), (w,110), (30,30,30), -1)
            if detected:
                cv2.putText(ov, "DETECTED:", (20,35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
                cv2.putText(ov, detected, (20,85),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,200), 2)
                cv2.putText(ov, f"{confidence:.0%}", (w-120, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,200), 2)
            else:
                cv2.putText(ov, "Sign a sentence...", (20,70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.6, (120,120,120), 2)
            
            # Buffer
            buf_pct = len(self.frame_buffer) / SEQUENCE_LENGTH
            buf_w = int(buf_pct * 250)
            cv2.rectangle(ov, (w-270,15), (w-20,30), (50,50,50), -1)
            cv2.rectangle(ov, (w-270,15), (w-270+buf_w,30), (0,200,100), -1)
            cv2.putText(ov, "Buffer", (w-270, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            
            # Bottom - last spoken
            cv2.rectangle(ov, (0,h-90), (w,h), (25,25,25), -1)
            cv2.putText(ov, "Last Spoken:", (20,h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            cv2.putText(ov, self.last_sentence if self.last_sentence else "...", (20,h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(ov, "ENTER=speak  R=reset  C=clear  Q=quit", (20,h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130,130,130), 1)
            cv2.putText(ov, f"Total: {self.total_sentences}", (w-150,h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130,130,130), 1)
            
            cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
            cv2.imshow("Sentence Translator", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 13:  # Enter key — manually trigger speech
                if self.current_sentence:
                    self.speak_sentence(self.current_sentence)
                    self.frame_buffer.clear()
                elif detected:
                    self.speak_sentence(detected)
                else:
                    print("Nothing detected yet to speak.")
            elif key == ord('r'):
                self.current_sentence, self.frame_buffer = "", deque(maxlen=SEQUENCE_LENGTH)
                print("Reset")
            elif key == ord('c'):
                self.current_sentence, self.last_sentence, self.frame_buffer = "", "", deque(maxlen=SEQUENCE_LENGTH)
                print("Cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Total recognized: {self.total_sentences}")


def main():
    if not os.path.exists("my_sentence_model.pth"):
        print("\n✗ my_sentence_model.pth not found")
        print("Run: python sentence_engine.py\n")
        return
    
    try:
        LiveSentenceTranslator().run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()