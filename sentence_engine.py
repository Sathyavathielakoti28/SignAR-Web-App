"""
Sentence Engine — Train Model on YOUR Custom Sentences
Edit CUSTOM_SENTENCES list below with your own sentences!

Usage:
  python sentence_engine.py              # Collect + train
  python sentence_engine.py --train      # Train only
"""

import numpy as np
import pickle
import os
import time
import cv2
import mediapipe as mp
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions


# ══════════════════════════════════════════════════════════════
# ✏️ EDIT THIS: Add your own sentences here!
# ══════════════════════════════════════════════════════════════
CUSTOM_SENTENCES = []
n=int(input("How many sentences you want to add:"))
for i in range(n):
    st=input("enter a sentence:")
    CUSTOM_SENTENCES.append(st)
# Add as many sentences as you want!
# ══════════════════════════════════════════════════════════════


MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
SEQUENCE_LENGTH = 40
FEATURE_DIM = 63


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
                 num_classes=len(CUSTOM_SENTENCES), dropout=0.3):
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


class SentenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]


class DataCollector:
    def __init__(self, task_path=MODEL_PATH):
        self.latest_result, self.sequences, self.labels = None, [], []
        self.extractor = FeatureExtractor()
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=RunningMode.LIVE_STREAM, num_hands=1,
            min_hand_detection_confidence=0.7,
            result_callback=lambda r,i,t: setattr(self, 'latest_result', r))
        self.detector = HandLandmarker.create_from_options(opts)
    
    def collect_sentence(self, sentence, num_samples=8):
        cap = cv2.VideoCapture(0)
        count, frame_buffer, last_ts, recording = 0, [], 0, False
        
        print(f"\n{'='*65}\n  \"{sentence}\"\n  Target: {num_samples} samples\n{'='*65}")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = time.monotonic_ns() // 1_000_000
            if ts <= last_ts: ts = last_ts + 1
            last_ts = ts
            self.detector.detect_async(mp_img, ts)
            
            if self.latest_result and self.latest_result.hand_landmarks:
                hand = self.latest_result.hand_landmarks[0]
                h, w = frame.shape[:2]
                for lm in hand:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,80), -1)
                if recording:
                    frame_buffer.append(hand)
                    cv2.circle(frame, (w-30, 30), 15, (0,0,255), -1)
            
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (w,140), (30,30,30), -1)
            cv2.putText(ov, f'"{sentence}"', (20,50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(ov, f"Sample {count}/{num_samples}", (20,90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
            status = f"REC... {len(frame_buffer)} frames" if recording else "Press R to record"
            color = (0,0,255) if recording else (0,255,0)
            cv2.putText(ov, status, (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if recording and len(frame_buffer) >= 15:
                cv2.rectangle(ov, (0,h-50), (w,h), (40,40,40), -1)
                cv2.putText(ov, "Press SPACE when finished", (20,h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
            cv2.imshow("Sentence Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and not recording:
                recording, frame_buffer = True, []
                print(f"  🔴 Recording...")
            elif key == ord(' ') and recording and len(frame_buffer) >= 15:
                try:
                    self.sequences.append(self.extractor.extract_sequence(frame_buffer))
                    self.labels.append(sentence)
                    count += 1
                    print(f"  ✓ Saved {count}/{num_samples} ({len(frame_buffer)} frames)")
                    recording, frame_buffer = False, []
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    recording, frame_buffer = False, []
            elif key == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"  ✓ Done: {count} samples")
    
    def save(self, path="my_sentences.pkl"):
        if not self.sequences: return False
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"sequences": np.array(self.sequences, dtype=np.float32),
                        "labels": np.array(self.labels),
                        "sentences": CUSTOM_SENTENCES}, f)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)
        print(f"\n✓ Saved → {path} ({len(self.sequences)} samples)")
        return True
    
    def load(self, path="my_sentences.pkl"):
        if not os.path.exists(path): return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.sequences, self.labels = list(data["sequences"]), list(data["labels"])
            print(f"✓ Loaded ← {path} ({len(self.sequences)} samples)")
            return True
        except: return False


def train_model(dataset_path="my_sentences.pkl", epochs=60):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    sent_to_idx = {s: i for i, s in enumerate(CUSTOM_SENTENCES)}
    label_idx = [sent_to_idx[lbl] for lbl in data["labels"]]
    
    print(f"\n{'='*65}\n  Training ({len(data['sequences'])} samples)\n{'='*65}")
    
    X_tr, X_te, y_tr, y_te = train_test_split(data["sequences"], label_idx,
                                               test_size=0.15, random_state=42, stratify=label_idx)
    train_loader = DataLoader(SentenceDataset(X_tr, y_tr), batch_size=8, shuffle=True)
    test_loader = DataLoader(SentenceDataset(X_te, y_te), batch_size=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceLSTM(num_classes=len(CUSTOM_SENTENCES)).to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    print(f"Device: {device} | Epochs: {epochs}\n")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = sum([(criterion((outputs := model(seqs.to(device))), 
                          lbls.to(device)), optimizer.zero_grad(), 
                          outputs.sum().backward())[0].item() 
                          for seqs, lbls in train_loader for _ in [optimizer.step()]])
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, lbls in test_loader:
                outputs = model(seqs.to(device))
                correct += (torch.max(outputs, 1)[1] == lbls.to(device)).sum().item()
                total += lbls.size(0)
        
        acc = 100 * correct / total
        scheduler.step(train_loss / len(train_loader))
        print(f"Epoch {epoch+1:2d}/{epochs}  Loss: {train_loss/len(train_loader):.4f}  Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({'model': model.state_dict(), 'sentences': CUSTOM_SENTENCES, 
                       'accuracy': acc}, "my_sentence_model.pth")
    
    print(f"\n{'='*65}\n  ✓ Best: {best_acc:.2f}% → my_sentence_model.pth\n{'='*65}\n")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        if not os.path.exists("my_sentences.pkl"):
            print("✗ my_sentences.pkl not found. Collect data first.")
            return
        train_model()
        return
    
    print("\n" + "="*65 + "\n  Sentence Engine — YOUR Custom Sentences\n" + "="*65)
    collector = DataCollector()
    
    if os.path.exists("my_sentences.pkl"):
        if input("\nResume? (y/n): ").strip().lower() == 'y':
            collector.load("my_sentences.pkl")
    
    print(f"\nYour {len(CUSTOM_SENTENCES)} sentences:")
    for i, s in enumerate(CUSTOM_SENTENCES, 1):
        print(f"  {i:2d}. {s}")
    
    print("\n" + "="*65)
    print("Press R → sign sentence → press SPACE (8 times each)\n" + "="*65 + "\n")
    
    for sentence in CUSTOM_SENTENCES:
        choice = input(f"\"{sentence}\"? (y/n/q): ").strip().lower()

        if choice == 'y':
            collector.collect_sentence(sentence, 8)

        elif choice == 'q':
            print("\nExiting collection...")
            break

        else:
            print("Skipped.")

    
    if collector.sequences:
        collector.save()
        if input("\nTrain now? (y/n): ").strip().lower() == 'y':
            train_model()
            print("\n✓ Run: python sentence_translator.py")

if __name__ == "__main__":
    main()