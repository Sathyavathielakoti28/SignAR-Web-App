"""
AR Sign Language Translator
Full 26-Letter ASL Support
MediaPipe Tasks API (VIDEO mode)
Latest MediaPipe Compatible
"""

import cv2
import time
import numpy as np
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SignLanguageTranslator:
    def __init__(self):
        # ── MediaPipe Hand Landmarker ─────────────────────────
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # ── State ─────────────────────────────────────────────
        self.gesture_buffer = deque(maxlen=10)
        self.last_capture = time.time()
        self.current_letter = ""
        self.sentence = ""

    # ─────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────
    def _dist(self, a, b):
        return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    def _finger_states(self, lm):
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        s = [1 if lm[tips[0]].x < lm[pips[0]].x else 0]
        for i in range(1, 5):
            s.append(1 if lm[tips[i]].y < lm[pips[i]].y else 0)
        return s

    def _all_curled(self, lm):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        return all(lm[tips[i]].y > lm[pips[i]].y for i in range(4))

    def _touch(self, lm, a, b, thr=0.06):
        return self._dist(lm[a], lm[b]) < thr

    # ─────────────────────────────────────────────────────────
    # Full A–Z Recognition
    # ─────────────────────────────────────────────────────────
    def recognize_letter(self, lm):
        f = self._finger_states(lm)
        th, ix, mi, ri, pi = f

        if self._all_curled(lm) and not self._touch(lm, 4, 8, 0.07): return 'A'
        if ix==1 and mi==1 and ri==1 and pi==1 and lm[4].x > lm[5].x: return 'B'
        if (lm[8].y>lm[6].y and lm[12].y>lm[10].y and lm[16].y>lm[14].y and
            lm[20].y>lm[18].y and lm[8].y<lm[5].y and abs(lm[4].x-lm[8].x)<0.15): return 'C'
        if ix==1 and mi==0 and ri==0 and pi==0 and self._touch(lm,4,12,0.07): return 'D'
        if f==[0,0,0,0,0] and lm[8].y>lm[6].y and lm[4].y>lm[3].y: return 'E'
        if self._touch(lm,4,8,0.06) and mi==1 and ri==1 and pi==1: return 'F'
        if ix==1 and mi==0 and ri==0 and pi==0 and abs(lm[8].y-lm[5].y)<0.05 and lm[4].y<lm[3].y: return 'G'
        if ix==1 and mi==1 and ri==0 and pi==0 and abs(lm[8].y-lm[12].y)<0.05: return 'H'
        if f == [0,0,0,0,1]: return 'I'
        if pi==1 and ix==0 and mi==0 and ri==0 and th==1: return 'J'
        if ix==1 and mi==1 and ri==0 and pi==0 and lm[4].y<lm[8].y and self._dist(lm[4],lm[8])>0.08: return 'K'
        if th==1 and ix==1 and mi==0 and ri==0 and pi==0: return 'L'
        if f==[0,0,0,0,0] and lm[4].y>lm[8].y and lm[8].y<lm[5].y: return 'M'
        if f==[0,0,0,0,0] and lm[4].y>lm[8].y and self._dist(lm[8],lm[12])<0.05: return 'N'
        if self._touch(lm,4,8,0.07) and lm[12].y>lm[10].y and lm[16].y>lm[14].y and lm[20].y>lm[18].y: return 'O'
        if ix==1 and mi==1 and ri==0 and pi==0 and lm[8].y>lm[5].y and lm[4].x<lm[8].x: return 'P'
        if ix==1 and mi==0 and ri==0 and pi==0 and lm[8].y>lm[5].y and lm[4].y>lm[3].y: return 'Q'
        if ix==1 and mi==1 and ri==0 and pi==0 and self._dist(lm[8],lm[12])<0.04: return 'R'
        if f==[0,0,0,0,0] and lm[4].x<lm[8].x and lm[4].y<lm[8].y: return 'S'
        if ix==0 and mi==0 and ri==0 and pi==0 and lm[4].x>lm[6].x and lm[4].y<lm[8].y: return 'T'
        if ix==1 and mi==1 and ri==0 and pi==0 and self._dist(lm[8],lm[12])<0.06 and lm[8].y<lm[5].y: return 'U'
        if ix==1 and mi==1 and ri==0 and pi==0 and self._dist(lm[8],lm[12])>0.06: return 'V'
        if ix==1 and mi==1 and ri==1 and pi==0: return 'W'
        if ix==0 and mi==0 and ri==0 and pi==0 and lm[8].y>lm[7].y and lm[7].y<lm[6].y: return 'X'
        if f == [1,0,0,0,1]: return 'Y'
        if ix==1 and mi==0 and ri==0 and pi==0 and th==0 and lm[8].y<lm[6].y: return 'Z'
        return None

    # ─────────────────────────────────────────────────────────
    def process_frame(self, frame, timestamp_ms):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected = None

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                h, w = frame.shape[:2]

                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]

                for a,b in connections:
                    cv2.line(frame,
                             (int(hand[a].x*w), int(hand[a].y*h)),
                             (int(hand[b].x*w), int(hand[b].y*h)),
                             (0,180,0),2)

                for lm in hand:
                    cv2.circle(frame,
                               (int(lm.x*w), int(lm.y*h)),
                               4,(0,255,0),-1)

                detected = self.recognize_letter(hand)

        self.gesture_buffer.append(detected)

        stable = None
        if detected:
            counts = {}
            for g in self.gesture_buffer:
                if g:
                    counts[g] = counts.get(g,0)+1
            stable = max(counts, key=counts.get) if counts else None

        if stable and time.time() - self.last_capture > 1.5:
            self.current_letter = stable
            self.sentence += stable
            self.last_capture = time.time()

        return frame, stable

    # ─────────────────────────────────────────────────────────
    def draw_ui(self, frame, letter):
        h,w = frame.shape[:2]
        ov = frame.copy()

        cv2.rectangle(ov,(0,0),(w,80),(30,30,30),-1)
        text = f"Letter: {letter}" if letter else "No gesture detected"
        color = (0,255,80) if letter else (120,120,120)
        cv2.putText(ov,text,(20,55),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,color,2)

        cv2.rectangle(ov,(0,h-90),(w,h),(30,30,30),-1)
        cv2.putText(ov,f"Recognised Letters are: {self.sentence or '...'}",
                    (20,h-45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)

        cv2.putText(ov,"                   Q=quit                      C=clear",
                    (20,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,140,140),1)

        cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
        return frame

    # ─────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not accessible")
            return

        print("\nSign Language Translator — Full A-Z")
        print("Q=quit                   C=clear \n")

        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = int((time.time() - start) * 1000)
            frame, letter = self.process_frame(frame, ts)
            frame = self.draw_ui(frame, letter)

            cv2.imshow("Sign Language Translator — A to Z", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): self.sentence = ""

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SignLanguageTranslator().run()