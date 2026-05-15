"""
Test and Demo Script for AR Sign Language Translator
Compatible with MediaPipe 0.10+ (Tasks API)
"""

import sys
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...\n")

    dependencies = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'pyttsx3': 'pyttsx3'
    }

    missing = []

    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - OK")
        except ImportError:
            print(f"✗ {package:20s} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("\n✅ All dependencies installed!")
    return True


def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False

        ret, _ = cap.read()
        cap.release()

        if not ret:
            print("❌ Camera read failed")
            return False

        print("✅ Camera is working!")
        return True

    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False


def test_mediapipe():
    """Test MediaPipe Tasks API"""
    print("\nTesting MediaPipe hand detection...")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import numpy as np

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )

        detector = vision.HandLandmarker.create_from_options(options)

        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=dummy_image
        )

        detector.detect(mp_image)

        print("✅ MediaPipe Tasks API working!")
        return True

    except Exception as e:
        print(f"❌ MediaPipe error: {e}")
        return False


def test_tts():
    """Test text-to-speech"""
    print("\nTesting text-to-speech...")

    try:
        import pyttsx3

        engine = pyttsx3.init()
        print("✅ TTS engine initialized!")

        response = input("Speak test message? (y/n): ")
        if response.lower() == 'y':
            engine.say("Text to speech is working correctly.")
            engine.runAndWait()

        return True

    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False


def demo_hand_tracking():
    """Live hand tracking demo using MediaPipe Tasks"""
    print("\nStarting hand tracking demo...")
    print("Press 'q' to quit\n")

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )

        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:
                cv2.putText(frame, "Hand detected ✓",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show your hand...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            cv2.imshow("Hand Tracking Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("AR SIGN LANGUAGE TRANSLATOR - TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        ("Dependencies", check_dependencies),
        ("Camera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("Text-to-Speech", test_tts),
    ]

    results = []

    for name, test in tests:
        results.append((name, test()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        print(f"{name:20s}: {'✅ PASS' if passed else '❌ FAIL'}")

    if all(passed for _, passed in results):
        print("\n✅ System ready!")

        if input("\nRun hand tracking demo? (y/n): ").lower() == 'y':
            demo_hand_tracking()
    else:
        print("\n❌ Fix issues before proceeding.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(0)
