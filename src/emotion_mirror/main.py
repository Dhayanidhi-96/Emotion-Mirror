import cv2
import time
import sys
import argparse

from emotion_mirror.detector import EmotionDetector
from emotion_mirror.display import draw_hud

def main():
    parser = argparse.ArgumentParser(description="Emotion Mirror - Realtime Face Emotion Detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default is 0)")
    parser.add_argument("--mtcnn", action="store_true", default=True, help="Use MTCNN detector (default)")
    args = parser.parse_args()

    print("Initializing Emotion Mirror...")
    print("Loading pre-trained FER model (this may take a few seconds on first run)...")

    detector = EmotionDetector(mtcnn=args.mtcnn)

    print(f"Opening webcam device {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open webcam source {args.camera}.", file=sys.stderr)
        return 1

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Warmup: wait 2 seconds for camera to stabilize ──
    print("Warming up camera (2 seconds)...")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Starting...", (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Emotion Mirror", frame)
        cv2.waitKey(1)

    print("\n--- Emotion Mirror Running ---")
    print("Press 'Q' or 'ESC' inside the window to Quit.")
    print("---------------------------------")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.", file=sys.stderr)
            break

        # Mirror flip
        frame = cv2.flip(frame, 1)

        # Detect emotions
        detections = detector.detect_emotions(frame)

        # Draw overlays
        draw_hud(frame, detections)

        # FPS counter
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Mirror", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Emotion Mirror stopped.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
