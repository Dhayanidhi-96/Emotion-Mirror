import cv2
import numpy as np

# BGR color palette for different emotions
EMOTION_COLORS = {
    "angry": (45, 50, 245),       # Red
    "disgust": (75, 180, 60),     # Green
    "fear": (180, 50, 180),       # Purple
    "happy": (45, 220, 255),      # Yellow-Gold
    "sad": (245, 130, 48),        # Blue
    "surprise": (0, 130, 250),    # Orange
    "neutral": (200, 200, 200)    # Light Gray
}

# Minimum face area (width * height) to filter out false detections
MIN_FACE_AREA = 5000


def draw_hud(frame, detections):
    """
    Draw clean overlays — just bounding box + emotion label per face.
    """
    h_img, w_img, _ = frame.shape

    # Filter out small / false detections
    valid = []
    for det in detections:
        x, y, w, h = det["box"]
        if w * h >= MIN_FACE_AREA:
            valid.append(det)

    # Title bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "EMOTION MIRROR", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if not valid:
        # "No Face Detected" banner
        msg = "No Face Detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = int(w_img / 2 - tw / 2)
        cy = h_img - 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - 15, cy - th - 10), (cx + tw + 15, cy + 10), (30, 30, 150), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, msg, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return

    for det in valid:
        x, y, w, h = det["box"]
        emotions = det["emotions"]

        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]
        color = EMOTION_COLORS.get(dominant, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Label background
        label = f"{dominant.upper()}: {int(confidence * 100)}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - lh - 14), (x + lw + 10, y), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Label text
        cv2.putText(frame, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
