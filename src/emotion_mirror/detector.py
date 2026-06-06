import cv2
from fer.fer import FER

class EmotionDetector:
    def __init__(self, mtcnn: bool = True):
        """
        Initialize the FER emotion detector.
        
        Args:
            mtcnn: Use MTCNN for face detection (more accurate, recommended). 
                   If False, uses OpenCV Haar Cascades (faster but less reliable).
        """
        # MTCNN is more robust at detecting faces in varied lighting / angles.
        self.detector = FER(mtcnn=mtcnn)

    def detect_emotions(self, frame):
        """
        Detect faces and emotions in the given frame.
        
        Args:
            frame: OpenCV BGR frame.
            
        Returns:
            A list of dictionaries containing:
            - 'box': [x, y, width, height]
            - 'emotions': dict of emotions and their scores (0.0 to 1.0)
        """
        # fer expects BGR or RGB (internally converts if needed, but it handles standard OpenCV frames)
        try:
            results = self.detector.detect_emotions(frame)
            return results
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            return []
