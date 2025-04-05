import cv2
import threading
import time
import mediapipe as mp

from .base_camera_tracker import BaseCameraTracker
from .constants import *


class SingleCameraTracker(BaseCameraTracker):
    """Handles tracking for a single standard webcam."""
    
    def __init__(self, camera_id, intrinsics=None, show=False):
        super().__init__(camera_id, intrinsics, show)
        
        self.cap = cv2.VideoCapture(camera_id)
        self.camera_info = self.get_camera_info()
        
        # Increase buffer size for smoother video
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Attempt to set a default resolution if needed
        if self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) < 1280:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.thread = threading.Thread(target=self.run_tracker)
    
    def get_camera_info(self):
        """Get camera information for display (if available)."""
        info = {
            "id": self.camera_id,
            "type": CAMERA_TYPE_STANDARD,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
        }
        try:
            info["backend"] = self.cap.getBackendName()
            if hasattr(cv2, 'CAP_PROP_SERIAL'):
                info["serial"] = self.cap.get(cv2.CAP_PROP_SERIAL)
            if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                info["name"] = self.cap.get(cv2.CAP_PROP_DEVICE_NAME)
        except Exception as e:
            print(f"Could not get detailed camera info: {e}")
        return info
    
    def start(self):
        self.running = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
    
    def run_tracker(self):
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print(f"Failed to read from camera {self.camera_id}")
                time.sleep(0.1)
                continue
            self.process_frame(image)