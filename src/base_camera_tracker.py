##################################################################
#  Base Camera Tracker
##################################################################


import threading
import numpy as np
import time
import os

import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

from .constants import *
from .camera_config import CameraConfig


class BaseCameraTracker:
    """Abstract base class for camera tracking."""
    
    def __init__(self, camera_id, intrinsics=None, show=False):
        self.camera_id = camera_id
        self.intrinsics = intrinsics or DEFAULT_INTRINSICS.copy()
        self.show = show
        self.running = False
        self.frame = None             # Current frame
        self.processed_frame = None   # Frame with visualizations
        self.hand_landmarks = None    # Latest hand landmarks
        self.detection_confidence = 0.0
        self.lock = threading.Lock()  # Thread safety for landmarks/confidence
        
        # MediaPipe hand detector
        self.hands = mp_hands.Hands(
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Camera info placeholder
        self.camera_info = {
            "id": self.camera_id,
            "width": 0,
            "height": 0,
            "fps": 0
        }
    
    def start(self):
        """Start the camera tracking thread."""
        raise NotImplementedError
    
    def stop(self):
        """Stop the camera tracking thread."""
        raise NotImplementedError
    
    def get_hand_landmarks(self):
        """Thread-safe getter for the latest hand landmarks."""
        with self.lock:
            if self.hand_landmarks is not None:
                return self.hand_landmarks.copy(), self.detection_confidence
            return None, 0.0
    
    def process_frame(self, image):
        """Process a frame for hand detection and store the results."""
        if image is None:
            return image
        
        # Convert to RGB for MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Restore BGR for display
        image.flags.writeable = True
        
        # Store the raw frame
        self.frame = image.copy()
        self.add_camera_info_overlay(image)
        
        if results.multi_hand_landmarks:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_handedness = results.multi_handedness[0]
            confidence = hand_handedness.classification[0].score
            
            # Extract 3D keypoints
            landmark_array = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                dtype=np.float32
            )
            
            with self.lock:
                self.hand_landmarks = landmark_array
                self.detection_confidence = confidence
            
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            
            # Add confidence text
            cv2.putText(
                image, 
                f"Confidence: {confidence:.2f}",
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
        else:
            with self.lock:
                self.hand_landmarks = None
                self.detection_confidence = 0.0
            
            cv2.putText(
                image, 
                "No Hand Detected",
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
        
        if self.show:
            # Flip for a mirrored display
            self.processed_frame = cv2.flip(image, 1)
        else:
            self.processed_frame = None
        
        return self.processed_frame
    
    def add_camera_info_overlay(self, image):
        """Add camera ID/resolution overlay to the image."""
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        camera_text = f"Camera {self.camera_id}"
        if "name" in self.camera_info:
            camera_text += f" - {self.camera_info['name']}"
        if "type" in self.camera_info:
            camera_text += f" ({self.camera_info['type']})"
        
        resolution_text = f"{self.camera_info['width']}x{self.camera_info['height']} @ {self.camera_info['fps']:.1f}fps"
        
        cv2.putText(
            image,
            camera_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 255),
            2
        )
        
        cv2.putText(
            image,
            resolution_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )