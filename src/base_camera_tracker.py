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
import mediapipe.python.solutions.hands as mp_hands

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
        self.hand_coverage = None     # Mask showing where hand has been detected
        self.coverage_percentage = 0  # Percentage of area covered
        
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
    
    def reset_coverage(self):
        """Reset the hand coverage mask."""
        self.hand_coverage = None
        self.coverage_percentage = 0
    
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
        
        # We'll flip the image at the end after all drawings are done
        
        # Initialize hand coverage mask if needed
        if self.hand_coverage is None:
            self.hand_coverage = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Get image dimensions for text positioning
        image_height, image_width = image.shape[:2]
        
        self.add_camera_info_overlay(image, image_width)
        if results.multi_hand_landmarks:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_handedness = results.multi_handedness[0]
            confidence = hand_handedness.classification[0].score
            
            # Process hand landmarks for coverage tracking
            if self.hand_landmarks is not None:
                # Convert landmarks to pixel coordinates
                image_height, image_width = image.shape[:2]
                points = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * image_width), int(lm.y * image_height)
                    points.append([x, y])
                
                # Create convex hull and update coverage mask
                if points:
                    points = np.array(points)
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(self.hand_coverage, hull, 255)
                    
                    # Calculate coverage percentage
                    coverage_pixels = np.count_nonzero(self.hand_coverage)
                    total_pixels = self.hand_coverage.shape[0] * self.hand_coverage.shape[1]
                    self.coverage_percentage = (coverage_pixels / total_pixels) * 100
            
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
                (image_width - 240, 60), 
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
                (image_width - 240, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
        
        # Visualize the hand coverage
        if self.hand_coverage is not None:
            # Create a copy of the current frame for visualization
            coverage_display = image.copy()
            
            # Create colored overlay for coverage visualization
            coverage_color = np.zeros_like(coverage_display)
            coverage_color[:, :] = (0, 165, 255)  # Orange color for coverage
            
            # Convert mask to BGR for visualization
            mask_bgr = cv2.cvtColor(self.hand_coverage, cv2.COLOR_GRAY2BGR)
            
            # Apply color to covered areas
            colored_overlay = cv2.bitwise_and(coverage_color, mask_bgr)
            
            # Blend with original image using transparency
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(colored_overlay, alpha, image, 1.0, 0, image)
            
            # image =cv2.flip(image, 1)
            
            # Add coverage information
            cv2.putText(
                image, 
                f"Hand Coverage: {self.coverage_percentage:.1f}%",
                (image_width - 300, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Add guidance based on coverage percentage
            if self.coverage_percentage < 70:  # Threshold for "good" coverage
                cv2.putText(
                    image, 
                    "Please move your hand to cover more area!",
                    (image_width - 450, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255),  # Yellow color
                    2
                )
            else:
                cv2.putText(
                    image, 
                    "Good coverage!",
                    (image_width - 200, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0),  # Green color
                    2
                )
                
        
        if self.show:
            # Flip for a mirrored display after all drawing operations
            self.processed_frame = image.copy()
        else:
            self.processed_frame = None
        
        return self.processed_frame
    
    def add_camera_info_overlay(self, image, image_width):
        """Add camera ID/resolution overlay to the image."""
        overlay = image.copy()
        cv2.rectangle(overlay, (image_width - 400, 0), (image_width, 70), (0, 0, 0), -1)
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
            (image_width - 390, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 255),
            2
        )
        
        cv2.putText(
            image,
            resolution_text,
            (image_width - 390, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )
