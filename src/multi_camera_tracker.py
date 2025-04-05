##################################################################
# Enhanced Multi-Camera Hand Tracker
##################################################################

import cv2
import numpy as np
import time

from .constants import *
from .camera_config import CameraConfig
from .single_camera_tracker import SingleCameraTracker
from .oakd_handler import OAKDCameraTracker
from .hand_pose_fuser import EnhancedHandPoseFuser
from .hand_visualizer import HandVisualizer3D
from .camera_calibrator import PyTorchCameraCalibrator


class EnhancedMultiCameraHandTracker:
    """Main class for multi-camera hand tracking (with optional OAK-D)."""
    
    def __init__(self, config_path=CONFIG_PATH, show_visualizations=True):
        self.config = CameraConfig.load_config(config_path)
        self.primary_camera_idx = CameraConfig.get_primary_camera_idx(self.config)
        self.camera_ids = [cam["id"] for cam in self.config["cameras"]]
        self.num_cameras = len(self.camera_ids)
        self.show_visualizations = show_visualizations
        
        print(f"Primary camera index: {self.primary_camera_idx}")
        print(f"Using cameras: {self.camera_ids}")
        
        # Initialize camera trackers
        self.camera_trackers = []
        for cam_cfg in self.config["cameras"]:
            cam_id = cam_cfg["id"]
            cam_type = cam_cfg.get("type", CAMERA_TYPE_STANDARD)
            intrinsics = cam_cfg.get("intrinsics", DEFAULT_INTRINSICS.copy())
            
            if cam_type == CAMERA_TYPE_OAKD:
                # If you have actual device_info, pass it here. 
                # This example passes None or a dummy index.
                # Adjust logic as needed to pick the correct OAK-D device.
                tracker = OAKDCameraTracker(
                    device_info=None,    # In real usage, you’d find matching device_info
                    index=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            else:
                tracker = SingleCameraTracker(
                    camera_id=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            self.camera_trackers.append(tracker)
        
        # Calibrator
        self.calibrator = PyTorchCameraCalibrator(self.num_cameras, self.primary_camera_idx)
        
        # Pose fuser
        self.pose_fuser = EnhancedHandPoseFuser(self.num_cameras, self.primary_camera_idx)
        
        # State & control
        self.state = "INITIALIZING"  # or CALIBRATING, TRACKING
        self.running = False
        self.calibration_start_time = None
        
        # Visualizer
        self.visualizer = HandVisualizer3D() if show_visualizations else None
    
    def start(self):
        # Start camera trackers
        for t in self.camera_trackers:
            t.start()
        
        self.running = True
        self.state = "INITIALIZING"
        print("Initializing - press SPACE to start calibration.")
        
        try:
            while self.running:
                self.update()
                self.display()
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False
                elif key == 32:  # SPACE
                    if self.state == "INITIALIZING":
                        self.start_calibration()
                    elif self.state == "CALIBRATING" and self.calibrator.is_calibrated:
                        self.state = "TRACKING"
                        print("Calibration complete - now tracking")
        finally:
            self.stop()
    
    def start_calibration(self):
        self.state = "CALIBRATING"
        self.calibration_start_time = time.time()
        print("Starting calibration - move your hand around the views.")
    
    def update(self):
        observations = {}
        for idx, tracker in enumerate(self.camera_trackers):
            lm, conf = tracker.get_hand_landmarks()
            observations[idx] = (lm, conf)
        
        if self.state == "CALIBRATING":
            # Add samples
            for idx, (lm, conf) in observations.items():
                if lm is not None:
                    self.calibrator.add_sample(idx, lm, conf)
                    
            # Add 0.1 second delay to allow hand to move between frames
            time.sleep(0.1)
                    
            # Check progress
            self.calibrator.check_calibration_progress()
            
            # Possibly auto-switch after some time if done
            if self.calibrator.is_calibrated:
                if (time.time() - self.calibration_start_time) > 5:
                    self.state = "TRACKING"
                    print("Calibration complete - now tracking.")
        
        elif self.state == "TRACKING":
            if self.calibrator.is_calibrated:
                camera_transforms = [
                    self.calibrator.get_transform_matrix(i) for i in range(self.num_cameras)
                ]
                fused_pose = self.pose_fuser.update(observations, camera_transforms)
                if fused_pose is not None and self.visualizer is not None:
                    # Get landmarks in fused world space
                    landmarks_3d = self.pose_fuser.get_hand_landmarks_from_pose(fused_pose)
                    self.visualizer.update(landmarks_3d, fused_pose, camera_transforms, self.primary_camera_idx)
    
    def process_frame(self, frame, camera_idx, camera_id):
        """
        Process a single frame by adding borders and titles.
        
        Args:
            frame: The camera frame to process
            camera_idx: The index of the camera in the trackers list
            camera_id: The ID of the camera
            
        Returns:
            Processed frame with borders and titles
        """
        # Create a copy of the frame to avoid modifying the original
        processed = frame.copy()
        
        # Check if the frame is flipped (in the camera tracker)
        # The frame from camera_tracker is already flipped horizontally
        h, w = processed.shape[:2]
        
        # Add border to primary camera (5px red border)
        if camera_idx == self.primary_camera_idx:
            # Add red border (top, bottom, left, right)
            border_color = (0, 0, 255)  # BGR format - Red
            border_thickness = 10
            
            # Draw rectangle border
            cv2.rectangle(
                processed, 
                (0, 0), 
                (w-1, h-1), 
                border_color, 
                border_thickness
            )
            
            # Add "Primary Camera" title at the top
            title = "Primary Camera"
            font_scale = 1.0
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size to center it
            text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
            
            # For flipped frames, we need to mirror the x-coordinate
            # by flipping the text horizontally to make it readable
            # First create a small image with just the text and background
            text_img = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)
            
            # Draw the text on this small image
            cv2.putText(
                text_img,
                title,
                (5, text_size[1] + 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
            
            # Flip this text image horizontally
            # text_img = cv2.flip(text_img, 1)
            
            # Calculate position to place the text image
            text_x = (w - text_img.shape[1]) // 2
            text_y = border_thickness + 10
            
            # Create a black background for the text
            cv2.rectangle(
                processed,
                (text_x - 5, text_y - 5),
                (text_x + text_img.shape[1] + 5, text_y + text_img.shape[0] + 5),
                (0, 0, 0),
                -1
            )
            
            # Overlay the flipped text image onto the processed frame
            roi = processed[text_y:text_y + text_img.shape[0], text_x:text_x + text_img.shape[1]]
            # Create a mask of the text and its inverse mask
            text_gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
            _, text_mask = cv2.threshold(text_gray, 10, 255, cv2.THRESH_BINARY)
            text_mask_inv = cv2.bitwise_not(text_mask)
            
            # Black-out the area of text in ROI
            roi_bg = cv2.bitwise_and(roi, roi, mask=text_mask_inv)
            # Take only text from text_img
            text_fg = cv2.bitwise_and(text_img, text_img, mask=text_mask)
            # Put text in ROI and modify the processed image
            dst = cv2.add(roi_bg, text_fg)
            processed[text_y:text_y + text_img.shape[0], text_x:text_x + text_img.shape[1]] = dst
        else:
            # Add subtitle for secondary cameras
            subtitle = f"Camera {camera_id}"
            font_scale = 0.7
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size for positioning
            text_size = cv2.getTextSize(subtitle, font, font_scale, thickness)[0]
            
            # Create a small image for the text
            text_img = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)
            
            # Draw text on the small image
            cv2.putText(
                text_img,
                subtitle,
                (5, text_size[1] + 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
            
            # Flip this text image horizontally
            # text_img = cv2.flip(text_img, 1)
            
            # Calculate position to place the text image
            text_x = (w - text_img.shape[1]) // 2
            text_y = h - text_img.shape[0] - 20  # Position at bottom
            
            # Create a black background for the text
            cv2.rectangle(
                processed,
                (text_x - 5, text_y - 5),
                (text_x + text_img.shape[1] + 5, text_y + text_img.shape[0] + 5),
                (0, 0, 0),
                -1
            )
            
            # Overlay the flipped text image onto the processed frame
            roi = processed[text_y:text_y + text_img.shape[0], text_x:text_x + text_img.shape[1]]
            # Create a mask of the text and its inverse mask
            text_gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
            _, text_mask = cv2.threshold(text_gray, 10, 255, cv2.THRESH_BINARY)
            text_mask_inv = cv2.bitwise_not(text_mask)
            
            # Black-out the area of text in ROI
            roi_bg = cv2.bitwise_and(roi, roi, mask=text_mask_inv)
            # Take only text from text_img
            text_fg = cv2.bitwise_and(text_img, text_img, mask=text_mask)
            # Put text in ROI and modify the processed image
            dst = cv2.add(roi_bg, text_fg)
            processed[text_y:text_y + text_img.shape[0], text_x:text_x + text_img.shape[1]] = dst
            
        return processed

    def display(self):
        if not self.show_visualizations:
            return
        
        # Gather processed frames
        frames = []
        for idx, t in enumerate(self.camera_trackers):
            if t.processed_frame is not None:
                # Process the frame to add borders and titles
                camera_id = self.camera_ids[idx]
                processed_frame = self.process_frame(t.processed_frame, idx, camera_id)
                frames.append(processed_frame)
        
        if not frames:
            return
        
        # Make sure all frames are same size
        h, w = frames[0].shape[:2]
        for i in range(1, len(frames)):
            frames[i] = cv2.resize(frames[i], (w, h))
        
        if len(frames) == 1:
            grid = frames[0]
        elif len(frames) == 2:
            grid = np.hstack(frames)
        else:
            # up to 4 in a 2x2
            top_row = np.hstack(frames[:2])
            if len(frames) > 2:
                bottom_row = frames[2]
                if len(frames) > 3:
                    bottom_row = np.hstack([bottom_row, frames[3]])
                else:
                    # add black placeholder
                    bottom_row = np.hstack([bottom_row, np.zeros_like(frames[0])])
                grid = np.vstack([top_row, bottom_row])
            else:
                grid = top_row
        
        status_text = f"State: {self.state}"
        if self.state == "CALIBRATING":
            progress = int(self.calibrator.calibration_progress * 100)
            status_text += f" ({progress}%)"
        cv2.putText(
            grid,
            status_text,
            (10, grid.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Multi-Camera Hand Tracker", grid)
    
    def stop(self):
        for t in self.camera_trackers:
            t.stop()
        cv2.destroyAllWindows()
        if self.visualizer:
            self.visualizer.close()
        print("Shutting down.")