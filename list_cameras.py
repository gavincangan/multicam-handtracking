#!/usr/bin/env python3
import cv2
import time

def list_available_cameras():
    """
    Checks camera indices 0-9 and prints information about available cameras.
    """
    print("Searching for available cameras...")
    found_cameras = 0
    
    for idx in range(10):  # Check indices 0-9
        print(f"Checking camera index {idx}...", end=" ", flush=True)
        cap = cv2.VideoCapture(idx)
        
        # Small delay to allow camera to initialize
        time.sleep(0.1)
        
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Read a test frame
            ret, frame = cap.read()
            status = "OK" if ret else "Error reading frame"
            
            # Get backend name if possible
            backend = "Unknown"
            try:
                backend = cap.getBackendName()
            except:
                pass
                
            # Print camera information
            print("FOUND")
            print(f"  - Camera {idx}:")
            print(f"    Resolution: {width}x{height}")
            print(f"    FPS: {fps}")
            print(f"    Backend: {backend}")
            print(f"    Status: {status}")
            
            found_cameras += 1
            
            # Clean up
            cap.release()
        else:
            print("Not available")
        
    print(f"\nFound {found_cameras} camera(s)")
    return found_cameras

if __name__ == "__main__":
    print("Camera Detection Tool")
    print("=====================")
    num_cameras = list_available_cameras()
    print("\nTo use these cameras in the calibration stage, run the multi-camera tracker script.")

import cv2
import time

def list_available_cameras(max_cameras=10):
    """
    Check and list available camera devices.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        
    Returns:
        A list of available camera indices
    """
    available_cameras = []
    
    print("Checking for available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                # Get camera properties if possible
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera {i} is available - Resolution: {width}x{height}")
            else:
                print(f"Camera {i} opened but failed to grab frame")
            cap.release()
        else:
            print(f"Camera {i} is not available")
            
    return available_cameras

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    available = list_available_cameras()
    
    if available:
        print(f"\nFound {len(available)} available camera(s): {available}")
        print("\nYou can use these camera indices in your application.")
    else:
        print("\nNo cameras were detected!")
        print("Please check your camera connections or permissions.")

