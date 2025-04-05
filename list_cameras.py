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

