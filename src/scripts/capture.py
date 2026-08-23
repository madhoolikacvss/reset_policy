# scripts/capture_camera_image.py

"""
Quick script to capture and save images from both cameras.
Usage: python capture_camera_image.py [--save_dir DIR] [--timeout SECONDS]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Add the src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reset_policy.perception.cameras import CameraPair, CameraConfig, save_rgb


def capture_images(save_dir: str = "captured_images", timeout: float = 5.0):
    """
    Capture images from both cameras and save them.
    
    Args:
        save_dir: Directory to save images
        timeout: Timeout in seconds to wait for camera frames
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing cameras...")
    print(f"  External: /dev/video4")
    print(f"  Wrist: /dev/video10")
    
    cameras = None
    
    try:
        # Initialize cameras
        cameras = CameraPair()
        
        print(f"Waiting for camera frames (timeout: {timeout}s)...")
        
        # Read frames
        external, wrist = cameras.read(timeout=timeout)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save external camera image using the module-level save_rgb function
        external_path = save_path / f"external_{timestamp}.jpg"
        save_rgb(external_path, external)
        print(f"✅ External camera saved: {external_path}")
        
        # Save wrist camera image
        wrist_path = save_path / f"wrist_{timestamp}.jpg"
        save_rgb(wrist_path, wrist)
        print(f"✅ Wrist camera saved: {wrist_path}")
        
        # Show image info
        print(f"\nImage info:")
        print(f"  External: {external.shape[1]}x{external.shape[0]} pixels")
        print(f"  Wrist: {wrist.shape[1]}x{wrist.shape[0]} pixels")
        
        print("\n✅ Capture complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    finally:
        # Close cameras
        if cameras is not None:
            cameras.close()
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture images from both cameras")
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="captured_images",
        help="Directory to save images (default: captured_images)"
    )
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=5.0,
        help="Timeout in seconds to wait for camera frames (default: 5.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Image width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Image height (default: 720)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Camera FPS (default: 15)"
    )
    
    args = parser.parse_args()
    
    # Create config with optional resolution
    config = CameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    
    sys.exit(capture_images(args.save_dir, args.timeout))


if __name__ == "__main__":
    main()
