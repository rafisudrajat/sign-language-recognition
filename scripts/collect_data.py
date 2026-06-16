#!/usr/bin/env python3
"""
Data Collection CLI Script
Capture images for training the sign language recognition model.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.image_capture import ImageCapture


def main():
    """Main function to run data collection."""
    parser = argparse.ArgumentParser(
        description='Sign Language Recognition - Data Collection'
    )
    
    parser.add_argument(
        '--label', 
        type=str, 
        default='K',
        help='Label for the captured images (default: K)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='train_images',
        help='Output directory for captured images (default: train_images)'
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=150,
        help='Number of images to capture (default: 150)'
    )
    
    parser.add_argument(
        '--num-bg-frames',
        type=int,
        default=300,
        help='Number of frames for background accumulation (default: 300)'
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--auto-capture',
        action='store_true',
        help='Auto-capture without waiting for space key'
    )
    
    args = parser.parse_args()
    
    print(f"Starting data collection for label: {args.label}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images to capture: {args.num_images}")
    
    # Create image capture instance
    capture = ImageCapture(
        camera_index=args.camera_index,
        window_name=f'Capture Image - {args.label}'
    )
    
    try:
        # Run image capture
        capture.capture_frames(
            num_background_frames=args.num_bg_frames,
            output_dir=args.output_dir,
            label=args.label,
            num_images_to_capture=args.num_images,
            wait_for_space=not args.auto_capture
        )
        
        print(f"Data collection completed for label: {args.label}")
        
    except KeyboardInterrupt:
        print("Data collection interrupted by user")
        capture.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"Error during data collection: {e}")
        capture.cleanup()
        sys.exit(1)


if __name__ == '__main__':
    main()
