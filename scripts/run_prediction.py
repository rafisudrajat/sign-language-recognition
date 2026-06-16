#!/usr/bin/env python3
"""
Real-time Prediction CLI Script
Run real-time sign language recognition.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.prediction import SignLanguagePredictor
from src.config import DEFAULT_MODEL_PATH


def main():
    """Main function to run real-time prediction."""
    parser = argparse.ArgumentParser(
        description='Sign Language Recognition - Real-time Prediction'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to the trained model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--num-bg-frames',
        type=int,
        default=300,
        help='Number of frames for background accumulation (default: 300)'
    )
    
    args = parser.parse_args()
    
    print(f"Starting real-time sign language recognition")
    print(f"Model path: {args.model_path}")
    print(f"Camera index: {args.camera_index}")
    
    try:
        # Create predictor
        predictor = SignLanguagePredictor(
            model_path=args.model_path,
            camera_index=args.camera_index
        )
        
        print("Press Enter to start background accumulation...")
        
        # Run real-time prediction
        predictor.run_real_time_prediction(
            num_background_frames=args.num_bg_frames
        )
        
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print(f"Please train the model first using: python scripts/train_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
