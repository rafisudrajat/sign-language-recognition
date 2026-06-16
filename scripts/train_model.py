#!/usr/bin/env python3
"""
Model Training CLI Script
Train the sign language recognition model.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.training import ModelTrainer
from src.config import DEFAULT_MODEL_PATH


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(
        description='Sign Language Recognition - Model Training'
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        default='train_images',
        help='Directory containing training images (default: train_images)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to save the trained model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the model after training'
    )
    
    args = parser.parse_args()
    
    print(f"Starting model training")
    print(f"Training data directory: {args.train_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model output path: {args.model_path}")
    
    try:
        # Create model trainer
        trainer = ModelTrainer(train_dir=args.train_dir)
        
        # Train the model
        model, history = trainer.train_model(
            epochs=args.epochs,
            model_path=args.model_path,
            save_model=not args.no_save
        )
        
        print("Training completed successfully!")
        
        # Evaluate the model
        print("Evaluating model...")
        evaluation = trainer.evaluate_model()
        print(f"Test Loss: {evaluation['loss']:.4f}")
        print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
