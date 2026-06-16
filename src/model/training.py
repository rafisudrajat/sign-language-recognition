"""
Training Module
Handles training of sign language recognition models.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model_builder import ModelBuilder
from src.config import (
    TRAIN_IMAGES_DIR, IMAGE_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, SHUFFLE, SEED
)


class ModelTrainer:
    """
    A class for training sign language recognition models.
    """
    
    def __init__(self, train_dir: str = None, model_builder: ModelBuilder = None):
        """
        Initialize the model trainer.
        
        Args:
            train_dir: Directory containing training images.
            model_builder: ModelBuilder instance for creating models.
        """
        self.train_dir = train_dir or TRAIN_IMAGES_DIR
        self.model_builder = model_builder or ModelBuilder()
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.history = None
    
    def create_datasets(self):
        """
        Create training, validation, and test datasets.
        
        Returns:
            tuple: (train_dataset, validation_dataset, test_dataset)
        """
        # Create training dataset
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            shuffle=SHUFFLE,
            seed=SEED,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE
        )
        
        # Create validation dataset
        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            shuffle=SHUFFLE,
            seed=SEED,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE
        )
        
        # Get class names
        self.class_names = self.train_dataset.class_names
        
        # Split validation into validation and test
        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        self.test_dataset = self.validation_dataset.take(val_batches // 5)
        self.validation_dataset = self.validation_dataset.skip(val_batches // 5)
        
        # Prefetch datasets
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return self.train_dataset, self.validation_dataset, self.test_dataset
    
    def train_model(
        self,
        epochs: int = 50,
        model_path: str = None,
        save_model: bool = True
    ):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs.
            model_path: Path to save the trained model.
            save_model: Whether to save the model after training.
            
        Returns:
            tuple: (model, history)
        """
        # Create datasets
        self.create_datasets()
        
        # Print dataset information
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        print(f"Number of training batches: {tf.data.experimental.cardinality(self.train_dataset)}")
        print(f"Number of validation batches: {tf.data.experimental.cardinality(self.validation_dataset)}")
        print(f"Number of test batches: {tf.data.experimental.cardinality(self.test_dataset)}")
        
        # Build model if not already built
        if self.model_builder.get_model() is None:
            self.model_builder.build_vgg16_transfer_learning_model()
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks()
        
        # Train the model
        self.history = self.model_builder.get_model().fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save model if requested
        if save_model:
            if model_path is None:
                model_path = os.path.join("trained_models", "TL_VGG16.h5")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model_builder.save_model(model_path)
            print(f"Model saved to {model_path}")
        
        return self.model_builder.get_model(), self.history
    
    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.
        
        Returns:
            dict: Evaluation results.
        """
        if self.model_builder.get_model() is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        if self.test_dataset is None:
            self.create_datasets()
        
        results = self.model_builder.get_model().evaluate(self.test_dataset)
        
        evaluation_dict = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        return evaluation_dict
    
    def get_train_dataset(self):
        """Get the training dataset."""
        return self.train_dataset
    
    def get_validation_dataset(self):
        """Get the validation dataset."""
        return self.validation_dataset
    
    def get_test_dataset(self):
        """Get the test dataset."""
        return self.test_dataset
    
    def get_class_names(self):
        """Get the class names."""
        return self.class_names
    
    def get_history(self):
        """Get the training history."""
        return self.history
