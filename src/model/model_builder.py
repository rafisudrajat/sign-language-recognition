"""
Model Builder Module
Handles creation and configuration of machine learning models for sign language recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from src.config import (
    IMAGE_SIZE, NUM_CLASSES, 
    VALIDATION_SPLIT, SHUFFLE, SEED
)


class ModelBuilder:
    """
    A class for building and configuring sign language recognition models.
    """
    
    def __init__(self, input_shape=None, num_classes=None):
        """
        Initialize the model builder.
        
        Args:
            input_shape: Shape of input images (height, width, channels).
            num_classes: Number of output classes.
        """
        self.input_shape = input_shape or (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        self.num_classes = num_classes or NUM_CLASSES
        self.model = None
        self.history = None
    
    def build_vgg16_transfer_learning_model(
        self,
        base_model_weights: str = 'imagenet',
        freeze_base: bool = True,
        learning_rate: float = 0.001
    ) -> tf.keras.Model:
        """
        Build a VGG16 model with transfer learning.
        
        Args:
            base_model_weights: Weights for base VGG16 model.
            freeze_base: Whether to freeze the base model layers.
            learning_rate: Learning rate for the optimizer.
            
        Returns:
            Compiled Keras model.
        """
        # Input shape
        img_shape = self.input_shape
        
        # Create base model from pre-trained VGG16
        base_model = VGG16(
            input_shape=img_shape,
            include_top=False,
            weights=base_model_weights
        )
        
        # Freeze the base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Create data augmentation layer
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2),
        ])
        
        # Build the complete model
        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        
        # Preprocess input for VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
        
        # Add base model
        x = base_model(x, training=False)
        
        # Add custom layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = tf.keras.Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_callbacks(self) -> list:
        """
        Get standard training callbacks.
        
        Returns:
            List of Keras callbacks.
        """
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        return [reduce_lr, early_stopping]
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model file.
            
        Returns:
            Loaded Keras model.
        """
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def save_model(self, model_path: str):
        """
        Save the model to file.
        
        Args:
            model_path: Path to save the model.
        """
        if self.model:
            self.model.save(model_path)
    
    def get_model(self) -> tf.keras.Model:
        """
        Get the current model.
        
        Returns:
            The current Keras model.
        """
        return self.model
    
    def set_model(self, model: tf.keras.Model):
        """
        Set the current model.
        
        Args:
            model: Keras model to set.
        """
        self.model = model
