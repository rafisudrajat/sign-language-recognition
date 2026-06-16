"""
Prediction Module
Handles real-time prediction using trained models.
"""

import cv2
import ctypes
import numpy as np
import os
from typing import Optional, Tuple
from .model_builder import ModelBuilder
from src.data_collection.hand_segmentation import HandSegmenter
from src.config import (
    ROI_TOP, ROI_BOTTOM, ROI_RIGHT, ROI_LEFT,
    CAMERA_INDEX, CAMERA_API, WORD_DICT,
    IMAGE_SIZE, DEFAULT_MODEL_PATH
)


class SignLanguagePredictor:
    """
    A class for real-time sign language prediction using a trained model.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        camera_index: int = CAMERA_INDEX,
        camera_api=None,
        roi_top: int = ROI_TOP,
        roi_bottom: int = ROI_BOTTOM,
        roi_right: int = ROI_RIGHT,
        roi_left: int = ROI_LEFT,
        word_dict: dict = WORD_DICT,
        window_name: str = 'Sign language recognition'
    ):
        """
        Initialize the sign language predictor.
        
        Args:
            model_path: Path to the trained model file.
            camera_index: Index of the camera to use.
            camera_api: Camera API (e.g., cv2.CAP_DSHOW for Windows).
            roi_top: Top coordinate of ROI.
            roi_bottom: Bottom coordinate of ROI.
            roi_right: Right coordinate of ROI.
            roi_left: Left coordinate of ROI.
            word_dict: Dictionary mapping class indices to labels.
            window_name: Name of the display window.
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.camera_api = camera_api
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_right = roi_right
        self.roi_left = roi_left
        self.word_dict = word_dict
        self.window_name = window_name
        
        # Initialize components
        self.model_builder = ModelBuilder()
        self.hand_segmenter = HandSegmenter()
        self.cam = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        if os.path.exists(self.model_path):
            self.model_builder.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def initialize_camera(self):
        """Initialize the camera capture."""
        if self.camera_api:
            self.cam = cv2.VideoCapture(self.camera_index, self.camera_api)
        else:
            self.cam = cv2.VideoCapture(self.camera_index)
        
        if not self.cam.isOpened():
            raise RuntimeError("Could not open camera")
        
        return self.cam
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get the screen dimensions."""
        try:
            user32 = ctypes.windll.user32
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            return screen_width, screen_height
        except:
            # Fallback for non-Windows systems
            return 1920, 1080
    
    def resize_frame(self, frame: np.ndarray, screen_width: int, screen_height: int) -> np.ndarray:
        """
        Resize frame to fit screen while maintaining aspect ratio.
        
        Args:
            frame: Input frame.
            screen_width: Screen width.
            screen_height: Screen height.
            
        Returns:
            Resized frame.
        """
        frame_height, frame_width, _ = frame.shape
        scale_width = float(screen_width) / float(frame_width)
        scale_height = float(screen_height) / float(frame_height)
        
        img_scale = scale_width if scale_height > scale_width else scale_height
        new_x = int(frame.shape[1] * img_scale)
        new_y = int(frame.shape[0] * img_scale)
        
        return cv2.resize(frame, (new_x, new_y))
    
    def preprocess_image(self, thresholded: np.ndarray) -> np.ndarray:
        """
        Preprocess thresholded image for model prediction.
        
        Args:
            thresholded: Thresholded hand image.
            
        Returns:
            Preprocessed image ready for prediction.
        """
        # Resize to model input size
        thresholded = cv2.resize(thresholded, IMAGE_SIZE)
        
        # Convert grayscale to RGB
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        
        # Reshape for model input
        thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))
        
        return thresholded
    
    def predict(self, image: np.ndarray):
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image for prediction.
            
        Returns:
            tuple: (predicted_class_index, predicted_class_label)
        """
        if self.model_builder.get_model() is None:
            raise ValueError("Model has not been loaded yet")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        pred = self.model_builder.get_model().predict(processed_image)
        
        # Get predicted class
        class_index = np.argmax(pred)
        class_label = self.word_dict.get(class_index, f"Class_{class_index}")
        
        return class_index, class_label
    
    def run_real_time_prediction(self, num_background_frames: int = 300):
        """
        Run real-time sign language prediction.
        
        Args:
            num_background_frames: Number of frames to use for background accumulation.
        """
        # Initialize camera
        self.initialize_camera()
        
        # Setup display window
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.window_name, 
            cv2.WND_PROP_FULLSCREEN, 
            cv2.WINDOW_FULLSCREEN
        )
        
        num_frames = 0
        
        # For Windows systems
        try:
            input("Ready to Fetch Background? Press Enter to continue...")
        except:
            pass
        
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            
            # Flip frame to prevent inverted image
            frame = cv2.flip(frame, 1)
            
            # Resize frame
            screen_width, screen_height = self.get_screen_dimensions()
            frame = self.resize_frame(frame, screen_width, screen_height)
            
            # Extract ROI
            roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
            frame_copy = frame.copy()
            
            # Convert ROI to grayscale and blur
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            
            # Background accumulation phase
            if num_frames < num_background_frames:
                self.hand_segmenter.cal_accum_avg(gray_frame)
                cv2.putText(
                    frame_copy, 
                    "FETCHING BACKGROUND...PLEASE WAIT", 
                    (80, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 0, 255), 
                    2
                )
            else:
                # Hand segmentation and prediction phase
                hand = self.hand_segmenter.segment_hand(gray_frame)
                
                if hand is not None:
                    thresholded, hand_segment = hand
                    
                    # Draw contours around hand segment
                    cv2.drawContours(
                        frame_copy, 
                        [hand_segment + (self.roi_right, self.roi_top)], 
                        -1, 
                        (255, 0, 0), 
                        1
                    )
                    
                    # Make prediction
                    _, predicted_label = self.predict(thresholded)
                    
                    # Display prediction
                    cv2.putText(
                        frame_copy, 
                        predicted_label, 
                        (170, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    
                    # Display thresholded image
                    cv2.imshow("Thresholded Hand Image", thresholded)
                else:
                    cv2.putText(
                        frame_copy, 
                        'No hand detected...', 
                        (200, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
            
            # Draw ROI rectangle
            cv2.rectangle(
                frame_copy, 
                (self.roi_left, self.roi_top), 
                (self.roi_right, self.roi_bottom), 
                (255, 128, 0), 
                3
            )
            
            cv2.putText(
                frame_copy, 
                "Sign Language Recognition", 
                (10, 20), 
                cv2.FONT_ITALIC, 
                0.5, 
                (51, 255, 51), 
                1
            )
            
            # Display frame
            cv2.imshow(self.window_name, frame_copy)
            
            # Increment frame counter
            num_frames += 1
            
            # Close with Esc key
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Release camera and destroy all windows."""
        if self.cam:
            self.cam.release()
        cv2.destroyAllWindows()
    
    def get_model(self):
        """Get the current model."""
        return self.model_builder.get_model()
    
    def set_model(self, model):
        """Set the model."""
        self.model_builder.set_model(model)
