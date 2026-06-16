"""
Image Capture Module
Handles camera operations and image capture for data collection.
"""

import cv2
import ctypes
import os
import numpy as np
from typing import Optional, Tuple
from .hand_segmentation import HandSegmenter
from src.config import (
    ROI_TOP, ROI_BOTTOM, ROI_RIGHT, ROI_LEFT,
    CAMERA_INDEX, CAMERA_API
)


class ImageCapture:
    """
    A class for capturing images from a camera with hand segmentation.
    """
    
    def __init__(
        self, 
        camera_index: int = CAMERA_INDEX,
        camera_api=None,
        roi_top: int = ROI_TOP,
        roi_bottom: int = ROI_BOTTOM,
        roi_right: int = ROI_RIGHT,
        roi_left: int = ROI_LEFT,
        window_name: str = "Capture Image"
    ):
        """
        Initialize the image capture.
        
        Args:
            camera_index: Index of the camera to use.
            camera_api: Camera API (e.g., cv2.CAP_DSHOW for Windows).
            roi_top: Top coordinate of ROI.
            roi_bottom: Bottom coordinate of ROI.
            roi_right: Right coordinate of ROI.
            roi_left: Left coordinate of ROI.
            window_name: Name of the display window.
        """
        self.camera_index = camera_index
        self.camera_api = camera_api
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_right = roi_right
        self.roi_left = roi_left
        self.window_name = window_name
        
        self.cam = None
        self.hand_segmenter = HandSegmenter()
    
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
    
    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract Region of Interest from frame.
        
        Args:
            frame: Input frame.
            
        Returns:
            ROI region of the frame.
        """
        return frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
    
    def setup_display_window(self, fullscreen: bool = True):
        """
        Setup the display window.
        
        Args:
            fullscreen: Whether to display in fullscreen mode.
        """
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        if fullscreen:
            cv2.setWindowProperty(
                self.window_name, 
                cv2.WND_PROP_FULLSCREEN, 
                cv2.WINDOW_FULLSCREEN
            )
    
    def capture_frames(
        self,
        num_background_frames: int = 300,
        output_dir: str = "train_images",
        label: str = "K",
        num_images_to_capture: int = 150,
        wait_for_space: bool = True
    ) -> None:
        """
        Capture frames from camera and save hand images.
        
        Args:
            num_background_frames: Number of frames to use for background accumulation.
            output_dir: Directory to save captured images.
            label: Label for the captured images.
            num_images_to_capture: Number of images to capture.
            wait_for_space: Whether to wait for space key to capture images.
        """
        # Initialize camera
        self.initialize_camera()
        
        # Setup display window
        self.setup_display_window()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        num_frames = 0
        num_imgs_taken = 0
        
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
            roi = self.extract_roi(frame)
            frame_copy = frame.copy()
            
            # Convert ROI to grayscale and blur
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
            
            # Background accumulation phase
            if num_frames < num_background_frames:
                self.hand_segmenter.cal_accum_avg(gray_frame)
                if num_frames <= num_background_frames - 1:
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
                # Hand segmentation phase
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
                    
                    cv2.putText(
                        frame_copy, 
                        str(num_frames), 
                        (70, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    
                    cv2.putText(
                        frame_copy, 
                        f"{num_imgs_taken} images For {label}", 
                        (200, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    
                    # Display thresholded image
                    cv2.imshow("Thresholded Hand Image", thresholded)
                    
                    if num_imgs_taken <= num_images_to_capture:
                        if wait_for_space:
                            cv2.putText(
                                frame_copy, 
                                "Press space to capture the ROI", 
                                (80, 600), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, 
                                (0, 0, 255), 
                                2
                            )
                            k = cv2.waitKey(1)
                            if k % 256 == 32:  # Space key
                                # Save thresholded image
                                img_path = os.path.join(output_dir, label)
                                os.makedirs(img_path, exist_ok=True)
                                img_filename = os.path.join(img_path, f"{label}{num_imgs_taken}.jpg")
                                cv2.imwrite(img_filename, thresholded)
                                num_imgs_taken += 1
                        else:
                            # Auto-capture without waiting for space
                            img_path = os.path.join(output_dir, label)
                            os.makedirs(img_path, exist_ok=True)
                            img_filename = os.path.join(img_path, f"{label}{num_imgs_taken}.jpg")
                            cv2.imwrite(img_filename, thresholded)
                            num_imgs_taken += 1
                    else:
                        break
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
                "Sign Language Recognition Data Collection", 
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
