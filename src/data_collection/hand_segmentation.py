"""
Hand Segmentation Module
Provides functions for hand detection and segmentation from camera frames.
"""

import cv2
import numpy as np


class HandSegmenter:
    """
    A class for segmenting hand regions from video frames.
    Uses background subtraction and thresholding techniques.
    """
    
    def __init__(self, accumulated_weight=0.5):
        """
        Initialize the hand segmenter.
        
        Args:
            accumulated_weight (float): Weight for background accumulation.
                                    Higher values give more weight to newer frames.
        """
        self.background = None
        self.accumulated_weight = accumulated_weight
    
    def cal_accum_avg(self, frame):
        """
        Calculate accumulated average for background modeling.
        
        Args:
            frame (np.ndarray): Input frame in grayscale.
            
        Returns:
            None: Updates internal background state.
        """
        if self.background is None:
            self.background = frame.copy().astype("float")
            return None
        
        cv2.accumulateWeighted(frame, self.background, self.accumulated_weight)
    
    def segment_hand(self, frame, threshold=25):
        """
        Segment the hand region from the frame.
        
        Args:
            frame (np.ndarray): Input frame in grayscale.
            threshold (int): Threshold value for binarization.
            
        Returns:
            tuple: (thresholded_image, hand_contour) or None if no hand detected.
        """
        if self.background is None:
            return None
        
        # Compute absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype("uint8"), frame)
        
        # Apply threshold to get binary image
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find external contours
        contours, hierarchy = cv2.findContours(
            thresholded.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # If no contours found, return None
        if len(contours) == 0:
            return None
        
        # Get the largest contour (assumed to be the hand)
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment_max_cont)
    
    def reset_background(self):
        """Reset the background model."""
        self.background = None
    
    def get_background(self):
        """Get the current background model."""
        return self.background


def create_hand_segmenter(accumulated_weight=0.5):
    """
    Factory function to create a HandSegmenter instance.
    
    Args:
        accumulated_weight (float): Weight for background accumulation.
        
    Returns:
        HandSegmenter: Initialized hand segmenter instance.
    """
    return HandSegmenter(accumulated_weight)
