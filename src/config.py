"""
Configuration constants for Sign Language Recognition Project
"""

# ROI (Region of Interest) Configuration
ROI_TOP = 50
ROI_BOTTOM = 300
ROI_RIGHT = 700
ROI_LEFT = 950

# Background accumulation weight
ACCUMULATED_WEIGHT = 0.5

# Image Processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Data directories
TRAIN_IMAGES_DIR = "train_images"
TRAINED_MODELS_DIR = "trained_models"

# Model Configuration
NUM_CLASSES = 5
CLASS_NAMES = ['A', 'B', 'I', 'K', 'L']
WORD_DICT = {0: 'A', 1: 'B', 2: 'I', 3: 'K', 4: 'L'}

# Camera Configuration
CAMERA_INDEX = 0
CAMERA_API = None  # cv2.CAP_DSHOW on Windows

# Training Configuration
VALIDATION_SPLIT = 0.2
SHUFFLE = True
SEED = 123

# Data Augmentation
DATA_AUGMENTATION = True

# TensorFlow Data Pipeline
AUTOTUNE = None  # Will be set to tf.data.AUTOTUNE when TensorFlow is imported

# Model paths
DEFAULT_MODEL_PATH = f"{TRAINED_MODELS_DIR}/TL_VGG16.h5"
