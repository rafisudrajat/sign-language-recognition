# Sign Language Recognition

TF4012 - Image Processing Assignment

A modular sign language recognition system that detects hand gestures in real-time using computer vision and deep learning.

## Project Structure

```
sign-language-recognition/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                 # Configuration constants
│   ├── data_collection/          # Data collection modules
│   │   ├── __init__.py
│   │   ├── hand_segmentation.py  # Hand segmentation logic
│   │   └── image_capture.py      # Camera and image capture
│   └── model/                    # Model modules
│       ├── __init__.py
│       ├── model_builder.py      # Model architecture
│       ├── training.py           # Training logic
│       └── prediction.py         # Prediction/inference logic
├── scripts/                      # CLI entry points
│   ├── __init__.py
│   ├── collect_data.py           # Data collection script
│   ├── train_model.py            # Model training script
│   └── run_prediction.py         # Real-time prediction script
├── trained_models/               # Trained model files
├── train_images/                 # Training data
├── .gitignore
├── README.md
├── requirements.txt              # Project dependencies
└── train_images.zip              # Archived training data
```

## Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Real-time Hand Detection**: Background subtraction and contour detection
- **Transfer Learning**: Uses VGG16 pre-trained model for sign classification
- **Data Augmentation**: Prevents overfitting with random transformations
- **CLI Interface**: Easy-to-use command-line scripts for all operations

## Prerequisites

- Python 3.9+
- OpenCV
- TensorFlow/Keras
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rafisudrajat/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection

Collect training images for a specific sign:

```bash
# Collect images for sign 'A'
python scripts/collect_data.py --label A --num-images 150

# Collect images for sign 'B' with auto-capture
python scripts/collect_data.py --label B --num-images 150 --auto-capture

# Use custom camera
python scripts/collect_data.py --label K --camera-index 1
```

### 2. Model Training

Train the model on collected data:

```bash
# Train with default parameters
python scripts/train_model.py

# Train with custom parameters
python scripts/train_model.py --epochs 30 --batch-size 32

# Train with custom data directory
python scripts/train_model.py --train-dir my_training_data
```

### 3. Real-time Prediction

Run real-time sign language recognition:

```bash
# Run with default model
python scripts/run_prediction.py

# Run with custom trained model
python scripts/run_prediction.py --model-path my_model.h5

# Use custom camera
python scripts/run_prediction.py --camera-index 1
```

## Configuration

Edit `src/config.py` to modify default settings:

```python
# ROI Configuration
ROI_TOP = 50
ROI_BOTTOM = 300
ROI_RIGHT = 700
ROI_LEFT = 950

# Training Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# Model Configuration
NUM_CLASSES = 5
CLASS_NAMES = ['A', 'B', 'I', 'K', 'L']
```

## Project Architecture

### Data Collection Module (`src/data_collection/`)

- **`hand_segmentation.py`**: Provides `HandSegmenter` class for hand detection
  - Background accumulation using running average
  - Thresholding and contour detection
  - Configurable threshold and ROI parameters

- **`image_capture.py`**: Provides `ImageCapture` class for camera operations
  - Real-time frame capture
  - ROI extraction and display
  - Automatic image saving with labels

### Model Module (`src/model/`)

- **`model_builder.py`**: Provides `ModelBuilder` class
  - VGG16 transfer learning implementation
  - Data augmentation pipeline
  - Model compilation and saving

- **`training.py`**: Provides `ModelTrainer` class
  - Dataset creation from directory structure
  - Training loop with callbacks
  - Model evaluation

- **`prediction.py`**: Provides `SignLanguagePredictor` class
  - Real-time prediction using trained models
  - Hand segmentation integration
  - Visual feedback with predicted labels

## Supported Signs

Currently supports the following signs:
- A
- B  
- I
- K
- L

## Extending the Project

To add support for additional signs:

1. Collect training images for the new sign:
   ```bash
   python scripts/collect_data.py --label NEW_SIGN --num-images 150
   ```

2. Update the configuration in `src/config.py`:
   ```python
   NUM_CLASSES = 6
   CLASS_NAMES = ['A', 'B', 'I', 'K', 'L', 'NEW_SIGN']
   WORD_DICT = {0: 'A', 1: 'B', 2: 'I', 3: 'K', 4: 'L', 5: 'NEW_SIGN'}
   ```

3. Retrain the model:
   ```bash
   python scripts/train_model.py
   ```

## Troubleshooting

### Camera Issues

- **Windows**: Use `--camera-api cv2.CAP_DSHOW` if camera doesn't open
- **Linux**: Ensure proper permissions for camera device
- **Mac**: Use default camera index (0)

### Model Loading Issues

- Ensure the model path is correct
- Check that the model file exists in the specified location
- Verify TensorFlow version compatibility

### Performance Issues

- Reduce image size in `config.py` for faster processing
- Use smaller batch sizes for training
- Consider using a GPU for training

## License

This project is part of an academic assignment and is provided for educational purposes.

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision operations
- Original implementation inspired by DataFlair tutorials
