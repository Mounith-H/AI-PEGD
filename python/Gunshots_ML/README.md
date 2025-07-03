# Gunshot Detection System (TensorFlow Version)

This system uses machine learning to detect gunshots in audio files. It has been converted from scikit-learn to TensorFlow for better deployment and portability.

## Files Overview

- `test1.py` - Main training and testing script
- `deploy_model.py` - Deployment script for using the trained model
- `requirements.txt` - Python package requirements
- `install_dependencies.bat` - Windows batch script to install dependencies
- `drive-download-20250702T171625Z-1-001/` - Dataset folder containing audio samples

## Setup Instructions

### Option 1: Using the batch script (Windows)
1. Double-click `install_dependencies.bat`
2. Wait for all packages to install

### Option 2: Manual installation
1. Open Command Prompt or PowerShell
2. Navigate to the project directory
3. Run: `pip install tf-nightly librosa numpy scikit-learn geocoder`

## Usage

### Training the Model
1. Run the training script:
   ```
   python test1.py
   ```
2. The script will:
   - Load audio data from the dataset
   - Train a TensorFlow neural network
   - Save the model as `gunshot_detection_model.h5`
   - Test the model on sample files

### Using the Trained Model (Deployment)
1. After training, run the deployment script:
   ```
   python deploy_model.py
   ```
2. Enter the path to an audio file when prompted
3. The system will analyze the audio and provide results

### Programmatic Usage
```python
from deploy_model import GunShotDetector

# Initialize detector
detector = GunShotDetector()

# Analyze an audio file
result = detector.predict_audio_file("path/to/audio.wav")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Model Details

- **Architecture**: Dense Neural Network (TensorFlow/Keras)
- **Input Features**: MFCC (Mel-frequency cepstral coefficients)
- **Classes**: Environment sounds (0) vs Gunshots (1)
- **Output**: Binary classification with confidence score

## Dataset Structure

The dataset should be organized as:
```
drive-download-20250702T171625Z-1-001/
├── environment_audio/    # Non-gunshot audio files
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── gunshots_audio/      # Gunshot audio files
    ├── file1.wav
    ├── file2.wav
    └── ...
```

## Deployment Benefits

1. **Portability**: TensorFlow models can be easily deployed across different systems
2. **Performance**: Better performance for real-time inference
3. **Scalability**: Can be deployed to cloud services, mobile devices, or edge devices
4. **Model Persistence**: Trained model is saved and can be reused without retraining

## System Requirements

- Python 3.7 or higher
- Windows/Linux/MacOS
- Sufficient RAM for audio processing (4GB+ recommended)
- Audio files in WAV format

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`
2. **Audio loading errors**: Ensure audio files are in WAV format and not corrupted
3. **Model not found**: Run `test1.py` first to train and save the model
4. **Memory issues**: Reduce the number of audio files if you encounter memory problems

## Notes

- The system includes location detection using IP geolocation
- Model performance depends on the quality and quantity of training data
- For production deployment, consider additional security and validation measures
