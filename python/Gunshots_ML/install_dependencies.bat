@echo off
echo Installing required packages for Gunshot Detection System...
echo.

echo Installing TensorFlow Nightly (supports Python 3.13)...
pip install tf-nightly

echo Installing Librosa for audio processing...
pip install librosa>=0.9.0

echo Installing NumPy...
pip install numpy>=1.21.0

echo Installing Scikit-learn...
pip install scikit-learn>=1.1.0

echo Installing Geocoder for location services...
pip install geocoder>=1.38.0

echo.
echo Installation complete!
echo.
echo To run the training script: python test1.py
echo To run the deployment script: python deploy_model.py
echo.
pause
