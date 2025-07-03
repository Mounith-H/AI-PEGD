"""
TensorFlow Lite Gunshot Detection Deployment Script
This script uses the converted .tflite model for efficient inference
"""

import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import geocoder

class GunShotDetectorLite:
    def __init__(self, model_path, scaler_path):
        """Initialize the TensorFlow Lite gunshot detector"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.sample_rate = 22050
        self.duration = 2
        self.samples_per_clip = self.sample_rate * self.duration
        
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        print(f"✅ TensorFlow Lite model loaded from: {model_path}")
        print(f"✅ Scaler loaded from: {scaler_path}")
        print(f"📊 Model input shape: {self.input_details[0]['shape']}")
        print(f"📊 Model output shape: {self.output_details[0]['shape']}")
    
    def extract_features(self, file_path, max_pad_len=87):
        """Extract enhanced features from audio file (same as training)"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            if len(audio) < self.samples_per_clip:
                audio = np.pad(audio, (0, self.samples_per_clip - len(audio)))
            else:
                audio = audio[:self.samples_per_clip]
            
            # Extract multiple types of features for better accuracy
            features = []
            
            # 1. MFCC features (20 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            features.extend(mfccs.flatten())
            
            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            if len(spectral_centroids) < max_pad_len:
                spectral_centroids = np.pad(spectral_centroids, (0, max_pad_len - len(spectral_centroids)))
            else:
                spectral_centroids = spectral_centroids[:max_pad_len]
            features.extend(spectral_centroids)
            
            # 3. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            if len(spectral_rolloff) < max_pad_len:
                spectral_rolloff = np.pad(spectral_rolloff, (0, max_pad_len - len(spectral_rolloff)))
            else:
                spectral_rolloff = spectral_rolloff[:max_pad_len]
            features.extend(spectral_rolloff)
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            if len(zcr) < max_pad_len:
                zcr = np.pad(zcr, (0, max_pad_len - len(zcr)))
            else:
                zcr = zcr[:max_pad_len]
            features.extend(zcr)
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            if chroma.shape[1] < max_pad_len:
                pad_width = max_pad_len - chroma.shape[1]
                chroma = np.pad(chroma, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                chroma = chroma[:, :max_pad_len]
            features.extend(chroma.flatten())
            
            return np.array(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def predict(self, file_path):
        """Predict if audio contains gunshot using TensorFlow Lite model"""
        # Extract features
        features = self.extract_features(file_path)
        if features is None:
            return None, None, None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Prepare input for TensorFlow Lite
        input_data = features_scaled.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get prediction
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction_prob = output_data[0][0]
        
        prediction = 1 if prediction_prob > 0.5 else 0
        prediction_label = "Gunshot" if prediction == 1 else "Environment"
        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
        
        return prediction_label, confidence, prediction_prob
    
    def raise_alert(self, prediction_label, timestamp, latitude=None, longitude=None):
        """Raise alert if gunshot is detected"""
        if prediction_label == "Gunshot":
            print("\n🚨 ALERT! Gunshot Detected!")
            print(f"🕒 Time: {timestamp}")
            if latitude and longitude:
                print(f"📍 Location: Latitude {latitude}, Longitude {longitude}")
            else:
                print("📍 Location: Unknown (Could not fetch)")
        else:
            print("\n✅ No gunshot detected.")

def main():
    """Main function to demonstrate TensorFlow Lite model usage"""
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_model_path = os.path.join(current_dir, "gunshot_detection_model.tflite")
    scaler_path = os.path.join(current_dir, "gunshot_detection_model_scaler.pkl")
    dataset_path = os.path.join(current_dir, "drive-download-20250702T171625Z-1-001")
    
    # Check if required files exist
    if not os.path.exists(tflite_model_path):
        print(f"❌ TensorFlow Lite model not found at: {tflite_model_path}")
        print("Please run test1.py first to generate the .tflite model")
        return
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler not found at: {scaler_path}")
        print("Please run test1.py first to generate the scaler")
        return
    
    # Initialize detector
    detector = GunShotDetectorLite(tflite_model_path, scaler_path)
    
    print("\n🔍 Testing TensorFlow Lite model with sample files...")
    
    # Test with sample files from dataset
    test_files = []
    possible_test_paths = [
        os.path.join(dataset_path, "environment_audio"),
        os.path.join(dataset_path, "gunshots_audio")
    ]
    
    for test_dir in possible_test_paths:
        if os.path.exists(test_dir):
            wav_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
            if wav_files:
                test_files.append(os.path.join(test_dir, wav_files[0]))
    
    if test_files:
        for test_path in test_files[:2]:  # Test with first file from each category
            print(f"\n📂 Testing file: {os.path.basename(test_path)}")
            
            # Make prediction
            prediction_label, confidence, prob = detector.predict(test_path)
            
            if prediction_label is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Try to get location
                try:
                    location = geocoder.ip('me')
                    latitude, longitude = location.latlng if location.ok else (None, None)
                except:
                    latitude, longitude = None, None
                
                print(f"🔊 Prediction: {prediction_label} (Confidence: {confidence:.2f})")
                print(f"📊 Raw probability: {prob:.4f}")
                detector.raise_alert(prediction_label, timestamp, latitude, longitude)
            else:
                print("❌ Could not process the file.")
    else:
        print("❌ No test files found in the dataset directories.")
    
    print("\n🎉 TensorFlow Lite model testing completed!")
    print(f"💡 Model size: {os.path.getsize(tflite_model_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
