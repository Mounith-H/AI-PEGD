import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import geocoder
import pickle
from sklearn.preprocessing import StandardScaler

# ========== SETTINGS ==========
SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION
MODEL_PATH = "gunshot_detection_model.h5"

# ========== ENHANCED FEATURE EXTRACTION ==========
def extract_features(file_path, max_pad_len=87):  # Increased for more features
    """Extract enhanced audio features for better accuracy"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(audio) < SAMPLES_PER_CLIP:
            audio = np.pad(audio, (0, SAMPLES_PER_CLIP - len(audio)))
        else:
            audio = audio[:SAMPLES_PER_CLIP]
        
        # Extract multiple types of features for better accuracy
        features = []
        
        # 1. MFCC features (increased from 13 to 20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        features.extend(mfccs.flatten())
        
        # 2. Spectral Centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        if len(spectral_centroids) < max_pad_len:
            spectral_centroids = np.pad(spectral_centroids, (0, max_pad_len - len(spectral_centroids)))
        else:
            spectral_centroids = spectral_centroids[:max_pad_len]
        features.extend(spectral_centroids)
        
        # 3. Spectral Rolloff (shape of spectrum)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        if len(spectral_rolloff) < max_pad_len:
            spectral_rolloff = np.pad(spectral_rolloff, (0, max_pad_len - len(spectral_rolloff)))
        else:
            spectral_rolloff = spectral_rolloff[:max_pad_len]
        features.extend(spectral_rolloff)
        
        # 4. Zero Crossing Rate (speech/music discrimination)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        if len(zcr) < max_pad_len:
            zcr = np.pad(zcr, (0, max_pad_len - len(zcr)))
        else:
            zcr = zcr[:max_pad_len]
        features.extend(zcr)
        
        # 5. Chroma features (pitch class profiles)
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

# ========== ALERT SYSTEM ==========
def raise_alert(prediction_label, timestamp, latitude, longitude, confidence):
    """Raise alert if gunshot is detected"""
    if prediction_label == "Gunshot":
        print("\n🚨 ALERT! Gunshot Detected!")
        print(f"🔊 Confidence: {confidence:.2f}")
        print(f"🕒 Time: {timestamp}")
        if latitude and longitude:
            print(f"📍 Location: Latitude {latitude}, Longitude {longitude}")
        else:
            print("📍 Location: Unknown (Could not fetch)")
    else:
        print(f"\n✅ No gunshot detected. (Confidence: {confidence:.2f})")

# ========== PREDICTION CLASS ==========
class GunShotDetector:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the gunshot detector with a trained TensorFlow model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load the scaler
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print("⚠️ No scaler found. Predictions may be less accurate.")
            self.scaler = None
    
    def predict_audio_file(self, audio_path):
        """Predict if an audio file contains a gunshot"""
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        features = extract_features(audio_path)
        if features is None:
            print("❌ Could not extract features from the audio file.")
            return None
        
        # Apply scaling if available
        if self.scaler:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # TensorFlow prediction
        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        prediction_label = "Gunshot" if prediction == 1 else "Environment"
        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
        
        return {
            'prediction': prediction,
            'label': prediction_label,
            'confidence': confidence,
            'probability': prediction_prob
        }
    
    def monitor_audio_file(self, audio_path):
        """Monitor an audio file and raise alert if gunshot detected"""
        result = self.predict_audio_file(audio_path)
        
        if result is None:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try to get location
        try:
            location = geocoder.ip('me')
            latitude, longitude = location.latlng if location.ok else (None, None)
        except:
            latitude, longitude = None, None
        
        print(f"📂 File: {os.path.basename(audio_path)}")
        print(f"🔊 Prediction: {result['label']}")
        
        raise_alert(result['label'], timestamp, latitude, longitude, result['confidence'])
        
        return result

# ========== MAIN FUNCTION FOR DEPLOYMENT ==========
def main():
    """Main function for deployment usage"""
    # Example usage
    detector = GunShotDetector()
    
    # Test with a sample audio file (change this path as needed)
    test_audio_path = input("Enter path to audio file for testing (or press Enter to skip): ").strip()
    
    if test_audio_path and os.path.exists(test_audio_path):
        result = detector.monitor_audio_file(test_audio_path)
        if result:
            print(f"\nDetailed Result:")
            print(f"- Label: {result['label']}")
            print(f"- Confidence: {result['confidence']:.4f}")
            print(f"- Raw Probability: {result['probability']:.4f}")
    else:
        print("No valid audio file provided. Detector is ready for use.")
        print("\nTo use the detector in your code:")
        print("detector = GunShotDetector()")
        print("result = detector.predict_audio_file('path/to/audio.wav')")

if __name__ == "__main__":
    main()
