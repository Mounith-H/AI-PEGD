import os
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import pickle
import threading
import time
from collections import deque
import queue

# Load the trained model and scaler
MODEL_PATH = "gunshot_detection_model.h5"
SCALER_PATH = "gunshot_detection_model_scaler.pkl"

# Audio settings (must match training parameters)
SAMPLE_RATE = 22050
DURATION = 2  # seconds
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION
CHUNK_SIZE = 1024  # Buffer size for real-time audio

class RealTimeGunShotDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.audio_buffer = deque(maxlen=SAMPLES_PER_CLIP)
        self.is_running = False
        self.audio_queue = queue.Queue()
        
        # Load model and scaler
        self.load_model_and_scaler()
        
    def load_model_and_scaler(self):
        """Load the trained model and feature scaler"""
        try:
            # Load the TensorFlow model
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded from {MODEL_PATH}")
            else:
                print(f"❌ Model file not found: {MODEL_PATH}")
                return False
                
            # Load the scaler
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded from {SCALER_PATH}")
            else:
                print(f"❌ Scaler file not found: {SCALER_PATH}")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Error loading model or scaler: {e}")
            return False
    
    def extract_features(self, audio_data, max_pad_len=87):
        """Extract features from audio data (same as training)"""
        try:
            # Ensure audio is the right length
            if len(audio_data) < SAMPLES_PER_CLIP:
                audio_data = np.pad(audio_data, (0, SAMPLES_PER_CLIP - len(audio_data)))
            else:
                audio_data = audio_data[:SAMPLES_PER_CLIP]
            
            features = []
            
            # 1. MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=20)
            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            features.extend(mfccs.flatten())
            
            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=SAMPLE_RATE)[0]
            if len(spectral_centroids) < max_pad_len:
                spectral_centroids = np.pad(spectral_centroids, (0, max_pad_len - len(spectral_centroids)))
            else:
                spectral_centroids = spectral_centroids[:max_pad_len]
            features.extend(spectral_centroids)
            
            # 3. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=SAMPLE_RATE)[0]
            if len(spectral_rolloff) < max_pad_len:
                spectral_rolloff = np.pad(spectral_rolloff, (0, max_pad_len - len(spectral_rolloff)))
            else:
                spectral_rolloff = spectral_rolloff[:max_pad_len]
            features.extend(spectral_rolloff)
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            if len(zcr) < max_pad_len:
                zcr = np.pad(zcr, (0, max_pad_len - len(zcr)))
            else:
                zcr = zcr[:max_pad_len]
            features.extend(zcr)
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE)
            if chroma.shape[1] < max_pad_len:
                pad_width = max_pad_len - chroma.shape[1]
                chroma = np.pad(chroma, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                chroma = chroma[:, :max_pad_len]
            features.extend(chroma.flatten())
            
            return np.array(features)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_chunk = indata[:, 0]  # Take first channel
        else:
            audio_chunk = indata.flatten()
        
        # Add to queue for processing
        self.audio_queue.put(audio_chunk.copy())
    
    def process_audio(self):
        """Process audio chunks and detect gunshots"""
        print("🎧 Audio processing started...")
        
        while self.is_running:
            try:
                # Get audio chunk from queue (non-blocking)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to rolling buffer
                self.audio_buffer.extend(audio_chunk)
                
                # If we have enough samples for analysis
                if len(self.audio_buffer) >= SAMPLES_PER_CLIP:
                    # Get the last 2 seconds of audio
                    audio_data = np.array(list(self.audio_buffer)[-SAMPLES_PER_CLIP:])
                    
                    # Extract features
                    features = self.extract_features(audio_data)
                    
                    if features is not None:
                        # Scale features
                        features_scaled = self.scaler.transform(features.reshape(1, -1))
                        
                        # Make prediction
                        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
                        prediction_label = "Gunshot" if prediction_prob > 0.5 else "Environment"
                        confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
                        
                        # Only print if it's a gunshot or high confidence environment sound
                        if prediction_label == "Gunshot":
                            print(f"\n🚨 GUNSHOT DETECTED! Confidence: {confidence:.2f}")
                            print(f"⏰ Time: {time.strftime('%H:%M:%S')}")
                        elif confidence > 0.95:  # High confidence environment sounds
                            print(f"✅ Environment sound (Confidence: {confidence:.2f})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
                continue
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\n🎤 Available audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} - {device['max_input_channels']} channels")
        print()
    
    def start_detection(self, device_id=None):
        """Start real-time gunshot detection"""
        if self.model is None or self.scaler is None:
            print("❌ Model or scaler not loaded. Cannot start detection.")
            return
        
        self.is_running = True
        
        try:
            # List available devices
            self.list_audio_devices()
            
            # Start audio processing thread
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.daemon = True
            processing_thread.start()
            
            print(f"🎯 Starting real-time gunshot detection...")
            print(f"📊 Using model with 97.8% accuracy")
            print(f"🎤 Sample rate: {SAMPLE_RATE} Hz")
            print(f"⏱️  Analysis window: {DURATION} seconds")
            print(f"🔊 Speak loudly, clap, or play sounds to test!")
            print(f"❌ Press Ctrl+C to stop\n")
            
            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                device=device_id,
                callback=self.audio_callback
            ):
                print("🟢 Detection active - listening for gunshots...")
                
                # Keep running until interrupted
                try:
                    while self.is_running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping detection...")
                    
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            print("💡 Try running with a different device ID or check microphone permissions")
        finally:
            self.is_running = False
    
    def stop_detection(self):
        """Stop the detection"""
        self.is_running = False

def main():
    """Main function"""
    print("🔫 Real-Time Gunshot Detection System")
    print("=" * 50)
    
    # Create detector
    detector = RealTimeGunShotDetector()
    
    try:
        # Start detection (use default microphone)
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n👋 Detection stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    # Check if required files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("💡 Please run test1.py first to train the model")
        exit(1)
    
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler file not found: {SCALER_PATH}")
        print("💡 Please run test1.py first to generate the scaler")
        exit(1)
    
    main()
