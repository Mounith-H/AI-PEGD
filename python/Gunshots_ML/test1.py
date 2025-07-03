import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle

# ========== SETTINGS ==========
# Use the dataset in the current directory
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive-download-20250702T171625Z-1-001")
SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION

# ========== ENHANCED FEATURE EXTRACTION ==========
def extract_features(file_path, max_pad_len=87):  # Increased for more features
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

# ========== LOAD & EXTRACT DATA ==========
def load_and_extract(dataset_path):
    X_env, X_gun = [], []

    env_dir = os.path.join(dataset_path, "environment_audio")
    for file in os.listdir(env_dir):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(env_dir, file))
            if features is not None:
                X_env.append(features)

    gun_dir = os.path.join(dataset_path, "gunshots_audio")
    for file in os.listdir(gun_dir):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(gun_dir, file))
            if features is not None:
                X_gun.append(features)

    return np.array(X_env), np.array(X_gun)

# ========== ALERT SYSTEM ==========
def raise_alert(prediction_label):
    if prediction_label == "Gunshot":
        print("\n🚨 ALERT! Gunshot Detected!")
    else:
        print("\n✅ No gunshot detected.")

# ========== ENHANCED MODEL CREATION ==========
def create_model(input_shape):
    """Create an enhanced TensorFlow neural network model with better architecture"""
    model = keras.Sequential([
        # Input layer with batch normalization
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers with residual-like connections
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use a more sophisticated optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# ========== MODEL SAVING/LOADING ==========
def save_model(model, filepath):
    """Save the trained TensorFlow model"""
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a saved TensorFlow model"""
    if os.path.exists(filepath):
        model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model
    return None

# ========== TRAINING PHASE ==========
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gunshot_detection_model.h5")

print("Loading and processing data...")
X_env, X_gun = load_and_extract(DATASET_PATH)

if len(X_env) == 0 or len(X_gun) == 0:
    print("❌ No data found! Please check if the dataset folders contain .wav files.")
    exit()

print(f"Found {len(X_env)} environment sounds and {len(X_gun)} gunshot sounds")

# Enhanced data balancing - use more data instead of just undersampling
print("Balancing dataset with oversampling and undersampling...")

# If we have more environment sounds, use stratified sampling
if len(X_env) > len(X_gun):
    # Use more environment sounds (up to 2x gunshot sounds) for better representation
    max_env_samples = min(len(X_env), len(X_gun) * 2)
    X_env_balanced = resample(X_env, replace=False, n_samples=max_env_samples, random_state=42)
    
    # Oversample gunshot sounds to match
    X_gun_balanced = resample(X_gun, replace=True, n_samples=max_env_samples, random_state=42)
else:
    X_env_balanced = X_env
    X_gun_balanced = X_gun

print(f"Balanced dataset: {len(X_env_balanced)} environment, {len(X_gun_balanced)} gunshot sounds")

X = np.concatenate((X_env_balanced, X_gun_balanced))
y = np.array([0]*len(X_env_balanced) + [1]*len(X_gun_balanced))

# Feature scaling for better training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Check if model already exists
model = load_model(MODEL_PATH)

if model is None:
    print("Training enhanced TensorFlow model...")
    model = create_model(X_train.shape[1])
    
    # Enhanced training with callbacks for better performance
    callbacks = [
        # Stop training if no improvement for 10 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate if plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        # Save best model automatically
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model with enhanced parameters
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Increased epochs
        batch_size=16,  # Smaller batch size for better gradient updates
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!")
else:
    print("Using existing trained model...")
    # Load the scaler if it exists
    import pickle
    scaler_path = MODEL_PATH.replace('.h5', '_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded from file")

# Save the scaler for deployment
scaler_path = MODEL_PATH.replace('.h5', '_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

print("Evaluating enhanced model...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate comprehensive metrics
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

print(f"Enhanced Model Performance:")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"F1 Score:       {f1_score:.4f}")

print(classification_report(y_test, y_pred, target_names=["Environment", "Gunshot"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========== TENSORFLOW LITE CONVERSION ==========
def apply_quantization_aware_training(model, X_train, y_train, X_test, y_test):
    """Apply Quantization Aware Training (QAT) for RP2040 optimization"""
    print("\n🔄 Applying Quantization Aware Training for RP2040...")
    
    try:
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        quant_aware_model = quantize_model(model)
        
        # Compile the quantized model
        quant_aware_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Fine-tuning quantized model...")
        # Fine-tune the quantized model for a few epochs
        quant_aware_model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=5,  # Few epochs for fine-tuning
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return quant_aware_model
        
    except Exception as e:
        print(f"⚠️  Quantization Aware Training failed: {e}")
        print("🔄 Using standard model with post-training quantization instead...")
        return model

def convert_to_tflite_rp2040(model, tflite_path, X_train, use_quantization=True):
    """Convert TensorFlow model to TensorFlow Lite format optimized for RP2040"""
    print("\n🔄 Converting model to TensorFlow Lite format for RP2040...")
    
    # Create TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if use_quantization:
        print("🎯 Applying INT8 quantization for RP2040 optimization...")
        
        # Set representative dataset for quantization
        def representative_data_gen():
            for i in range(min(100, len(X_train))):  # Use up to 100 samples
                yield [X_train[i:i+1].astype(np.float32)]
        
        converter.representative_dataset = representative_data_gen
        
        try:
            # Try full INT8 quantization first
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            print("🎯 Attempting full INT8 quantization...")
            tflite_model = converter.convert()
            
        except Exception as e:
            print(f"⚠️  Full INT8 quantization failed: {e}")
            print("🔄 Trying hybrid quantization instead...")
            
            # Reset converter for hybrid quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            
            # Use hybrid quantization (weights only)
            tflite_model = converter.convert()
    else:
        print("🔄 Converting without quantization...")
        tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    try:
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved to: {tflite_path}")
        
        # Get file sizes for comparison
        h5_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        
        print(f"📊 Model size comparison:")
        print(f"   Original (.h5): {h5_size:.2f} MB")
        print(f"   TensorFlow Lite (.tflite): {tflite_size:.2f} MB")
        print(f"   Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
        
        if use_quantization:
            print(f"🎯 Model optimized for RP2040 deployment")
        
        return tflite_model
        
    except Exception as e:
        print(f"❌ Error saving TensorFlow Lite model: {e}")
        return None

def convert_tflite_to_header(tflite_path, header_path):
    """Convert .tflite file to .h header file for RP2040 compilation"""
    print(f"\n🔄 Converting {tflite_path} to C header file...")
    
    try:
        # Read the tflite file
        with open(tflite_path, 'rb') as f:
            tflite_data = f.read()
        
        # Create the header file content
        header_content = f"""// Auto-generated header file for RP2040
// Generated from: {os.path.basename(tflite_path)}
// Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

#ifndef TFLITE_MODEL_H
#define TFLITE_MODEL_H

alignas(8) const unsigned char tflite_model[] = {{
"""
        
        # Convert bytes to hex format
        hex_data = []
        for i, byte in enumerate(tflite_data):
            if i % 16 == 0:
                hex_data.append("\n  ")
            hex_data.append(f"0x{byte:02x}")
            if i < len(tflite_data) - 1:
                hex_data.append(", ")
        
        header_content += "".join(hex_data)
        header_content += f"""
}};

const unsigned int tflite_model_len = {len(tflite_data)};

#endif // TFLITE_MODEL_H
"""
        
        # Write the header file
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        print(f"✅ Header file saved to: {header_path}")
        print(f"📊 Model size: {len(tflite_data)} bytes ({len(tflite_data)/1024:.2f} KB)")
        print(f"🎯 Ready for RP2040 compilation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to header file: {e}")
        return False

def test_tflite_model(tflite_path, test_features):
    """Test the TensorFlow Lite model"""
    print("\n🧪 Testing TensorFlow Lite model...")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Test with a sample
        input_data = test_features.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_prob = output_data[0][0]
        prediction_label = "Gunshot" if prediction_prob > 0.5 else "Environment"
        confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
        
        print(f"✅ TensorFlow Lite model test successful!")
        print(f"   Prediction: {prediction_label}")
        print(f"   Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing TensorFlow Lite model: {e}")
        return False

# Apply Quantization Aware Training for RP2040 optimization
print("\n🎯 Preparing model for RP2040 deployment...")
quant_aware_model = apply_quantization_aware_training(model, X_train, y_train, X_test, y_test)

# Convert model to TensorFlow Lite with RP2040 optimizations
TFLITE_PATH = MODEL_PATH.replace('.h5', '.tflite')
HEADER_PATH = MODEL_PATH.replace('.h5', '.h')

tflite_model = convert_to_tflite_rp2040(quant_aware_model, TFLITE_PATH, X_train, use_quantization=True)

# Convert .tflite to .h header file for RP2040
if tflite_model:
    convert_tflite_to_header(TFLITE_PATH, HEADER_PATH)

def test_tflite_model_rp2040(tflite_path, test_features):
    """Test the quantized TensorFlow Lite model for RP2040"""
    print("\n🧪 Testing quantized TensorFlow Lite model...")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Prepare input data based on the model's expected input type
        input_data = test_features.reshape(1, -1)
        
        # Handle quantized input if needed
        if input_details[0]['dtype'] == np.int8:
            # For INT8 quantized models, we need to quantize the input
            input_scale = input_details[0]['quantization_parameters']['scales'][0]
            input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle quantized output if needed
        if output_details[0]['dtype'] == np.int8:
            output_scale = output_details[0]['quantization_parameters']['scales'][0]
            output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
            prediction_prob = float((output_data[0][0] - output_zero_point) * output_scale)
        else:
            prediction_prob = float(output_data[0][0])
        
        # Apply sigmoid if needed (since quantized models might not include it)
        if prediction_prob > 1.0 or prediction_prob < 0.0:
            prediction_prob = 1 / (1 + np.exp(-prediction_prob))  # Manual sigmoid
        
        prediction_label = "Gunshot" if prediction_prob > 0.5 else "Environment"
        confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
        
        print(f"✅ Quantized TensorFlow Lite model test successful!")
        print(f"   Prediction: {prediction_label}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Raw output: {prediction_prob:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing quantized TensorFlow Lite model: {e}")
        return False

if tflite_model and len(X_test) > 0:
    # Test the quantized TensorFlow Lite model with a sample
    test_tflite_model_rp2040(TFLITE_PATH, X_test[0])

# ========== PREDICT CUSTOM SOUND ==========
print("\n🔍 Testing prediction functionality...")

# Create a simple test case if no specific test file is available
test_files = []

# Check for common test file locations
possible_test_paths = [
    os.path.join(DATASET_PATH, "environment_audio"),
    os.path.join(DATASET_PATH, "gunshots_audio")
]

for test_dir in possible_test_paths:
    if os.path.exists(test_dir):
        wav_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
        if wav_files:
            test_files.append(os.path.join(test_dir, wav_files[0]))

if test_files:
    for test_path in test_files[:2]:  # Test with first file from each category
        print(f"\n📂 Testing file: {os.path.basename(test_path)}")
        
        features = extract_features(test_path)
        if features is not None:
            # Apply the same scaling as training data
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Reshape for TensorFlow prediction
            prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            prediction_label = "Gunshot" if prediction == 1 else "Environment"
            confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)

            print(f"🔊 Prediction: {prediction_label} (Confidence: {confidence:.2f})")
            raise_alert(prediction_label)
        else:
            print("❌ Could not extract features from the file.")
else:
    print("❌ No test files found in the dataset directories.")
