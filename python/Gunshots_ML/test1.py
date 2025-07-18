import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models

# ========== SETTINGS ==========
DATASET_PATH = r"drive-download-20250702T171625Z-1-001"
SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION
MAX_PAD_LEN = 44

# ========== FEATURE EXTRACTION ==========
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(audio) < SAMPLES_PER_CLIP:
            audio = np.pad(audio, (0, SAMPLES_PER_CLIP - len(audio)))
        else:
            audio = audio[:SAMPLES_PER_CLIP]
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs.T  # shape (44, 13)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ========== LOAD DATA ==========
def load_and_extract():
    X_env, X_gun = [], []

    for file in os.listdir(os.path.join(DATASET_PATH, "environment_audio")):
        if file.endswith(".wav"):
            f = extract_features(os.path.join(DATASET_PATH, "environment_audio", file))
            if f is not None:
                X_env.append(f)

    for file in os.listdir(os.path.join(DATASET_PATH, "gunshots_audio")):
        if file.endswith(".wav"):
            f = extract_features(os.path.join(DATASET_PATH, "gunshots_audio", file))
            if f is not None:
                X_gun.append(f)

    return np.array(X_env), np.array(X_gun)

# ========== MAIN ==========
print("Loading data...")
X_env, X_gun = load_and_extract()
X_env = resample(X_env, n_samples=len(X_gun), random_state=42)
X = np.concatenate((X_env, X_gun))
y = np.array([0]*len(X_env) + [1]*len(X_gun))

X = np.expand_dims(X, -1)  # shape: (samples, 44, 13, 1)

# ========== SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== MODEL ==========
model = models.Sequential([
    layers.Input(shape=(44, 13, 1)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(4, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ========== TRAIN ==========
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.1)

# ========== EVALUATE ==========
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# ========== QUANTIZED TFLITE EXPORT ==========
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

print("Converting to fully quantized INT8 TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("gunshot_model_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("\n✅ Saved: gunshot_model_quant.tflite")
print("\n📦 To convert to .h header for RP2040 use:")
print("xxd -i gunshot_model_quant.tflite > gunshot_model_quant.h")