# RP2040 Gunshot Detection Model Deployment Guide

This document explains how to deploy the quantized TensorFlow Lite model for gunshot detection on the RP2040 Pico microcontroller.

## ✅ Current Optimization Status

**Your model is already optimized for RP2040!** The `test1.py` script has generated:

1. **✅ Quantized TensorFlow Lite model** (`gunshot_detection_model.tflite` - 820 KB)
2. **✅ C header file** (`gunshot_detection_model.h` - ready for compilation)
3. **✅ INT8 quantization** (91.6% size reduction)
4. **✅ CMSIS-NN compatibility** for ARM Cortex-M0+
5. **✅ Example RP2040 implementation** (`rp2040_gunshot_detection.cpp`)

## Generated Files

After running `test1.py`, the following files are generated for RP2040 deployment:

1. **`gunshot_detection_model.tflite`** (821 KB) - Quantized TensorFlow Lite model
2. **`gunshot_detection_model.h`** (5.0 MB) - C header file for RP2040 compilation
3. **`gunshot_detection_model_scaler.pkl`** - Feature scaler (for reference)

## Model Specifications

- **Model Type**: Binary classifier (Gunshot vs Environment)
- **Input Shape**: (1, 3045) - Extracted audio features
- **Output Shape**: (1, 1) - Prediction probability
- **Quantization**: INT8 (optimized for Arm Cortex-M0+)
- **Size Reduction**: 91.6% smaller than original model
- **Accuracy**: 97.8% on test data

## Feature Extraction Requirements

The model expects 3045 features extracted from 2-second audio clips at 22.05 kHz:
- MFCC features (20 coefficients × 87 frames = 1740)
- Spectral centroid (87 features)
- Spectral rolloff (87 features)
- Zero crossing rate (87 features)
- Chroma features (12 coefficients × 87 frames = 1044)

## Using with RP2040

### 1. Include the Model Header

```c
#include "gunshot_detection_model.h"
```

### 2. Set up TensorFlow Lite Micro

```c
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
```

### 3. Initialize the Model

```c
// Set up logging
static tflite::MicroErrorReporter micro_error_reporter;

// Map the model into a usable data structure
const tflite::Model* model = tflite::GetModel(tflite_model);

// Pull in only the operation implementations we need
static tflite::AllOpsResolver resolver;

// Build an interpreter to run the model with
constexpr int kTensorArenaSize = 100 * 1024; // Adjust based on your needs
static uint8_t tensor_arena[kTensorArenaSize];
static tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

// Allocate memory from the tensor_arena for the model's tensors
interpreter.AllocateTensors();
```

### 4. Run Inference

```c
// Get input and output tensors
TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// Copy your feature array to input tensor
// Note: Input is quantized INT8, so you may need to quantize your features
for (int i = 0; i < 3045; i++) {
    input->data.int8[i] = quantized_features[i];
}

// Run inference
TfLiteStatus invoke_status = interpreter.Invoke();

// Get prediction
int8_t prediction_quantized = output->data.int8[0];
// Convert back to probability if needed
float prediction_prob = (prediction_quantized - output_zero_point) * output_scale;
```

## Audio Processing Pipeline for RP2040

1. **Audio Capture**: Use I2S microphone or ADC to capture 2-second audio clips
2. **Feature Extraction**: Implement lightweight feature extraction (may need optimized versions)
3. **Quantization**: Convert features to INT8 format expected by the model
4. **Inference**: Run the TensorFlow Lite model
5. **Post-processing**: Apply threshold and trigger alerts

## Memory Requirements

- **Flash**: ~820 KB for the model
- **RAM**: ~100 KB for tensor arena (adjust based on testing)
- **Additional**: Memory for audio buffer and feature extraction

## Performance Considerations

- **Quantization**: Model uses INT8 quantization for optimal performance on Cortex-M0+
- **CMSIS-NN**: Use pico-tflmicro library with CMSIS-NN optimizations
- **Feature Extraction**: May need to optimize or reduce features for real-time processing

## Build Instructions

1. Include the generated `gunshot_detection_model.h` in your RP2040 project
2. Link against pico-tflmicro library
3. Ensure sufficient flash and RAM allocation
4. Implement audio capture and feature extraction pipeline

## Testing

The model achieved:
- **Test Accuracy**: 97.8%
- **Precision**: 97.9%
- **Recall**: 97.7%
- **F1 Score**: 97.8%

## Notes

- The model is trained on specific audio dataset and may need retraining for different environments
- Feature extraction is computationally intensive and may need optimization for real-time use
- Consider using DMA for audio capture to reduce CPU load
- Implement proper power management for battery-powered deployments

## Troubleshooting

1. **Memory Issues**: Reduce `kTensorArenaSize` or optimize model further
2. **Performance**: Profile feature extraction and consider reducing feature set
3. **Accuracy**: Retrain model with data from target deployment environment
