// RP2040 Gunshot Detection Implementation Example
// This shows how to use the generated gunshot_detection_model.h on RP2040

#include "pico/stdlib.h"
#include "gunshot_detection_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Memory allocation for TensorFlow Lite Micro
constexpr int kTensorArenaSize = 100 * 1024; // 100KB - adjust based on testing
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Global variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Initialize the ML model
bool init_gunshot_detection() {
    // Map the model into a usable data structure
    model = tflite::GetModel(tflite_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model provided is schema version %d not equal to supported version %d.\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return false;
    }

    // Get pointers to the model's input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Model initialized successfully!\n");
    printf("Input shape: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
    printf("Input type: %d (INT8=%d)\n", input->type, kTfLiteInt8);
    printf("Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
    printf("Model size: %d bytes\n", tflite_model_len);
    
    return true;
}

// Run inference on extracted features
bool detect_gunshot(int8_t* features, int feature_count, float* confidence) {
    // Verify input size matches model expectation (3045 features)
    if (feature_count != 3045) {
        printf("Error: Expected 3045 features, got %d\n", feature_count);
        return false;
    }

    // Copy features to input tensor (already quantized to INT8)
    for (int i = 0; i < feature_count; i++) {
        input->data.int8[i] = features[i];
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke failed\n");
        return false;
    }

    // Get the output (INT8 quantized)
    int8_t output_quantized = output->data.int8[0];
    
    // Convert back to probability using quantization parameters
    float output_scale = output->params.scale;
    int32_t output_zero_point = output->params.zero_point;
    float prediction_prob = (output_quantized - output_zero_point) * output_scale;
    
    // Apply sigmoid if needed (quantized models might not include it)
    if (prediction_prob > 1.0f || prediction_prob < 0.0f) {
        prediction_prob = 1.0f / (1.0f + expf(-prediction_prob));
    }
    
    *confidence = prediction_prob > 0.5f ? prediction_prob : (1.0f - prediction_prob);
    
    return prediction_prob > 0.5f; // true = gunshot detected
}

// Main application
int main() {
    stdio_init_all();
    
    printf("RP2040 Gunshot Detection System Starting...\n");
    
    // Initialize the ML model
    if (!init_gunshot_detection()) {
        printf("Failed to initialize gunshot detection model\n");
        return -1;
    }
    
    printf("System ready for gunshot detection!\n");
    
    // Main loop - in real implementation, you would:
    // 1. Capture audio from microphone
    // 2. Extract 3045 features (MFCC, spectral, etc.)
    // 3. Quantize features to INT8
    // 4. Run detection
    
    while (true) {
        // Example: simulate feature extraction result
        // In reality, you need to implement audio capture and feature extraction
        static int8_t dummy_features[3045];
        
        // Fill with dummy data (replace with real feature extraction)
        for (int i = 0; i < 3045; i++) {
            dummy_features[i] = (int8_t)(rand() % 256 - 128); // Random INT8 values
        }
        
        float confidence;
        bool is_gunshot = detect_gunshot(dummy_features, 3045, &confidence);
        
        if (is_gunshot) {
            printf("🚨 GUNSHOT DETECTED! Confidence: %.2f%%\n", confidence * 100);
            // Trigger alert system (LED, buzzer, wireless transmission, etc.)
        } else {
            printf("✅ Environment sound. Confidence: %.2f%%\n", confidence * 100);
        }
        
        sleep_ms(2000); // Check every 2 seconds
    }
    
    return 0;
}
