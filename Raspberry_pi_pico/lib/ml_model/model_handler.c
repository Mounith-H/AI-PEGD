#include "model_handler.h"
#include "gunshot_detection_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions for math operations
static inline float fminf_safe(float a, float b) {
    return (a < b) ? a : b;
}

static inline float fmaxf_safe(float a, float b) {
    return (a > b) ? a : b;
}

static inline float fabsf_safe(float x) {
    return (x < 0) ? -x : x;
}

// Global variables
static float confidence_score = 0.0f;
static bool model_initialized = false;
static uint32_t debug_counter = 0;

// Simple feature extraction for audio analysis
typedef struct {
    float rms_energy;
    float zero_crossing_rate;
    float spectral_centroid;
    float peak_amplitude;
} audio_features_t;

// Extract basic audio features from the buffer
static void extract_audio_features(const int32_t* audio_buffer, size_t buffer_size, audio_features_t* features) {
    // Calculate RMS energy
    double sum_squares = 0.0;
    int32_t max_amplitude = 0;
    int zero_crossings = 0;
    
    for (size_t i = 0; i < buffer_size; i++) {
        int32_t sample = audio_buffer[i] >> 8;  // Convert to 24-bit
        sum_squares += (double)(sample * sample);
        
        // Track peak amplitude
        int32_t abs_sample = abs(sample);
        if (abs_sample > max_amplitude) {
            max_amplitude = abs_sample;
        }
        
        // Count zero crossings
        if (i > 0) {
            int32_t prev_sample = audio_buffer[i-1] >> 8;
            if ((sample >= 0 && prev_sample < 0) || (sample < 0 && prev_sample >= 0)) {
                zero_crossings++;
            }
        }
    }
    
    features->rms_energy = sqrt(sum_squares / buffer_size) / 8388608.0f;  // Normalize
    features->peak_amplitude = max_amplitude / 8388608.0f;  // Normalize
    features->zero_crossing_rate = (float)zero_crossings / buffer_size;
    
    // Simple spectral centroid approximation
    features->spectral_centroid = features->zero_crossing_rate * 0.5f;
}

// Simple gunshot detection algorithm based on audio characteristics
static float classify_gunshot(const audio_features_t* features) {
    float confidence = 0.0f;
    
    // More granular confidence calculation based on normalized feature values
    // Each feature contributes continuously rather than in steps
    
    // Peak amplitude contribution (0.0 to 0.4)
    // Normalize peak amplitude to 0-1 range, then scale to 0-0.4
    float peak_contribution = 0.0f;
    if (features->peak_amplitude > 0.001f) {
        // Logarithmic scaling for better sensitivity
        float normalized_peak = fminf_safe(1.0f, features->peak_amplitude / 0.1f);
        peak_contribution = 0.4f * normalized_peak;
    }
    
    // RMS energy contribution (0.0 to 0.3)
    float rms_contribution = 0.0f;
    if (features->rms_energy > 0.001f) {
        float normalized_rms = fminf_safe(1.0f, features->rms_energy / 0.05f);
        rms_contribution = 0.3f * normalized_rms;
    }
    
    // Zero crossing rate contribution (0.0 to 0.2)
    float zcr_contribution = 0.0f;
    if (features->zero_crossing_rate > 0.01f && features->zero_crossing_rate < 0.8f) {
        // Bell curve: peaks around 0.1-0.3 ZCR (typical for gunshots)
        float optimal_zcr = 0.2f;
        float zcr_distance = fabsf_safe(features->zero_crossing_rate - optimal_zcr);
        float zcr_score = fmaxf_safe(0.0f, 1.0f - (zcr_distance / 0.3f));
        zcr_contribution = 0.2f * zcr_score;
    }
    
    // Spectral characteristics contribution (0.0 to 0.1)
    float spectral_contribution = 0.0f;
    if (features->spectral_centroid > 0.01f && features->spectral_centroid < 0.5f) {
        float normalized_spectral = fminf_safe(1.0f, features->spectral_centroid / 0.3f);
        spectral_contribution = 0.1f * normalized_spectral;
    }
    
    // Sum all contributions
    confidence = peak_contribution + rms_contribution + zcr_contribution + spectral_contribution;
    
    // Ensure confidence is within [0, 1]
    if (confidence > 1.0f) confidence = 1.0f;
    if (confidence < 0.0f) confidence = 0.0f;
    
    return confidence;
}

// Initialize the model (simplified version without TensorFlow Lite)
int model_init(void) {
    // Check if the model data is available
    if (tflite_model == NULL) {
        return -1; // Model data not available
    }
    
    // For now, just verify we can access the model data
    // In a full implementation, this would initialize TensorFlow Lite Micro
    confidence_score = 0.0f;
    model_initialized = true;
    
    return 0; // Success
}

// Process audio data for gunshot detection using feature-based classification
bool model_process_audio(const int32_t* audio_buffer, size_t buffer_size, float confidence) {
    if (!model_initialized || audio_buffer == NULL || buffer_size == 0) {
        confidence_score = 0.0f;
        return false;
    }
    
    // Extract audio features
    audio_features_t features;
    extract_audio_features(audio_buffer, buffer_size, &features);
    
    // Classify using simple feature-based approach
    confidence_score = classify_gunshot(&features);
    
    // Return true if confidence exceeds detection threshold (lowered threshold)
    return confidence_score > confidence;  // Much lower threshold for testing
}

// Get the current confidence score
float model_get_confidence(void) {
    return confidence_score;
}
