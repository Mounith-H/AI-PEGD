// Feature Extraction for RP2040 - Simplified Implementation
// This file shows how to implement lightweight feature extraction on RP2040

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <stdint.h>
#include <math.h>

// Audio configuration (must match training parameters)
#define SAMPLE_RATE 22050
#define DURATION_SECONDS 2
#define SAMPLES_PER_CLIP (SAMPLE_RATE * DURATION_SECONDS)  // 44100 samples
#define MAX_PAD_LEN 87  // Number of frames for features

// Feature counts (must match training: total = 3045)
#define MFCC_COEFFS 20
#define MFCC_FEATURES (MFCC_COEFFS * MAX_PAD_LEN)  // 1740
#define SPECTRAL_FEATURES MAX_PAD_LEN              // 87 each for centroid, rolloff, zcr
#define CHROMA_COEFFS 12
#define CHROMA_FEATURES (CHROMA_COEFFS * MAX_PAD_LEN)  // 1044
#define TOTAL_FEATURES 3045

// Simplified MFCC calculation (you might need a more complete implementation)
void extract_mfcc_simple(float* audio, int audio_len, float* mfcc_out) {
    // This is a very simplified MFCC - you'll need a proper implementation
    // Consider using arm_math library or a lightweight DSP library
    
    // For now, this is a placeholder that demonstrates the structure
    for (int i = 0; i < MFCC_FEATURES; i++) {
        mfcc_out[i] = 0.0f; // Replace with actual MFCC calculation
    }
}

// Simplified spectral centroid calculation
void extract_spectral_centroid(float* audio, int audio_len, float* centroid_out) {
    // Simplified spectral centroid calculation
    // In practice, you'd need FFT and proper spectral analysis
    
    for (int frame = 0; frame < MAX_PAD_LEN; frame++) {
        float sum_weighted = 0.0f;
        float sum_magnitude = 0.0f;
        
        int frame_start = frame * (audio_len / MAX_PAD_LEN);
        int frame_end = (frame + 1) * (audio_len / MAX_PAD_LEN);
        
        for (int i = frame_start; i < frame_end && i < audio_len; i++) {
            float magnitude = fabsf(audio[i]);
            sum_weighted += i * magnitude;
            sum_magnitude += magnitude;
        }
        
        centroid_out[frame] = (sum_magnitude > 0) ? (sum_weighted / sum_magnitude) : 0.0f;
    }
}

// Simplified spectral rolloff calculation
void extract_spectral_rolloff(float* audio, int audio_len, float* rolloff_out) {
    // Simplified spectral rolloff - similar to centroid but different calculation
    for (int frame = 0; frame < MAX_PAD_LEN; frame++) {
        // Placeholder - implement proper spectral rolloff
        rolloff_out[frame] = 0.0f;
    }
}

// Zero crossing rate calculation
void extract_zero_crossing_rate(float* audio, int audio_len, float* zcr_out) {
    for (int frame = 0; frame < MAX_PAD_LEN; frame++) {
        int frame_start = frame * (audio_len / MAX_PAD_LEN);
        int frame_end = (frame + 1) * (audio_len / MAX_PAD_LEN);
        
        int zero_crossings = 0;
        for (int i = frame_start + 1; i < frame_end && i < audio_len; i++) {
            if ((audio[i] >= 0.0f && audio[i-1] < 0.0f) || 
                (audio[i] < 0.0f && audio[i-1] >= 0.0f)) {
                zero_crossings++;
            }
        }
        
        int frame_length = frame_end - frame_start;
        zcr_out[frame] = (frame_length > 0) ? ((float)zero_crossings / frame_length) : 0.0f;
    }
}

// Simplified chroma features
void extract_chroma_features(float* audio, int audio_len, float* chroma_out) {
    // Simplified chroma features - requires FFT and pitch analysis
    // This is a placeholder
    for (int i = 0; i < CHROMA_FEATURES; i++) {
        chroma_out[i] = 0.0f;
    }
}

// Quantize float features to INT8 for model input
void quantize_features_to_int8(float* features_float, int8_t* features_int8, int count) {
    // Apply quantization parameters from training
    // These values should match your trained model's input quantization
    float input_scale = 0.1f;      // Adjust based on your model
    int32_t input_zero_point = 0;  // Adjust based on your model
    
    for (int i = 0; i < count; i++) {
        float quantized = (features_float[i] / input_scale) + input_zero_point;
        
        // Clamp to INT8 range
        if (quantized > 127.0f) quantized = 127.0f;
        if (quantized < -128.0f) quantized = -128.0f;
        
        features_int8[i] = (int8_t)quantized;
    }
}

// Main feature extraction function
bool extract_all_features(float* audio_buffer, int8_t* output_features) {
    if (!audio_buffer || !output_features) {
        return false;
    }
    
    // Temporary float arrays for feature extraction
    static float temp_features[TOTAL_FEATURES];
    int feature_index = 0;
    
    // Extract MFCC features (1740 features)
    extract_mfcc_simple(audio_buffer, SAMPLES_PER_CLIP, &temp_features[feature_index]);
    feature_index += MFCC_FEATURES;
    
    // Extract spectral centroid (87 features)
    extract_spectral_centroid(audio_buffer, SAMPLES_PER_CLIP, &temp_features[feature_index]);
    feature_index += SPECTRAL_FEATURES;
    
    // Extract spectral rolloff (87 features)
    extract_spectral_rolloff(audio_buffer, SAMPLES_PER_CLIP, &temp_features[feature_index]);
    feature_index += SPECTRAL_FEATURES;
    
    // Extract zero crossing rate (87 features)
    extract_zero_crossing_rate(audio_buffer, SAMPLES_PER_CLIP, &temp_features[feature_index]);
    feature_index += SPECTRAL_FEATURES;
    
    // Extract chroma features (1044 features)
    extract_chroma_features(audio_buffer, SAMPLES_PER_CLIP, &temp_features[feature_index]);
    feature_index += CHROMA_FEATURES;
    
    // Verify we have the correct number of features
    if (feature_index != TOTAL_FEATURES) {
        return false;
    }
    
    // Quantize to INT8 for model input
    quantize_features_to_int8(temp_features, output_features, TOTAL_FEATURES);
    
    return true;
}

#endif // FEATURE_EXTRACTION_H
