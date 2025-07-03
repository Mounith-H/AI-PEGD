#ifndef MODEL_HANDLER_H_
#define MODEL_HANDLER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>

// Model configuration 
#define MODEL_INPUT_SIZE 1024  // Match with SAMPLE_BUFFER_SIZE

#ifdef __cplusplus
extern "C" {
#endif

// C interface for the model handler
int model_init(void);
bool model_process_audio(const int32_t* audio_buffer, size_t buffer_size, float confidence);
float model_get_confidence(void);

#ifdef __cplusplus
}
#endif

#endif  // MODEL_HANDLER_H_
