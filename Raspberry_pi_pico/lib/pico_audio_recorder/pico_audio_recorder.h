
#ifndef __PICO_AUDIO_RECORDER_
#define __PICO_AUDIO_RECORDER_
#include "hardware/pio.h"

#define INMP441_pio         	pio0
#define INMP441_SM          	0
#define INMP441_SCK          	18
#define INMP441_SD           	20
#define INMP441_SAMPLE_RATE  	16000
#define INMP441_BIT_PER_SAMPLE	16
#define PIO_DMA_BUFFER_SIZE		(16000/4)   //EI_CLASSIFIER_SLICE_SIZE


#ifdef __cplusplus
extern "C" {
#endif
void inmp441_pio_init(PIO pio,  uint sm, uint sd_pin, uint sideset_pin, uint32_t freq);
#ifdef __cplusplus
}
#endif

#endif
  