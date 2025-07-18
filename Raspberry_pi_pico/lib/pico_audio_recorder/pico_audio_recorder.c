#include "stdio.h"
#include "pico/stdlib.h"
#include "hardware/dma.h"
#include "string.h"
#include "inttypes.h"
#include "pico_audio_recorder.pio.h"
#include "pico_audio_recorder.h"
#include "hardware/clocks.h"
#include "hardware/rtc.h"
#include "hardware/irq.h"

int16_t buff1[PIO_DMA_BUFFER_SIZE];
int16_t buff2[PIO_DMA_BUFFER_SIZE];
int16_t *fw_ptr = buff1;
static int dma_read_buff1_channel;
static int dma_read_buff2_channel;

bool new_read_data=false;

void pio_irq_read_data() {
     if (pio_interrupt_get(INMP441_pio, 0)) {
        pio_interrupt_clear(INMP441_pio, 0);
        printf("irq 0:\n");
        //dma_channel_start(dma_read_buff1_channel);
     }
     if (pio_interrupt_get(INMP441_pio, 1)) {
        pio_interrupt_clear(INMP441_pio, 1);
        printf("irq:1\n");
     }
}

void dma_handler() {
//printf("dma irq1\n");
   if (dma_channel_get_irq1_status(dma_read_buff1_channel)) {
      dma_channel_acknowledge_irq1(dma_read_buff1_channel);
      dma_channel_set_write_addr(dma_read_buff1_channel, buff1, false);
      fw_ptr = buff1;
      new_read_data=true;
   }
   if (dma_channel_get_irq1_status(dma_read_buff2_channel)) {
      dma_channel_acknowledge_irq1(dma_read_buff2_channel);
      dma_channel_set_write_addr(dma_read_buff2_channel, buff2, false);
      fw_ptr = buff2;
      new_read_data=true;
   }
}

/*!
* \brief sdio memory card pio initialize
* \param pio: pio number
* \param sm state machine
* \param cmd_pin command pin
* \param clk_pin CLK pin
* \param data_pin_base data 0~4 pins(D0~D3)
* \param clk_int Integer part of the divisor
* \param clk_frac – Fractional part in 1/256ths
*/
void inmp441_pio_init(PIO pio,  uint sm, uint sd_pin, uint sideset_pin, uint32_t freq) { // sideset_pin ws_clk
   pio_gpio_init(pio, sd_pin);
   pio_gpio_init(pio, sideset_pin);
   pio_gpio_init(pio, sideset_pin+1);
   gpio_pull_down(sd_pin);
   //gpio_pull_up(sideset_pin);
   //gpio_pull_up(sideset_pin+1);
   
   //== INMP441 SM ==
   uint offset = pio_add_program(pio, &inmp441_program);
   pio_sm_config c = inmp441_program_get_default_config(offset);
   pio_sm_set_consecutive_pindirs(pio, sm, sd_pin, 1, false);
   pio_sm_set_consecutive_pindirs(pio, sm, sideset_pin, 2, true);
   sm_config_set_in_pins(&c, sd_pin);
   sm_config_set_set_pins(&c, sideset_pin,2);
   sm_config_set_sideset_pins(&c, sideset_pin);
   sm_config_set_in_shift(&c, false, true, INMP441_BIT_PER_SAMPLE);
   sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);
   float div = clock_get_hz(clk_sys)/(freq*64*2);
   sm_config_set_clkdiv(&c, div);

   pio_sm_init(pio, sm, offset, &c);

   // initial state machine but not run
   pio_sm_set_enabled(pio, sm, false);

   //** for test only
   uint pio_irq = pio_get_index(pio)? PIO1_IRQ_0:PIO0_IRQ_0;
   pio_set_irq0_source_enabled(pio, pis_interrupt0, true);
   //pio_set_irq0_source_enabled(pio, pis_interrupt1, true);
   irq_add_shared_handler(pio_irq, pio_irq_read_data, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
   irq_set_enabled(pio_irq, true);

   uint pio_base = (pio==pio0)?PIO0_BASE:PIO1_BASE;

   //==== READ DMA ===
   dma_read_buff1_channel = dma_claim_unused_channel(true);
   dma_read_buff2_channel = dma_claim_unused_channel(true);

   dma_channel_config dc1 = dma_channel_get_default_config(dma_read_buff1_channel);
   channel_config_set_write_increment(&dc1, true);
   channel_config_set_read_increment(&dc1, false);
   channel_config_set_dreq(&dc1, pio_get_dreq(pio, sm, false));
   channel_config_set_chain_to(&dc1, dma_read_buff2_channel);  
   channel_config_set_transfer_data_size(&dc1, DMA_SIZE_16); //DMA_SIZE_8,16,32


   dma_channel_configure(dma_read_buff1_channel,
            &dc1, buff1, (void*) (pio_base+PIO_RXF0_OFFSET),  // SDIO_DATA_READ_SM = 1
            PIO_DMA_BUFFER_SIZE, false); //DMA_SIZE_8 or 16 or 32

   dma_channel_config dc2 = dma_channel_get_default_config(dma_read_buff2_channel);
   channel_config_set_write_increment(&dc2, true);
   channel_config_set_read_increment(&dc2, false);
   //channel_config_set_bswap(&dc2, true);
   channel_config_set_dreq(&dc2, pio_get_dreq(pio, sm, false));
   channel_config_set_chain_to(&dc2, dma_read_buff1_channel); 
   channel_config_set_transfer_data_size(&dc2, DMA_SIZE_16); //DMA_SIZE_8,16,32

   dma_channel_configure(dma_read_buff2_channel,
             &dc2, buff2, (void*) (pio_base+PIO_RXF0_OFFSET),  // SDIO_DATA_READ_SM = 1
              PIO_DMA_BUFFER_SIZE, false); //DMA_SIZE_8 or 16 or 32
   dma_channel_set_irq1_enabled(dma_read_buff1_channel, true);
   dma_channel_set_irq1_enabled(dma_read_buff2_channel, true);

   // Configure the processor to run dma_handler() when DMA IRQ 0 is asserted
   //irq_set_exclusive_handler(DMA_IRQ_1, dma_handler);
   irq_add_shared_handler(DMA_IRQ_1, dma_handler, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
   irq_set_enabled(DMA_IRQ_1, true);

   dma_channel_set_write_addr(dma_read_buff2_channel, buff2, false);
   dma_channel_set_write_addr(dma_read_buff1_channel, buff1, true);
   pio_sm_set_enabled(pio, sm, true);
}