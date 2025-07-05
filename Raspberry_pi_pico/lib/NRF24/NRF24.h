
#ifndef __NRF24_H_
#define __NRF24_H_

#include "pico/stdlib.h"
#include "pico/stdio.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"

// NRF24L01 Register addresses
#define NRF_CONFIG_REGISTER         0x00
#define NRF_EN_AA_REGISTER          0x01
#define NRF_EN_RXADDR_REGISTER      0x02
#define NRF_SETUP_AW_REGISTER       0x03
#define NRF_SETUP_RETR_REGISTER     0x04
#define NRF_RF_CH_REGISTER          0x05
#define NRF_RF_SETUP_REGISTER       0x06
#define NRF_STATUS_REGISTER         0x07
#define NRF_OBSERVE_TX_REGISTER     0x08
#define NRF_RPD_REGISTER            0x09
#define NRF_RX_ADDR_P0_REGISTER     0x0A
#define NRF_RX_ADDR_P1_REGISTER     0x0B
#define NRF_RX_ADDR_P2_REGISTER     0x0C
#define NRF_RX_ADDR_P3_REGISTER     0x0D
#define NRF_RX_ADDR_P4_REGISTER     0x0E
#define NRF_RX_ADDR_P5_REGISTER     0x0F
#define NRF_TX_ADDR_REGISTER        0x10
#define NRF_RX_PW_P0_REGISTER       0x11
#define NRF_RX_PW_P1_REGISTER       0x12
#define NRF_RX_PW_P2_REGISTER       0x13
#define NRF_RX_PW_P3_REGISTER       0x14
#define NRF_RX_PW_P4_REGISTER       0x15
#define NRF_RX_PW_P5_REGISTER       0x16
#define NRF_FIFO_STATUS_REGISTER    0x17
#define NRF_DYNPD_REGISTER          0x1C
#define NRF_FEATURE_REGISTER        0x1D

// NRF24L01 Command definitions
#define NRF_R_REGISTER              0x00
#define NRF_W_REGISTER              0x20
#define NRF_R_RX_PAYLOAD            0x61
#define NRF_W_TX_PAYLOAD            0xA0
#define NRF_FLUSH_TX                0xE1
#define NRF_FLUSH_RX                0xE2
#define NRF_REUSE_TX_PL             0xE3
#define NRF_R_RX_PL_WID             0x60
#define NRF_W_ACK_PAYLOAD           0xA8
#define NRF_W_TX_PAYLOAD_NO_ACK     0xB0
#define NRF_NOP                     0xFF

// Status register bit definitions
#define NRF_STATUS_RX_DR            0x40
#define NRF_STATUS_TX_DS            0x20
#define NRF_STATUS_MAX_RT           0x10
#define NRF_STATUS_RX_P_NO          0x0E
#define NRF_STATUS_TX_FULL          0x01

// Config register bit definitions
#define NRF_CONFIG_MASK_RX_DR       0x40
#define NRF_CONFIG_MASK_TX_DS       0x20
#define NRF_CONFIG_MASK_MAX_RT      0x10
#define NRF_CONFIG_EN_CRC           0x08
#define NRF_CONFIG_CRCO             0x04
#define NRF_CONFIG_PWR_UP           0x02
#define NRF_CONFIG_PRIM_RX          0x01

// RF Setup register bit definitions
#define NRF_RF_SETUP_CONT_WAVE      0x80
#define NRF_RF_SETUP_RF_DR_LOW      0x20
#define NRF_RF_SETUP_PLL_LOCK       0x10
#define NRF_RF_SETUP_RF_DR_HIGH     0x08
#define NRF_RF_SETUP_RF_PWR         0x06

// FIFO Status register bit definitions
#define NRF_FIFO_STATUS_TX_REUSE    0x40
#define NRF_FIFO_STATUS_TX_FULL     0x20
#define NRF_FIFO_STATUS_TX_EMPTY    0x10
#define NRF_FIFO_STATUS_RX_FULL     0x02
#define NRF_FIFO_STATUS_RX_EMPTY    0x01

// Feature register bit definitions
#define NRF_FEATURE_EN_DPL          0x04
#define NRF_FEATURE_EN_ACK_PAY      0x02
#define NRF_FEATURE_EN_DYN_ACK      0x01

// Constants
#define NRF_MAX_PAYLOAD_SIZE        32
#define NRF_MAX_ADDR_SIZE           5
#define NRF_MIN_ADDR_SIZE           3
#define NRF_MAX_CHANNEL             125
#define NRF_MAX_PIPES               6

// Enums for better code readability
enum NRF24_DataRate {
    NRF24_DATA_RATE_1MBPS = 0,
    NRF24_DATA_RATE_2MBPS = 1,
    NRF24_DATA_RATE_250KBPS = 2
};

enum NRF24_PowerLevel {
    NRF24_POWER_LEVEL_NEG18DBM = 0,  // -18 dBm
    NRF24_POWER_LEVEL_NEG12DBM = 1,  // -12 dBm
    NRF24_POWER_LEVEL_NEG6DBM = 2,   // -6 dBm
    NRF24_POWER_LEVEL_0DBM = 3       // 0 dBm
};

enum NRF24_CRCLength {
    NRF24_CRC_DISABLED = 0,
    NRF24_CRC_8BIT = 1,
    NRF24_CRC_16BIT = 2
};

enum NRF24_AddressWidth {
    NRF24_ADDR_WIDTH_3BYTES = 1,
    NRF24_ADDR_WIDTH_4BYTES = 2,
    NRF24_ADDR_WIDTH_5BYTES = 3
};

enum NRF24_AutoRetransmitDelay {
    NRF24_ARD_250US = 0,
    NRF24_ARD_500US = 1,
    NRF24_ARD_750US = 2,
    NRF24_ARD_1000US = 3,
    NRF24_ARD_1250US = 4,
    NRF24_ARD_1500US = 5,
    NRF24_ARD_1750US = 6,
    NRF24_ARD_2000US = 7,
    NRF24_ARD_2250US = 8,
    NRF24_ARD_2500US = 9,
    NRF24_ARD_2750US = 10,
    NRF24_ARD_3000US = 11,
    NRF24_ARD_3250US = 12,
    NRF24_ARD_3500US = 13,
    NRF24_ARD_3750US = 14,
    NRF24_ARD_4000US = 15
};

// Structure to hold pipe configuration
typedef struct {
    uint8_t address[NRF_MAX_ADDR_SIZE];
    uint8_t address_width;
    uint8_t payload_size;
    bool auto_ack_enabled;
    bool dynamic_payload_enabled;
} NRF24_Pipe;



class NRF24
{
private: // Private variables
    spi_inst_t *spi;
    uint16_t sck;
    uint16_t miso;    
    uint16_t mosi;
    uint16_t csn;
    uint16_t ce;
    uint16_t irq;
    
    uint8_t status;
    uint8_t fifo_status;
    uint8_t payload_size;
    uint8_t address_width;
    uint8_t channel;
    uint8_t tx_power;
    uint8_t data_rate;
    uint8_t crc_length;
    uint8_t auto_retransmit_count;
    uint8_t auto_retransmit_delay;
    bool dynamic_payload_enabled;
    bool auto_ack_enabled;
    bool is_plus_variant;
    
    NRF24_Pipe pipes[NRF_MAX_PIPES];
    uint8_t rx_pipe_enabled;
    uint8_t tx_address[NRF_MAX_ADDR_SIZE];
    
    // Statistics
    uint16_t packets_lost;
    uint16_t packets_sent;
    uint16_t packets_received;
    uint8_t retransmit_count;

public: // Public variables (for compatibility)
    uint8_t messageLen = 32;  // Default to max payload size
    uint16_t packetsLost = 0;

private: // Private functions
    // Low-level GPIO control
    void csnLow() { gpio_put(csn, 0); }
    void csnHigh() { gpio_put(csn, 1); }
    void ceLow() { gpio_put(ce, 0); }
    void ceHigh() { gpio_put(ce, 1); }
    
    // Low-level register operations
    uint8_t readReg(uint8_t reg);
    void writeReg(uint8_t reg, uint8_t data);
    void writeReg(uint8_t reg, uint8_t *data, uint8_t size);
    void writeCommand(uint8_t cmd);
    uint8_t sendCommand(uint8_t cmd);
    
    // Internal utility functions
    void flushTx();
    void flushRx();
    void clearInterrupts();
    void updateStatus();
    void powerUp();
    void powerDown();
    void activateFeatures();
    bool isChipConnected();
    void setRegisterBit(uint8_t reg, uint8_t bit, bool value);
    bool getRegisterBit(uint8_t reg, uint8_t bit);

public: // Public functions
    // Constructor and destructor
    NRF24(spi_inst_t *spi, uint16_t sck, uint16_t mosi, uint16_t miso, uint16_t csn, uint16_t ce, uint16_t irq);
    ~NRF24();

    // Basic initialization and configuration
    bool begin();
    bool isConnected();
    void reset();
    void printDetails();
    
    // Power management
    void setPowerUp(bool power_up);
    bool isPoweredUp();
    
    // Mode switching
    void setModeRX();
    void setModeTX();
    void setModeStandby();
    bool isModeTX();
    bool isModeRX();
    
    // Channel and frequency
    void setChannel(uint8_t channel);
    uint8_t getChannel();
    void setFrequency(uint16_t frequency_mhz);
    uint16_t getFrequency();
    
    // Data rate configuration
    void setDataRate(NRF24_DataRate rate);
    NRF24_DataRate getDataRate();
    
    // Power level configuration
    void setPowerLevel(NRF24_PowerLevel level);
    NRF24_PowerLevel getPowerLevel();
    
    // CRC configuration
    void setCRCLength(NRF24_CRCLength length);
    NRF24_CRCLength getCRCLength();
    
    // Address configuration
    void setAddressWidth(NRF24_AddressWidth width);
    NRF24_AddressWidth getAddressWidth();
    void setTxAddress(uint8_t *address);
    void getTxAddress(uint8_t *address);
    
    // Pipe configuration
    void openReadingPipe(uint8_t pipe, uint8_t *address);
    void openWritingPipe(uint8_t *address);
    void closePipe(uint8_t pipe);
    void setPayloadSize(uint8_t size);
    void setPayloadSize(uint8_t pipe, uint8_t size);
    uint8_t getPayloadSize();
    uint8_t getPayloadSize(uint8_t pipe);
    
    // Auto-acknowledgment
    void setAutoAck(bool enable);
    void setAutoAck(uint8_t pipe, bool enable);
    bool isAutoAckEnabled();
    bool isAutoAckEnabled(uint8_t pipe);
    
    // Auto-retransmit configuration
    void setRetries(uint8_t delay, uint8_t count);
    void setRetryDelay(NRF24_AutoRetransmitDelay delay);
    void setRetryCount(uint8_t count);
    uint8_t getRetryCount();
    NRF24_AutoRetransmitDelay getRetryDelay();
    
    // Dynamic payload
    void enableDynamicPayloads();
    void disableDynamicPayloads();
    void enableDynamicPayload(uint8_t pipe);
    void disableDynamicPayload(uint8_t pipe);
    bool isDynamicPayloadEnabled();
    bool isDynamicPayloadEnabled(uint8_t pipe);
    
    // Payload with ACK
    void enableAckPayload();
    void disableAckPayload();
    void writeAckPayload(uint8_t pipe, uint8_t *data, uint8_t len);
    
    // Data transmission
    bool write(uint8_t *data, uint8_t len);
    bool write(uint8_t *data, uint8_t len, bool multicast);
    void startWrite(uint8_t *data, uint8_t len);
    void startWrite(uint8_t *data, uint8_t len, bool multicast);
    bool writeBlocking(uint8_t *data, uint8_t len, uint32_t timeout_ms);
    
    // Data reception
    bool available();
    bool available(uint8_t *pipe_num);
    uint8_t read(uint8_t *data, uint8_t len);
    uint8_t getDynamicPayloadSize();
    void startListening();
    void stopListening();
    
    // Status and diagnostics
    uint8_t getStatus();
    bool testCarrier();
    bool testRPD();
    uint8_t getObserveTx();
    uint8_t getLostPackets();
    uint8_t getRetransmitCount();
    void resetPacketLossCounters();
    
    // Interrupt handling
    void maskInterrupt(uint8_t interrupt);
    void unmaskInterrupt(uint8_t interrupt);
    bool isInterruptTriggered(uint8_t interrupt);
    void clearInterrupt(uint8_t interrupt);
    
    // FIFO operations
    bool isTxFifoEmpty();
    bool isTxFifoFull();
    bool isRxFifoEmpty();
    bool isRxFifoFull();
    void flushTxFifo();
    void flushRxFifo();
    
    // Advanced features
    void setCarrierWave(bool enable);
    bool isCarrierWave();
    void enterTestMode();
    void exitTestMode();
    
    // Statistics
    uint16_t getPacketsSent();
    uint16_t getPacketsReceived();
    uint16_t getPacketsLost();
    void resetStatistics();
    
    // Compatibility functions (for backward compatibility)
    void enableAck(uint8_t ack);
    void config(uint8_t *address, uint8_t channel = 2, uint8_t messageLen = 32);
    void modeRX();
    void modeTX();
    uint8_t newMessage();
    void sendMessage(uint8_t *data);
    void getMessage(uint8_t *buffer);
    uint8_t readRegister(uint8_t reg);
};

#endif
