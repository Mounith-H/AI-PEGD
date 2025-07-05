#include "NRF24.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include <string.h>
#include <stdio.h>
#include "pico/stdlib.h"

// Constructor
NRF24::NRF24(spi_inst_t *spi, uint16_t sck, uint16_t mosi, uint16_t miso, uint16_t csn, uint16_t ce, uint16_t irq)
{
    this->spi = spi;
    this->sck = sck;
    this->mosi = mosi;
    this->miso = miso;
    this->csn = csn;
    this->ce = ce;
    this->irq = irq;
    
    // Initialize default values
    this->status = 0;
    this->fifo_status = 0;
    this->payload_size = 32;
    this->address_width = 5;
    this->channel = 2;
    this->tx_power = NRF24_POWER_LEVEL_0DBM;
    this->data_rate = NRF24_DATA_RATE_1MBPS;
    this->crc_length = NRF24_CRC_16BIT;
    this->auto_retransmit_count = 3;
    this->auto_retransmit_delay = NRF24_ARD_250US;
    this->dynamic_payload_enabled = false;
    this->auto_ack_enabled = true;
    this->is_plus_variant = false;
    this->rx_pipe_enabled = 0;
    
    // Initialize statistics
    this->packets_lost = 0;
    this->packets_sent = 0;
    this->packets_received = 0;
    this->retransmit_count = 0;
    
    // Initialize pipe configurations
    for (int i = 0; i < NRF_MAX_PIPES; i++) {
        memset(pipes[i].address, 0, NRF_MAX_ADDR_SIZE);
        pipes[i].address_width = 5;
        pipes[i].payload_size = 32;
        pipes[i].auto_ack_enabled = true;
        pipes[i].dynamic_payload_enabled = false;
    }
    
    // Initialize default TX address
    memset(tx_address, 0, NRF_MAX_ADDR_SIZE);
}

// Destructor
NRF24::~NRF24()
{
    powerDown();
}

// Initialize the NRF24L01 module
bool NRF24::begin()
{
    // Initialize SPI
    spi_init(this->spi, 4000000); // Start with 4MHz for initialization
    gpio_set_function(sck, GPIO_FUNC_SPI);
    gpio_set_function(miso, GPIO_FUNC_SPI);
    gpio_set_function(mosi, GPIO_FUNC_SPI);
    
    // Initialize control pins
    gpio_init(csn);
    gpio_init(ce);
    gpio_set_dir(csn, GPIO_OUT);
    gpio_set_dir(ce, GPIO_OUT);
    
    // Initialize IRQ pin if specified
    if (irq != 0xFF) {
        gpio_init(irq);
        gpio_set_dir(irq, GPIO_IN);
        gpio_pull_up(irq);
    }
    
    // Set initial pin states
    ceLow();
    csnHigh();
    
    // Wait for chip to stabilize
    sleep_ms(5);
    
    // Check if chip is connected
    if (!isChipConnected()) {
        return false;
    }
    
    // Reset to known state
    reset();
    
    // Detect if this is a + variant
    uint8_t setup = readReg(NRF_RF_SETUP_REGISTER);
    writeReg(NRF_RF_SETUP_REGISTER, setup | 0x80);
    if (readReg(NRF_RF_SETUP_REGISTER) & 0x80) {
        is_plus_variant = true;
        writeReg(NRF_RF_SETUP_REGISTER, setup); // Restore original value
    }
    
    // Enable features for + variant
    if (is_plus_variant) {
        activateFeatures();
    }
    
    // Increase SPI speed after successful initialization
    spi_set_baudrate(this->spi, 8000000); // 8MHz for normal operation
    
    return true;
}

// Check if the chip is connected and responding
bool NRF24::isConnected()
{
    return isChipConnected();
}

// Reset the NRF24L01 to default state
void NRF24::reset()
{
    // Power down
    powerDown();
    sleep_ms(2);
    
    // Reset all registers to default values
    writeReg(NRF_CONFIG_REGISTER, 0x08);
    writeReg(NRF_EN_AA_REGISTER, 0x3F);
    writeReg(NRF_EN_RXADDR_REGISTER, 0x03);
    writeReg(NRF_SETUP_AW_REGISTER, 0x03);
    writeReg(NRF_SETUP_RETR_REGISTER, 0x03);
    writeReg(NRF_RF_CH_REGISTER, 0x02);
    writeReg(NRF_RF_SETUP_REGISTER, 0x0E);
    writeReg(NRF_STATUS_REGISTER, 0x70);
    
    // Clear FIFOs
    flushTx();
    flushRx();
    
    // Clear interrupts
    clearInterrupts();
    
    // Set default addresses
    uint8_t addr_p0_p1_default[] = {0xE7, 0xE7, 0xE7, 0xE7, 0xE7};
    uint8_t addr_p2_default = 0xC2;
    uint8_t addr_p3_default = 0xC3;
    uint8_t addr_p4_default = 0xC4;
    uint8_t addr_p5_default = 0xC5;
    
    writeReg(NRF_RX_ADDR_P0_REGISTER, addr_p0_p1_default, 5);
    writeReg(NRF_RX_ADDR_P1_REGISTER, addr_p0_p1_default, 5);
    writeReg(NRF_RX_ADDR_P2_REGISTER, addr_p2_default);
    writeReg(NRF_RX_ADDR_P3_REGISTER, addr_p3_default);
    writeReg(NRF_RX_ADDR_P4_REGISTER, addr_p4_default);
    writeReg(NRF_RX_ADDR_P5_REGISTER, addr_p5_default);
    writeReg(NRF_TX_ADDR_REGISTER, addr_p0_p1_default, 5);
    
    // Set default payload sizes
    writeReg(NRF_RX_PW_P0_REGISTER, 0x20);
    writeReg(NRF_RX_PW_P1_REGISTER, 0x20);
    writeReg(NRF_RX_PW_P2_REGISTER, 0x20);
    writeReg(NRF_RX_PW_P3_REGISTER, 0x20);
    writeReg(NRF_RX_PW_P4_REGISTER, 0x20);
    writeReg(NRF_RX_PW_P5_REGISTER, 0x20);
    
    // Reset feature register
    writeReg(NRF_FEATURE_REGISTER, 0x00);
    writeReg(NRF_DYNPD_REGISTER, 0x00);
    
    // Update internal state
    payload_size = 32;
    address_width = 5;
    channel = 2;
    dynamic_payload_enabled = false;
    auto_ack_enabled = true;
    rx_pipe_enabled = 0x03; // Pipe 0 and 1 enabled by default
}

// Low-level register operations
uint8_t NRF24::readReg(uint8_t reg)
{
    reg = NRF_R_REGISTER | (reg & 0x1F);
    uint8_t result = 0;
    csnLow();
    spi_write_blocking(spi, &reg, 1);
    spi_read_blocking(spi, 0x00, &result, 1);
    csnHigh();
    return result;
}

void NRF24::writeReg(uint8_t reg, uint8_t data)
{
    writeReg(reg, &data, 1);
}

void NRF24::writeReg(uint8_t reg, uint8_t *data, uint8_t size)
{
    reg = NRF_W_REGISTER | (reg & 0x1F);
    csnLow();
    spi_write_blocking(spi, &reg, 1);
    spi_write_blocking(spi, data, size);
    csnHigh();
}

void NRF24::writeCommand(uint8_t cmd)
{
    csnLow();
    spi_write_blocking(spi, &cmd, 1);
    csnHigh();
}

uint8_t NRF24::sendCommand(uint8_t cmd)
{
    uint8_t status = 0;
    csnLow();
    spi_write_read_blocking(spi, &cmd, &status, 1);
    csnHigh();
    return status;
}

// Internal utility functions
void NRF24::flushTx()
{
    writeCommand(NRF_FLUSH_TX);
}

void NRF24::flushRx()
{
    writeCommand(NRF_FLUSH_RX);
}

void NRF24::clearInterrupts()
{
    writeReg(NRF_STATUS_REGISTER, NRF_STATUS_RX_DR | NRF_STATUS_TX_DS | NRF_STATUS_MAX_RT);
}

void NRF24::updateStatus()
{
    status = readReg(NRF_STATUS_REGISTER);
    fifo_status = readReg(NRF_FIFO_STATUS_REGISTER);
}

void NRF24::powerUp()
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    config |= NRF_CONFIG_PWR_UP;
    writeReg(NRF_CONFIG_REGISTER, config);
    sleep_us(1500); // Wait for power up
}

void NRF24::powerDown()
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    config &= ~NRF_CONFIG_PWR_UP;
    writeReg(NRF_CONFIG_REGISTER, config);
}

void NRF24::activateFeatures()
{
    uint8_t activate_cmd = 0x50;
    uint8_t activate_data = 0x73;
    csnLow();
    spi_write_blocking(spi, &activate_cmd, 1);
    spi_write_blocking(spi, &activate_data, 1);
    csnHigh();
}

bool NRF24::isChipConnected()
{
    uint8_t setup_aw = readReg(NRF_SETUP_AW_REGISTER);
    return (setup_aw & 0x0C) == 0x00 && (setup_aw & 0x03) != 0x00;
}

void NRF24::setRegisterBit(uint8_t reg, uint8_t bit, bool value)
{
    uint8_t reg_val = readReg(reg);
    if (value) {
        reg_val |= (1 << bit);
    } else {
        reg_val &= ~(1 << bit);
    }
    writeReg(reg, reg_val);
}

bool NRF24::getRegisterBit(uint8_t reg, uint8_t bit)
{
    uint8_t reg_val = readReg(reg);
    return (reg_val & (1 << bit)) != 0;
}

// Power management
void NRF24::setPowerUp(bool power_up)
{
    if (power_up) {
        powerUp();
    } else {
        powerDown();
    }
}

bool NRF24::isPoweredUp()
{
    return getRegisterBit(NRF_CONFIG_REGISTER, 1);
}

// Mode switching
void NRF24::setModeRX()
{
    ceLow();
    setRegisterBit(NRF_CONFIG_REGISTER, 0, true); // Set PRIM_RX
    powerUp();
    ceHigh();
    sleep_us(130); // RX settling time
}

void NRF24::setModeTX()
{
    ceLow();
    setRegisterBit(NRF_CONFIG_REGISTER, 0, false); // Clear PRIM_RX
    powerUp();
    sleep_us(130); // TX settling time
}

void NRF24::setModeStandby()
{
    ceLow();
    powerUp();
}

bool NRF24::isModeTX()
{
    return !getRegisterBit(NRF_CONFIG_REGISTER, 0);
}

bool NRF24::isModeRX()
{
    return getRegisterBit(NRF_CONFIG_REGISTER, 0);
}

// Channel and frequency configuration
void NRF24::setChannel(uint8_t channel)
{
    if (channel > NRF_MAX_CHANNEL) {
        channel = NRF_MAX_CHANNEL;
    }
    this->channel = channel;
    writeReg(NRF_RF_CH_REGISTER, channel);
}

uint8_t NRF24::getChannel()
{
    return readReg(NRF_RF_CH_REGISTER);
}

void NRF24::setFrequency(uint16_t frequency_mhz)
{
    if (frequency_mhz < 2400 || frequency_mhz > 2525) {
        return; // Invalid frequency
    }
    uint8_t channel = frequency_mhz - 2400;
    setChannel(channel);
}

uint16_t NRF24::getFrequency()
{
    return 2400 + getChannel();
}

// Data rate configuration
void NRF24::setDataRate(NRF24_DataRate rate)
{
    uint8_t rf_setup = readReg(NRF_RF_SETUP_REGISTER);
    
    // Clear existing data rate bits
    rf_setup &= ~(NRF_RF_SETUP_RF_DR_LOW | NRF_RF_SETUP_RF_DR_HIGH);
    
    switch (rate) {
        case NRF24_DATA_RATE_250KBPS:
            rf_setup |= NRF_RF_SETUP_RF_DR_LOW;
            break;
        case NRF24_DATA_RATE_1MBPS:
            // Both bits clear for 1Mbps
            break;
        case NRF24_DATA_RATE_2MBPS:
            rf_setup |= NRF_RF_SETUP_RF_DR_HIGH;
            break;
    }
    
    writeReg(NRF_RF_SETUP_REGISTER, rf_setup);
    this->data_rate = rate;
}

NRF24_DataRate NRF24::getDataRate()
{
    uint8_t rf_setup = readReg(NRF_RF_SETUP_REGISTER);
    
    if (rf_setup & NRF_RF_SETUP_RF_DR_LOW) {
        return NRF24_DATA_RATE_250KBPS;
    } else if (rf_setup & NRF_RF_SETUP_RF_DR_HIGH) {
        return NRF24_DATA_RATE_2MBPS;
    } else {
        return NRF24_DATA_RATE_1MBPS;
    }
}

// Power level configuration
void NRF24::setPowerLevel(NRF24_PowerLevel level)
{
    uint8_t rf_setup = readReg(NRF_RF_SETUP_REGISTER);
    rf_setup &= ~NRF_RF_SETUP_RF_PWR; // Clear power bits
    rf_setup |= (level << 1); // Set new power level
    writeReg(NRF_RF_SETUP_REGISTER, rf_setup);
    this->tx_power = level;
}

NRF24_PowerLevel NRF24::getPowerLevel()
{
    uint8_t rf_setup = readReg(NRF_RF_SETUP_REGISTER);
    return (NRF24_PowerLevel)((rf_setup & NRF_RF_SETUP_RF_PWR) >> 1);
}

// CRC configuration
void NRF24::setCRCLength(NRF24_CRCLength length)
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    
    switch (length) {
        case NRF24_CRC_DISABLED:
            config &= ~NRF_CONFIG_EN_CRC;
            break;
        case NRF24_CRC_8BIT:
            config |= NRF_CONFIG_EN_CRC;
            config &= ~NRF_CONFIG_CRCO;
            break;
        case NRF24_CRC_16BIT:
            config |= NRF_CONFIG_EN_CRC;
            config |= NRF_CONFIG_CRCO;
            break;
    }
    
    writeReg(NRF_CONFIG_REGISTER, config);
    this->crc_length = length;
}

NRF24_CRCLength NRF24::getCRCLength()
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    
    if (!(config & NRF_CONFIG_EN_CRC)) {
        return NRF24_CRC_DISABLED;
    } else if (config & NRF_CONFIG_CRCO) {
        return NRF24_CRC_16BIT;
    } else {
        return NRF24_CRC_8BIT;
    }
}

// Address configuration
void NRF24::setAddressWidth(NRF24_AddressWidth width)
{
    uint8_t aw = width + 1; // Convert to register value (2-4)
    writeReg(NRF_SETUP_AW_REGISTER, aw);
    this->address_width = width + 2; // Convert to actual width (3-5)
}

NRF24_AddressWidth NRF24::getAddressWidth()
{
    uint8_t aw = readReg(NRF_SETUP_AW_REGISTER);
    return (NRF24_AddressWidth)(aw - 1);
}

void NRF24::setTxAddress(uint8_t *address)
{
    writeReg(NRF_TX_ADDR_REGISTER, address, address_width);
    writeReg(NRF_RX_ADDR_P0_REGISTER, address, address_width); // For auto-ack
    memcpy(tx_address, address, address_width);
}

void NRF24::getTxAddress(uint8_t *address)
{
    memcpy(address, tx_address, address_width);
}

// Pipe configuration
void NRF24::openReadingPipe(uint8_t pipe, uint8_t *address)
{
    if (pipe >= NRF_MAX_PIPES) return;
    
    // Enable the pipe
    uint8_t en_rxaddr = readReg(NRF_EN_RXADDR_REGISTER);
    en_rxaddr |= (1 << pipe);
    writeReg(NRF_EN_RXADDR_REGISTER, en_rxaddr);
    
    // Set address
    if (pipe <= 1) {
        writeReg(NRF_RX_ADDR_P0_REGISTER + pipe, address, address_width);
    } else {
        writeReg(NRF_RX_ADDR_P0_REGISTER + pipe, address[address_width - 1]);
    }
    
    // Set payload size
    writeReg(NRF_RX_PW_P0_REGISTER + pipe, payload_size);
    
    // Update internal state
    memcpy(pipes[pipe].address, address, address_width);
    pipes[pipe].address_width = address_width;
    pipes[pipe].payload_size = payload_size;
    rx_pipe_enabled |= (1 << pipe);
}

void NRF24::openWritingPipe(uint8_t *address)
{
    setTxAddress(address);
}

void NRF24::closePipe(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return;
    
    uint8_t en_rxaddr = readReg(NRF_EN_RXADDR_REGISTER);
    en_rxaddr &= ~(1 << pipe);
    writeReg(NRF_EN_RXADDR_REGISTER, en_rxaddr);
    
    rx_pipe_enabled &= ~(1 << pipe);
}

void NRF24::setPayloadSize(uint8_t size)
{
    if (size > NRF_MAX_PAYLOAD_SIZE) size = NRF_MAX_PAYLOAD_SIZE;
    this->payload_size = size;
    this->messageLen = size; // For compatibility
    
    // Update all enabled pipes
    for (uint8_t i = 0; i < NRF_MAX_PIPES; i++) {
        if (rx_pipe_enabled & (1 << i)) {
            writeReg(NRF_RX_PW_P0_REGISTER + i, size);
            pipes[i].payload_size = size;
        }
    }
}

void NRF24::setPayloadSize(uint8_t pipe, uint8_t size)
{
    if (pipe >= NRF_MAX_PIPES || size > NRF_MAX_PAYLOAD_SIZE) return;
    
    writeReg(NRF_RX_PW_P0_REGISTER + pipe, size);
    pipes[pipe].payload_size = size;
}

uint8_t NRF24::getPayloadSize()
{
    return payload_size;
}

uint8_t NRF24::getPayloadSize(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return 0;
    return pipes[pipe].payload_size;
}

// Auto-acknowledgment
void NRF24::setAutoAck(bool enable)
{
    if (enable) {
        writeReg(NRF_EN_AA_REGISTER, 0x3F); // Enable for all pipes
    } else {
        writeReg(NRF_EN_AA_REGISTER, 0x00); // Disable for all pipes
    }
    this->auto_ack_enabled = enable;
}

void NRF24::setAutoAck(uint8_t pipe, bool enable)
{
    if (pipe >= NRF_MAX_PIPES) return;
    
    uint8_t en_aa = readReg(NRF_EN_AA_REGISTER);
    if (enable) {
        en_aa |= (1 << pipe);
    } else {
        en_aa &= ~(1 << pipe);
    }
    writeReg(NRF_EN_AA_REGISTER, en_aa);
    pipes[pipe].auto_ack_enabled = enable;
}

bool NRF24::isAutoAckEnabled()
{
    return auto_ack_enabled;
}

bool NRF24::isAutoAckEnabled(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return false;
    return pipes[pipe].auto_ack_enabled;
}

// Auto-retransmit configuration
void NRF24::setRetries(uint8_t delay, uint8_t count)
{
    if (delay > 15) delay = 15;
    if (count > 15) count = 15;
    
    uint8_t setup_retr = (delay << 4) | count;
    writeReg(NRF_SETUP_RETR_REGISTER, setup_retr);
    
    this->auto_retransmit_delay = (NRF24_AutoRetransmitDelay)delay;
    this->auto_retransmit_count = count;
}

void NRF24::setRetryDelay(NRF24_AutoRetransmitDelay delay)
{
    uint8_t setup_retr = readReg(NRF_SETUP_RETR_REGISTER);
    setup_retr = (setup_retr & 0x0F) | (delay << 4);
    writeReg(NRF_SETUP_RETR_REGISTER, setup_retr);
    this->auto_retransmit_delay = delay;
}

void NRF24::setRetryCount(uint8_t count)
{
    if (count > 15) count = 15;
    uint8_t setup_retr = readReg(NRF_SETUP_RETR_REGISTER);
    setup_retr = (setup_retr & 0xF0) | count;
    writeReg(NRF_SETUP_RETR_REGISTER, setup_retr);
    this->auto_retransmit_count = count;
}

uint8_t NRF24::getRetryCount()
{
    return readReg(NRF_SETUP_RETR_REGISTER) & 0x0F;
}

NRF24_AutoRetransmitDelay NRF24::getRetryDelay()
{
    return (NRF24_AutoRetransmitDelay)((readReg(NRF_SETUP_RETR_REGISTER) & 0xF0) >> 4);
}

// Dynamic payload
void NRF24::enableDynamicPayloads()
{
    writeReg(NRF_FEATURE_REGISTER, readReg(NRF_FEATURE_REGISTER) | NRF_FEATURE_EN_DPL);
    writeReg(NRF_DYNPD_REGISTER, 0x3F); // Enable for all pipes
    this->dynamic_payload_enabled = true;
}

void NRF24::disableDynamicPayloads()
{
    writeReg(NRF_FEATURE_REGISTER, readReg(NRF_FEATURE_REGISTER) & ~NRF_FEATURE_EN_DPL);
    writeReg(NRF_DYNPD_REGISTER, 0x00); // Disable for all pipes
    this->dynamic_payload_enabled = false;
}

void NRF24::enableDynamicPayload(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return;
    
    writeReg(NRF_FEATURE_REGISTER, readReg(NRF_FEATURE_REGISTER) | NRF_FEATURE_EN_DPL);
    writeReg(NRF_DYNPD_REGISTER, readReg(NRF_DYNPD_REGISTER) | (1 << pipe));
    pipes[pipe].dynamic_payload_enabled = true;
}

void NRF24::disableDynamicPayload(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return;
    
    writeReg(NRF_DYNPD_REGISTER, readReg(NRF_DYNPD_REGISTER) & ~(1 << pipe));
    pipes[pipe].dynamic_payload_enabled = false;
}

bool NRF24::isDynamicPayloadEnabled()
{
    return dynamic_payload_enabled;
}

bool NRF24::isDynamicPayloadEnabled(uint8_t pipe)
{
    if (pipe >= NRF_MAX_PIPES) return false;
    return pipes[pipe].dynamic_payload_enabled;
}

// Payload with ACK
void NRF24::enableAckPayload()
{
    writeReg(NRF_FEATURE_REGISTER, readReg(NRF_FEATURE_REGISTER) | NRF_FEATURE_EN_ACK_PAY | NRF_FEATURE_EN_DPL);
    writeReg(NRF_DYNPD_REGISTER, readReg(NRF_DYNPD_REGISTER) | 0x01); // Enable for pipe 0
}

void NRF24::disableAckPayload()
{
    writeReg(NRF_FEATURE_REGISTER, readReg(NRF_FEATURE_REGISTER) & ~NRF_FEATURE_EN_ACK_PAY);
}

void NRF24::writeAckPayload(uint8_t pipe, uint8_t *data, uint8_t len)
{
    if (pipe >= NRF_MAX_PIPES || len > NRF_MAX_PAYLOAD_SIZE) return;
    
    uint8_t cmd = NRF_W_ACK_PAYLOAD | pipe;
    csnLow();
    spi_write_blocking(spi, &cmd, 1);
    spi_write_blocking(spi, data, len);
    csnHigh();
}

// Data transmission
bool NRF24::write(uint8_t *data, uint8_t len)
{
    return write(data, len, false);
}

bool NRF24::write(uint8_t *data, uint8_t len, bool multicast)
{
    if (len > NRF_MAX_PAYLOAD_SIZE) return false;
    
    // Switch to TX mode
    setModeTX();
    
    // Clear previous interrupts
    clearInterrupts();
    
    // Write payload
    uint8_t cmd = multicast ? NRF_W_TX_PAYLOAD_NO_ACK : NRF_W_TX_PAYLOAD;
    csnLow();
    spi_write_blocking(spi, &cmd, 1);
    spi_write_blocking(spi, data, len);
    csnHigh();
    
    // Start transmission
    ceHigh();
    sleep_us(15); // Minimum pulse width
    ceLow();
    
    // Wait for transmission to complete
    uint32_t timeout = 0;
    while (!(readReg(NRF_STATUS_REGISTER) & (NRF_STATUS_TX_DS | NRF_STATUS_MAX_RT))) {
        sleep_us(1);
        if (++timeout > 10000) { // 10ms timeout
            flushTx();
            return false;
        }
    }
    
    uint8_t status = readReg(NRF_STATUS_REGISTER);
    bool result = (status & NRF_STATUS_TX_DS) != 0;
    
    // Clear interrupts
    clearInterrupts();
    
    // Update statistics
    if (result) {
        packets_sent++;
    } else {
        packets_lost++;
    }
    
    return result;
}

void NRF24::startWrite(uint8_t *data, uint8_t len)
{
    startWrite(data, len, false);
}

void NRF24::startWrite(uint8_t *data, uint8_t len, bool multicast)
{
    if (len > NRF_MAX_PAYLOAD_SIZE) return;
    
    // Switch to TX mode
    setModeTX();
    
    // Clear previous interrupts
    clearInterrupts();
    
    // Write payload
    uint8_t cmd = multicast ? NRF_W_TX_PAYLOAD_NO_ACK : NRF_W_TX_PAYLOAD;
    csnLow();
    spi_write_blocking(spi, &cmd, 1);
    spi_write_blocking(spi, data, len);
    csnHigh();
    
    // Start transmission
    ceHigh();
    sleep_us(15); // Minimum pulse width
    ceLow();
}

bool NRF24::writeBlocking(uint8_t *data, uint8_t len, uint32_t timeout_ms)
{
    startWrite(data, len);
    
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    uint32_t current_time = start_time;
    
    while (current_time - start_time < timeout_ms) {
        uint8_t status = readReg(NRF_STATUS_REGISTER);
        if (status & (NRF_STATUS_TX_DS | NRF_STATUS_MAX_RT)) {
            bool result = (status & NRF_STATUS_TX_DS) != 0;
            clearInterrupts();
            
            if (result) {
                packets_sent++;
            } else {
                packets_lost++;
            }
            
            return result;
        }
        sleep_us(10);
        current_time = to_ms_since_boot(get_absolute_time());
    }
    
    // Timeout occurred
    flushTx();
    return false;
}

// Data reception
bool NRF24::available()
{
    uint8_t pipe_num;
    return available(&pipe_num);
}

bool NRF24::available(uint8_t *pipe_num)
{
    uint8_t status = readReg(NRF_STATUS_REGISTER);
    
    if (status & NRF_STATUS_RX_DR) {
        *pipe_num = (status & NRF_STATUS_RX_P_NO) >> 1;
        return true;
    }
    
    return false;
}

uint8_t NRF24::read(uint8_t *data, uint8_t len)
{
    uint8_t payload_len = len;
    
    if (dynamic_payload_enabled) {
        payload_len = getDynamicPayloadSize();
        if (payload_len > len) payload_len = len;
    }
    
    csnLow();
    uint8_t cmd = NRF_R_RX_PAYLOAD;
    spi_write_blocking(spi, &cmd, 1);
    spi_read_blocking(spi, 0x00, data, payload_len);
    csnHigh();
    
    // Clear RX interrupt
    writeReg(NRF_STATUS_REGISTER, NRF_STATUS_RX_DR);
    
    packets_received++;
    return payload_len;
}

uint8_t NRF24::getDynamicPayloadSize()
{
    uint8_t result = 0;
    csnLow();
    uint8_t cmd = NRF_R_RX_PL_WID;
    spi_write_blocking(spi, &cmd, 1);
    spi_read_blocking(spi, 0x00, &result, 1);
    csnHigh();
    
    if (result > NRF_MAX_PAYLOAD_SIZE) {
        flushRx();
        return 0;
    }
    
    return result;
}

void NRF24::startListening()
{
    setModeRX();
}

void NRF24::stopListening()
{
    ceLow();
    if (readReg(NRF_FEATURE_REGISTER) & NRF_FEATURE_EN_ACK_PAY) {
        sleep_us(130);
    }
}

// Status and diagnostics
uint8_t NRF24::getStatus()
{
    return readReg(NRF_STATUS_REGISTER);
}

bool NRF24::testCarrier()
{
    return readReg(NRF_RPD_REGISTER) & 0x01;
}

bool NRF24::testRPD()
{
    return testCarrier();
}

uint8_t NRF24::getObserveTx()
{
    return readReg(NRF_OBSERVE_TX_REGISTER);
}

uint8_t NRF24::getLostPackets()
{
    return (readReg(NRF_OBSERVE_TX_REGISTER) & 0xF0) >> 4;
}

uint8_t NRF24::getRetransmitCount()
{
    return readReg(NRF_OBSERVE_TX_REGISTER) & 0x0F;
}

void NRF24::resetPacketLossCounters()
{
    writeReg(NRF_RF_CH_REGISTER, channel); // Reset by writing to channel register
}

// Interrupt handling
void NRF24::maskInterrupt(uint8_t interrupt)
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    config |= interrupt;
    writeReg(NRF_CONFIG_REGISTER, config);
}

void NRF24::unmaskInterrupt(uint8_t interrupt)
{
    uint8_t config = readReg(NRF_CONFIG_REGISTER);
    config &= ~interrupt;
    writeReg(NRF_CONFIG_REGISTER, config);
}

bool NRF24::isInterruptTriggered(uint8_t interrupt)
{
    return (readReg(NRF_STATUS_REGISTER) & interrupt) != 0;
}

void NRF24::clearInterrupt(uint8_t interrupt)
{
    writeReg(NRF_STATUS_REGISTER, interrupt);
}

// FIFO operations
bool NRF24::isTxFifoEmpty()
{
    return readReg(NRF_FIFO_STATUS_REGISTER) & NRF_FIFO_STATUS_TX_EMPTY;
}

bool NRF24::isTxFifoFull()
{
    return readReg(NRF_FIFO_STATUS_REGISTER) & NRF_FIFO_STATUS_TX_FULL;
}

bool NRF24::isRxFifoEmpty()
{
    return readReg(NRF_FIFO_STATUS_REGISTER) & NRF_FIFO_STATUS_RX_EMPTY;
}

bool NRF24::isRxFifoFull()
{
    return readReg(NRF_FIFO_STATUS_REGISTER) & NRF_FIFO_STATUS_RX_FULL;
}

void NRF24::flushTxFifo()
{
    flushTx();
}

void NRF24::flushRxFifo()
{
    flushRx();
}

// Advanced features
void NRF24::setCarrierWave(bool enable)
{
    uint8_t rf_setup = readReg(NRF_RF_SETUP_REGISTER);
    if (enable) {
        rf_setup |= NRF_RF_SETUP_CONT_WAVE;
    } else {
        rf_setup &= ~NRF_RF_SETUP_CONT_WAVE;
    }
    writeReg(NRF_RF_SETUP_REGISTER, rf_setup);
}

bool NRF24::isCarrierWave()
{
    return (readReg(NRF_RF_SETUP_REGISTER) & NRF_RF_SETUP_CONT_WAVE) != 0;
}

void NRF24::enterTestMode()
{
    setCarrierWave(true);
    setModeTX();
    ceHigh();
}

void NRF24::exitTestMode()
{
    ceLow();
    setCarrierWave(false);
}

// Statistics
uint16_t NRF24::getPacketsSent()
{
    return packets_sent;
}

uint16_t NRF24::getPacketsReceived()
{
    return packets_received;
}

uint16_t NRF24::getPacketsLost()
{
    return packets_lost;
}

void NRF24::resetStatistics()
{
    packets_sent = 0;
    packets_received = 0;
    packets_lost = 0;
    retransmit_count = 0;
}

// Print detailed information about the module
void NRF24::printDetails()
{
    printf("NRF24L01%s Details:\n", is_plus_variant ? "+" : "");
    printf("STATUS: 0x%02X\n", readReg(NRF_STATUS_REGISTER));
    printf("CONFIG: 0x%02X\n", readReg(NRF_CONFIG_REGISTER));
    printf("EN_AA: 0x%02X\n", readReg(NRF_EN_AA_REGISTER));
    printf("EN_RXADDR: 0x%02X\n", readReg(NRF_EN_RXADDR_REGISTER));
    printf("SETUP_AW: 0x%02X\n", readReg(NRF_SETUP_AW_REGISTER));
    printf("SETUP_RETR: 0x%02X\n", readReg(NRF_SETUP_RETR_REGISTER));
    printf("RF_CH: %d\n", readReg(NRF_RF_CH_REGISTER));
    printf("RF_SETUP: 0x%02X\n", readReg(NRF_RF_SETUP_REGISTER));
    printf("OBSERVE_TX: 0x%02X\n", readReg(NRF_OBSERVE_TX_REGISTER));
    printf("FIFO_STATUS: 0x%02X\n", readReg(NRF_FIFO_STATUS_REGISTER));
    printf("DYNPD: 0x%02X\n", readReg(NRF_DYNPD_REGISTER));
    printf("FEATURE: 0x%02X\n", readReg(NRF_FEATURE_REGISTER));
    
    printf("\nData Rate: ");
    switch (getDataRate()) {
        case NRF24_DATA_RATE_250KBPS: printf("250kbps"); break;
        case NRF24_DATA_RATE_1MBPS: printf("1Mbps"); break;
        case NRF24_DATA_RATE_2MBPS: printf("2Mbps"); break;
    }
    printf("\n");
    
    printf("Power Level: ");
    switch (getPowerLevel()) {
        case NRF24_POWER_LEVEL_NEG18DBM: printf("-18dBm"); break;
        case NRF24_POWER_LEVEL_NEG12DBM: printf("-12dBm"); break;
        case NRF24_POWER_LEVEL_NEG6DBM: printf("-6dBm"); break;
        case NRF24_POWER_LEVEL_0DBM: printf("0dBm"); break;
    }
    printf("\n");
    
    printf("CRC Length: ");
    switch (getCRCLength()) {
        case NRF24_CRC_DISABLED: printf("Disabled"); break;
        case NRF24_CRC_8BIT: printf("8-bit"); break;
        case NRF24_CRC_16BIT: printf("16-bit"); break;
    }
    printf("\n");
    
    printf("Address Width: %d bytes\n", address_width);
    printf("Channel: %d (%.3f GHz)\n", channel, 2.4 + (channel * 0.001));
    printf("Payload Size: %d bytes\n", payload_size);
    printf("Auto Retransmit: %d retries, %d µs delay\n", 
           getRetryCount(), (getRetryDelay() + 1) * 250);
    
    // Print addresses
    printf("\nAddresses:\n");
    printf("TX: ");
    for (int i = 0; i < address_width; i++) {
        printf("%02X ", tx_address[i]);
    }
    printf("\n");
    
    for (int i = 0; i < NRF_MAX_PIPES; i++) {
        if (rx_pipe_enabled & (1 << i)) {
            printf("RX P%d: ", i);
            for (int j = 0; j < address_width; j++) {
                printf("%02X ", pipes[i].address[j]);
            }
            printf("(Size: %d)\n", pipes[i].payload_size);
        }
    }
    
    printf("\nStatistics:\n");
    printf("Packets Sent: %d\n", packets_sent);
    printf("Packets Received: %d\n", packets_received);
    printf("Packets Lost: %d\n", packets_lost);
}

// Compatibility functions (for backward compatibility)
void NRF24::enableAck(uint8_t ack)
{
    setAutoAck(ack != 0);
}

void NRF24::config(uint8_t *address, uint8_t channel, uint8_t messageLen)
{
    this->channel = channel;
    this->messageLen = messageLen;
    
    csnHigh();
    ceLow();
    sleep_ms(11);
    
    // Basic configuration
    writeReg(NRF_CONFIG_REGISTER, NRF_CONFIG_EN_CRC | NRF_CONFIG_CRCO | NRF_CONFIG_PWR_UP);
    sleep_us(1500);
    
    // Disable auto-ack by default for compatibility
    writeReg(NRF_EN_AA_REGISTER, 0x00);
    
    // Set address width to 5 bytes
    writeReg(NRF_SETUP_AW_REGISTER, 0x03);
    
    // Set auto retransmit
    writeReg(NRF_SETUP_RETR_REGISTER, 0x00);
    
    // Set channel
    writeReg(NRF_RF_CH_REGISTER, channel);
    
    // Set RF setup (2Mbps, 0dBm)
    writeReg(NRF_RF_SETUP_REGISTER, 0x0E);
    
    // Set addresses
    writeReg(NRF_RX_ADDR_P0_REGISTER, address, 5);
    writeReg(NRF_TX_ADDR_REGISTER, address, 5);
    
    // Set payload size
    writeReg(NRF_RX_PW_P0_REGISTER, messageLen);
    
    // Update internal state
    setChannel(channel);
    setPayloadSize(messageLen);
    setAddressWidth(NRF24_ADDR_WIDTH_5BYTES);
    setTxAddress(address);
    openReadingPipe(0, address);
}

void NRF24::modeRX()
{
    setModeRX();
}

void NRF24::modeTX()
{
    setModeTX();
}

uint8_t NRF24::newMessage()
{
    return available();
}

void NRF24::sendMessage(uint8_t *data)
{
    write(data, messageLen);
}

void NRF24::getMessage(uint8_t *buffer)
{
    read(buffer, messageLen);
}

uint8_t NRF24::readRegister(uint8_t reg)
{
    return readReg(reg);
}
