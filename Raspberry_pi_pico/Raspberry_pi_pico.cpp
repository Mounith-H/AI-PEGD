// Hardware Connections:
//
// INMP441 Microphone:
// - VDD  -> 3.3V (Pin 36 on Pico)
// - GND  -> GND  (Pin 38 on Pico)
// - SD   -> GP20 (Pin 26 on Pico) [DATA]
// - SCK  -> GP18 (Pin 24 on Pico) [BCLK]
// - WS   -> GP19 (Pin 25 on Pico) [LRCLK]
// - L/R  -> GND  (for left channel)
//
// LED Connection:
// - Anode   -> GP13 (Pin 17 on Pico) through a 220Ω resistor
// - Cathode -> GND  (any GND pin on Pico)
//
// FTDI Module Connection:
// - FTDI RX -> GP0 (Pin 1 on Pico, UART0 TX)
// - FTDI TX -> GP1 (Pin 2 on Pico, UART0 RX)
// - FTDI VCC -> 3.3V (Pin 36 on Pico)
// - FTDI GND -> GND (Pin 38 on Pico)
//
// NEO-6M GPS Module Connection:
// - VCC -> 3.3V (Pin 36 on Pico)
// - GND -> GND (Pin 38 on Pico)
// - TX  -> GP9 (Pin 11 on Pico, UART1 RX)
// - RX  -> GP8 (Pin 12 on Pico, UART1 TX)
//
// NRF24L01 Module Connection:
// - VCC -> 3.3V (Pin 36 on Pico)
// - GND -> GND (Pin 38 on Pico)
// - SCK -> GP2 (Pin 4 on Pico)
// - MOSI -> GP3 (Pin 5 on Pico)
// - MISO -> GP4 (Pin 6 on Pico)
// - CSN -> GP5 (Pin 7 on Pico)
// - CE -> GP6 (Pin 9 on Pico)
// - IRQ -> GP7 (Pin 10 on Pico) (optional, can be used for interrupts)
//
// Power Control MH-CD42 Module:
// - Key -> GP17 (Pin 22 on Pico) (PWM) [Very important, used to power to Pico]

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <cstdio>
#include <string.h>  // For strncmp, memcpy, memset
#include <math.h>    // For fabs
#include <time.h>    // For time functions
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/mutex.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/rtc.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "hardware/uart.h"
#include "hardware/spi.h"
#include "pico_audio_recorder.h"
#include "power_control.pio.h"  // This will be generated from power_control.pio
#include "NRF24.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

extern bool new_read_data;
extern int16_t *fw_ptr;

#define ENABLE_DEBUG 1
#define ENABLE_BOOTLOGO 1
#define ENABLE_GPS 1
#define ENABLE_NRF24 1

// Define the confidence threshold for gunshot detection --------------------------------------------
#define Confidence_THRESHOLD 0.8f
// --------------------------------------------------------------------------------------------------

#define UART_TX_PIN 0
#define UART_RX_PIN 1
#define UART_ID uart0
#define BAUD_RATE 115200
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY  UART_PARITY_NONE

#define LED_INBUILT_IN 25       // Built-in LED
#define LED_PIN 13              // External LED for indication

#define INMP441_SCK_PIN 18      // BCLK
#define INMP441_WS_PIN  19      // LRCLK/WS
#define INMP441_SD_PIN  20      // DATA

#define NRF_SPI_PORT spi0
#define NRF_SCK_PIN 2  // SPI Clock
#define NRF_MOSI_PIN 3 // SPI MOSI
#define NRF_MISO_PIN 4 // SPI MISO
#define NRF_CSN_PIN 5  // SPI Chip Select
#define NRF_CE_PIN 6   // Chip Enable
#define NRF_IRQ_PIN 7  // IRQ pin (optional)

// NRF24L01 Register Definitions
#define NRF_CONFIG      0x00
#define NRF_STATUS      0x07
#define NRF_RF_SETUP    0x06
#define NRF_RF_CH       0x05
#define NRF_TX_ADDR     0x10
#define NRF_RX_ADDR_P0  0x0A
#define NRF_FIFO_STATUS 0x17
#define NRF_R_REGISTER    0x00
#define NRF_W_REGISTER    0x20
#define NRF_NOP           0xFF
#define RF_DR_2MBPS     0x08  // 2 Mbps mode (default)
#define RF_DR_1MBPS     0x00  // 1 Mbps mode
#define RF_DR_250KBPS   0x20  // 250 kbps mode
#define RF_PWR_0DBM    0x06  // Maximum power: 0dBm
#define RF_PWR_NEG_6DBM  0x04  // -6dBm
#define RF_PWR_NEG_12DBM 0x02  // -12dBm
#define RF_PWR_NEG_18DBM 0x00  // Minimum power: -18dBm

// NRF24L01 Default Settings
#define NRF_CHANNEL 2  // Default channel for NRF24L01 (2.4GHz band)
#define NODE_ID "01"    // Node identifier
#define MAX_PAYLOAD_SIZE 24  // NRF24L01 max payload size (increased from 10 to 24)

#define GPS_UART_ID uart1
#define GPS_UART_TX_PIN 8
#define GPS_UART_RX_PIN 9
#define GPS_BAUD_RATE 9600
#define GPS_DATA_BITS 8
#define GPS_STOP_BITS 1
#define GPS_PARITY    UART_PARITY_NONE
#define GPS_BUFFER_SIZE 256

#define POWER_CONTROL_PIN 17   // Key pin for MH-CD42 module
#define POWER_CONTROL_PIO pio1 // Using pio1 to avoid conflicts
#define POWER_CONTROL_SM 1     // State machine 1
#define POWER_FREQ 2           //  frequency
#define POWER_DUTY_CYCLE 30    // 50% duty cycle

// --------------------------- LOGO ---------------------------

#if ENABLE_BOOTLOGO
const char *logo = R"LOGO(
'##::::'##:'########:'##:::::::
 ###::'###: ##.....:: ##:::::::
 ####'####: ##::::::: ##:::::::
 ## ### ##: ######::: ##:::::::
 ##. #: ##: ##...:::: ##:::::::
 ##:.:: ##: ##::::::: ##:::::::
 ##:::: ##: ########: ########:
..:::::..::........::........::
)LOGO";
#endif
// --------------------------- LOGO ---------------------------


// GPS location structure
struct gps_location_t {
    bool valid;
    float latitude;
    float longitude;
} g_current_location = {false, 0.0f, 0.0f};

// Message structure for transmitting data
typedef struct {
    char node_id[3];        // 2 chars + null terminator
    uint16_t msg_id;        // Message ID (0-65535)
    char date[11];          // DD/MM/YYYY + null terminator
    char time[9];           // HH:MM:SS + null terminator
    char latitude[12];      // Including decimal point and sign
    char longitude[12];     // Including decimal point and sign
    float confidence;
} gunshot_message_t;

#define NMEA_GPRMC "$GPRMC"
#define NMEA_GPGGA "$GPGGA"

// GPS globals
static char gps_buffer[GPS_BUFFER_SIZE];
static uint line_pos = 0;  // Current position in line buffer
static mutex_t gps_mutex;
static bool start_of_sentence = false;
static uint32_t last_valid_sentence = 0;
static uint32_t sentence_count = 0;
static bool valid_gps_data = false;
static float latitude = 0.0f;
static float longitude = 0.0f;

// GPS date/time globals
static char gps_date[11] = "01/01/2025";  // DD/MM/YYYY format, default fallback
static char gps_time[9] = "00:00:00";    // HH:MM:SS format, default fallback
static char local_date[11] = "01/01/2025";  // Local date in DD/MM/YYYY format
static char local_time[9] = "00:00:00";    // Local time in HH:MM:SS format
static bool gps_datetime_valid = false;


// Inter-core communication variables
static volatile bool core1_ready = false;
static volatile float latest_confidence = 0.0f;
static volatile uint32_t gunshot_detections = 0;
static volatile bool gunshot_detected = false;

// Global variables and mutex
static mutex_t printf_mutex;
static mutex_t audio_mutex;  // Mutex for audio data sharing between cores

// Global message ID counter
static uint8_t g_message_id = 0;

// Global gunshot message ID counter
static uint16_t g_gunshot_msg_id = 1;  // Start from 1 for readability

bool led_on = false;


// Message fragment structure for reliable transmission
typedef struct {
    uint8_t msg_id;        // Message ID (0-255)
    uint8_t fragment_num;  // Current fragment number (0-based)
    uint8_t total_fragments; // Total number of fragments
    uint8_t data_length;   // Length of data in this fragment
    uint8_t data[20];      // Data payload (24 - 4 header bytes = 20 data bytes)
} fragment_packet_t;

uint8_t address[5] = {'g','y','r','o','c'};  // 5-byte address gyroc
NRF24 nrf(NRF_SPI_PORT, NRF_SCK_PIN, NRF_MOSI_PIN, NRF_MISO_PIN, NRF_CSN_PIN, NRF_CE_PIN, NRF_IRQ_PIN);


// Function declarations
void init_sync(void);
void init_uart(void);
void safe_uart_puts(const char* str);
void safe_printf(const char* format, ...);
bool init_ml_model();
void core1_entry(void);  // Core 1 entry point for ML processing
void display_boot_logo(void);
bool init_power_control(void);
void update_power_control(float freq, float duty_cycle);
void init_gps(void);
bool wait_for_gps_fix(uint32_t timeout_ms);
void process_gps(void);
bool parse_gprmc(const char* sentence, float* out_lat, float* out_lon, bool* out_valid);
float nmea_to_decimal_degrees(float nmea_coord, char direction);
void gps_uart_irq_handler();
void get_gps_datetime(char* date_out, char* time_out, bool* valid_out);
bool test_microphone();
void nrf24l01_send_message(const char* message);
void create_json_message(char* buffer, size_t buffer_size, const gunshot_message_t* msg);
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr);
void init_nrf24(void);
void convert_utc_to_local_date_time(int utc_hour, int utc_min, int utc_sec, int utc_day, int utc_month, int utc_year, int offset_hour, int offset_min, char* out_time_str, char* out_date_str);

// Initialize power control
bool init_power_control(void) {
    safe_uart_puts("Initializing power control...\r\n");
    
    // Load power control program into PIO memory
    PIO pio = POWER_CONTROL_PIO;
    uint sm = POWER_CONTROL_SM;
    
    if (!pio_can_add_program(pio, &power_control_program)) {
        safe_uart_puts("Failed to load power control program - no memory\n");
        return false;
    }
    
    uint offset = pio_add_program(pio, &power_control_program);
    
    // Calculate clock divider for desired frequency (1 kHz)
    // The period will be 2^16, so to get 1kHz:
    // CPU_freq / (desired_freq * 2^16) = clk_div
    uint clk_div = clock_get_hz(clk_sys) / (POWER_FREQ * (1u << 16));
    
    // Initialize the PWM program
    power_control_program_init(pio, sm, offset, POWER_CONTROL_PIN, clk_div);
    
    // Set the period (using full 16-bit resolution)
    uint32_t period = (1u << 16) - 1;
    uint32_t level = period / 2;  // 50% duty cycle
    
    power_control_set_period(pio, sm, period);
    power_control_set_level(pio, sm, level);
    
    safe_uart_puts("Power control initialized successfully\r\n");
    return true;
}

// Update power control settings (0-100 for duty cycle)
void update_power_control(float freq, float duty_cycle) {
    PIO pio = POWER_CONTROL_PIO;
    uint sm = POWER_CONTROL_SM;
    
    // Clamp duty cycle to valid range
    if (duty_cycle < 0.0f) duty_cycle = 0.0f;
    if (duty_cycle > 100.0f) duty_cycle = 100.0f;
    
    // Convert duty cycle percentage to level
    uint32_t period = (1u << 16) - 1;
    uint32_t level = (uint32_t)((duty_cycle / 100.0f) * period);
    
    // Update the PWM
    power_control_set_level(pio, sm, level);
}

void display_boot_logo(void) {
#if ENABLE_BOOTLOGO
    printf("%s\n", logo);
#endif
}

////------------- NRF Functions START ------------////

void init_nrf24(void) {
#if ENABLE_NRF24
    // Initialize NRF24L01 with the new comprehensive library
    // Initialize the radio
    if (!nrf.begin()) {
        printf("ERROR: Failed to initialize NRF24L01!\n");
        printf("Please check your wiring and power supply.\n");
        while (1) {
            // Fast blink to indicate error
            gpio_put(LED_INBUILT_IN, 1);
            sleep_ms(100);
            gpio_put(LED_INBUILT_IN, 0);
            sleep_ms(100);
        }
    }
    printf("NRF24L01 initialized successfully!\n");
    printf("Chip variant: NRF24L01%s\n", nrf.isConnected() ? "+" : "");    // Configure the radio to match STM32 receiver settings
    nrf.setPowerLevel(NRF24_POWER_LEVEL_NEG12DBM);    // Low power to match receiver
    nrf.setDataRate(NRF24_DATA_RATE_2MBPS);           // 2Mbps to match receiver
    nrf.setChannel(2);                                // Channel 2 to match STM32 receiver
    nrf.setCRCLength(NRF24_CRC_16BIT);                // 16-bit CRC to match receiver
    nrf.setAutoAck(true);                             // Enable auto-ack to match receiver
    nrf.setRetries(5, 15);                            // 5 retries with 4ms delay for reliability
    nrf.enableDynamicPayloads();                      // Enable dynamic payloads to match receiver
    uint8_t tx_address[] = {'g', 'y', 'r', 'o', 'c'};   // Set communication address (make sure this matches receiver address)
    nrf.openWritingPipe(tx_address);
    nrf.setModeTX();// Configure as transmitter
    printf("Radio Configuration:\n");
    printf("Channel: %d\n", nrf.getChannel());
    printf("Data Rate: 2Mbps\n");
    printf("Power Level: Low\n");
    printf("CRC Length: 16-bit\n");
    printf("Auto Ack: Enabled\n");
    printf("Dynamic Payloads: Enabled\n");
    printf("TX Address: %c%c%c%c%c\n", tx_address[0], tx_address[1], tx_address[2], tx_address[3], tx_address[4]);
    printf("\n=== DETAILED DEBUG INFO ===\n");
    nrf.printDetails();
    printf("============================\n\n");
    safe_uart_puts("NRF24L01 initialized successfully\r\n");
#endif
}

// Send message via NRF24L01 with fragmentation support
void nrf24l01_send_message(const char* message) {
    if (!message) return;
    
    size_t message_length = strlen(message);
    const uint8_t MAX_DATA_PER_FRAGMENT = 20; // 24 byte payload - 4 byte header
    
    // Calculate number of fragments needed
    size_t total_fragments = (message_length + MAX_DATA_PER_FRAGMENT - 1) / MAX_DATA_PER_FRAGMENT;
    
    // Limit to 255 fragments (uint8_t limit)
    if (total_fragments > 255) {
        // safe_uart_puts("ERROR: Message too long for fragmentation\r\n");
        return;
    }
    
    // Assign unique message ID
    uint8_t current_msg_id = g_message_id++;
    
    for (size_t fragment_num = 0; fragment_num < total_fragments; fragment_num++) {
        fragment_packet_t packet;
        
        // Set header
        packet.msg_id = current_msg_id;
        packet.fragment_num = fragment_num;
        packet.total_fragments = total_fragments;
        
        // Calculate data for this fragment
        size_t start_pos = fragment_num * MAX_DATA_PER_FRAGMENT;
        size_t remaining = message_length - start_pos;
        size_t data_length = (remaining > MAX_DATA_PER_FRAGMENT) ? MAX_DATA_PER_FRAGMENT : remaining;
        
        packet.data_length = data_length;
        
        // Copy data
        memset(packet.data, 0, sizeof(packet.data));
        memcpy(packet.data, message + start_pos, data_length);
        
        // Send fragment using the new NRF24 library
        bool success = nrf.write((uint8_t*)&packet, sizeof(fragment_packet_t));
        
        // Get detailed status information
        uint8_t status = nrf.getStatus();
        uint8_t observe_tx = nrf.getObserveTx();
        bool carrier = nrf.testCarrier();
        sleep_ms(50);
    }
}

// Create JSON message from data
void create_json_message(char* buffer, size_t buffer_size, const gunshot_message_t* msg) {
    snprintf(buffer, buffer_size, 
            "{\"node_id\":\"%s\",\"msg\":\"%u\",\"date\":\"%s\",\"time\":\"%s\",\"latitude\":\"%s\",\"longitude\":\"%s\",\"confidence\":%.2f}",
            msg->node_id, msg->msg_id, msg->date, msg->time, msg->latitude, msg->longitude, msg->confidence);
}

////------------ NRF Functions END ------------////

////------------ GPS Functions BEGIN ------------////

void init_gps(void) {
#if ENABLE_GPS
    safe_uart_puts("\r\n=== GPS Initialization Starting ===\r\n");
    safe_uart_puts("\r\n=== Program Starting (GPS Debug Mode) ===\r\n");
    safe_uart_puts("Initializing GPS...\r\n");
    
    // Initialize UART for GPS
    uart_init(GPS_UART_ID, GPS_BAUD_RATE);
    uart_set_hw_flow(GPS_UART_ID, false, false);
    uart_set_format(GPS_UART_ID, GPS_DATA_BITS, GPS_STOP_BITS, GPS_PARITY);
    
    // Set UART pins
    gpio_set_function(GPS_UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(GPS_UART_RX_PIN, GPIO_FUNC_UART);
    
    // Force immediate RX, no buffering
    uart_set_fifo_enabled(GPS_UART_ID, false);
    
    // Reset UART and clear any pending data
    uart_is_readable(GPS_UART_ID);  // Clear RX flag
    while (uart_is_readable(GPS_UART_ID)) {
        uart_getc(GPS_UART_ID);  // Clear any pending data
    }
    
    // Initialize mutex
    mutex_init(&gps_mutex);
    
    // Set up and enable interrupt handler
    int GPS_UART_ID_IRQ = UART1_IRQ;  // UART1 for GPS
    irq_set_exclusive_handler(GPS_UART_ID_IRQ, gps_uart_irq_handler);
    irq_set_enabled(GPS_UART_ID_IRQ, true);
    
    // Enable UART to send interrupts - RX only
    uart_set_irq_enables(GPS_UART_ID, true, false);
    
    safe_uart_puts("GPS UART initialized with interrupt handling\r\n");
    safe_uart_puts("All received characters will be echoed:\r\n\r\n");
    safe_uart_puts("Starting GPS monitoring...\r\n");
    safe_uart_puts("Waiting for initial GPS fix (timeout: 60 seconds)...\r\n");
    if (wait_for_gps_fix(60000)) safe_uart_puts("GPS fix acquired successfully!\r\n");
    else safe_uart_puts("GPS fix timeout - continuing without initial fix.\r\n");
    sleep_ms(100); // give time for things to settle.
#endif
}

bool wait_for_gps_fix(uint32_t timeout_ms) {
    uint32_t start = to_ms_since_boot(get_absolute_time());
    const uint32_t PRINT_INTERVAL = 1000;   // Status update every second
    const uint32_t STATUS_INTERVAL = 100;   // Detailed status every 100ms

    char buf[256];  // Buffer for formatting messages

    // Print initial status
    snprintf(buf, sizeof(buf), "\r\n=== Starting GPS Fix Wait ===\r\n");
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "Timeout set to: %lu ms\r\n", timeout_ms);
    safe_uart_puts(buf);

    int attempts = 0;
    while (!g_current_location.valid) {
        attempts++;
        uint32_t now = to_ms_since_boot(get_absolute_time());
        
        // Check timeout
        if (now - start >= timeout_ms) {
            snprintf(buf, sizeof(buf), "\r\n*** GPS Fix Timeout ***\r\n");
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "- Time elapsed: %d seconds\r\n", timeout_ms / 1000);
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "- Total attempts: %d\r\n", attempts);
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "- Last known state:\r\n");
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "  Valid: %s\r\n", g_current_location.valid ? "YES" : "NO");
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "  Latitude: %.6f\r\n", g_current_location.latitude);
            safe_uart_puts(buf);
            snprintf(buf, sizeof(buf), "  Longitude: %.6f\r\n", g_current_location.longitude);
            safe_uart_puts(buf);
            return false;
        }
        
        sleep_ms(50);  // Check more frequently but don't hog CPU
    }
    
    // GPS fix obtained!
    uint32_t end = to_ms_since_boot(get_absolute_time());
    snprintf(buf, sizeof(buf), "\r\n=== GPS Fix Obtained! ===\r\n");
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Time taken: %lu ms\r\n", end - start);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Total attempts: %d\r\n", attempts);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Position:\r\n");
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "  Latitude: %.6f\r\n", g_current_location.latitude);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "  Longitude: %.6f\r\n", g_current_location.longitude);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "=========================\r\n\r\n");
    safe_uart_puts(buf);
    return true;
}

// Initialize UART for GPS
void gps_uart_irq_handler() {
    while (uart_is_readable(GPS_UART_ID)) {
        uint8_t ch = uart_getc(GPS_UART_ID);
        
        // // Echo character to debug UART
        // uart_putc(UART_ID, ch);
        
        // Store in buffer for NMEA sentence processing
        if (line_pos < GPS_BUFFER_SIZE - 1) {
            gps_buffer[line_pos++] = ch;
            
            // Check for line ending
            if (ch == '\n' || ch == '\r') {
                if (line_pos > 1) {  // Ensure we have more than just the newline
                    gps_buffer[line_pos] = '\0';  // Null terminate
                    process_gps();  // Process the complete NMEA sentence
                }
                line_pos = 0;  // Reset for next sentence
            }
        } else {
            // Buffer overflow - reset
            line_pos = 0;
        }
    }
}

// Process GPS data - Now called from interrupt handler for complete sentences
void process_gps() {
    uint32_t now = to_ms_since_boot(get_absolute_time());
    
    // Check for GPRMC sentence
    if (strncmp(gps_buffer, "$GPRMC", 6) == 0) {
        sentence_count++;
        float lat, lon;
        bool valid;
        
        if (parse_gprmc(gps_buffer, &lat, &lon, &valid)) {
            mutex_enter_blocking(&gps_mutex);
            g_current_location.valid = valid;
            if (valid) {
                g_current_location.latitude = lat;
                g_current_location.longitude = lon;
                valid_gps_data = true;
                latitude = lat;
                longitude = lon;
                last_valid_sentence = now;
            }
            mutex_exit(&gps_mutex);
        }
    }
    
    // Print periodic status - MODIFY THIS SECTION 
    // can disable or adjust frequency as needed
    // this is only to show the GPS status this is not an error message
    static uint32_t last_status = 0;
}

// Parse GPRMC sentence Returns true if parsing successful and fix is valid
bool parse_gprmc(const char* sentence, float* out_lat, float* out_lon, bool* out_valid) {
    char field_buffer[32];  // Increased buffer for safety
    const char* ptr = sentence;
    int field = 0;
    int pos = 0;
    float nmea_lat = 0, nmea_lon = 0;
    char lat_dir = 'N', lon_dir = 'E';
    char time_str[16] = {0};
    char date_str[16] = {0};
    *out_valid = false;

    // Skip $GPRMC
    while (*ptr && *ptr != ',') ptr++;
    if (!*ptr) return false;
    ptr++;

    // Parse fields - need to go up to field 9 to get date
    while (*ptr && field < 10) {
        if (*ptr == ',' || *ptr == '*') {
            field_buffer[pos] = '\0';
            switch(field) {
                case 0:  // Time HHMMSS.SSS
                    if (strlen(field_buffer) >= 6) {
                        strcpy(time_str, field_buffer);
                    }
                    break;
                case 1:  // Status
                    *out_valid = (field_buffer[0] == 'A');
                    break;
                case 2:  // Latitude DDMM.MMMMM
                    nmea_lat = atof(field_buffer);
                    break;
                case 3:  // N/S
                    lat_dir = field_buffer[0];
                    break;
                case 4:  // Longitude DDDMM.MMMMM
                    nmea_lon = atof(field_buffer);
                    break;
                case 5:  // E/W
                    lon_dir = field_buffer[0];
                    break;
                case 8:  // Date DDMMYY
                    if (strlen(field_buffer) >= 6) {
                        strcpy(date_str, field_buffer);
                    }
                    break;
            }
            pos = 0;
            field++;
            ptr++;
            continue;
        }
        if (pos < sizeof(field_buffer) - 1) {
            field_buffer[pos++] = *ptr;
        }
        ptr++;
    }

    if (*out_valid) {
        *out_lat = nmea_to_decimal_degrees(nmea_lat, lat_dir);
        *out_lon = nmea_to_decimal_degrees(nmea_lon, lon_dir);
        
        // Update GPS date/time if we have valid data
        if (strlen(time_str) >= 6 && strlen(date_str) >= 6) {
            mutex_enter_blocking(&gps_mutex);
            
            // Parse time HHMMSS
            char hours[3] = {time_str[0], time_str[1], 0};
            char minutes[3] = {time_str[2], time_str[3], 0};
            char seconds[3] = {time_str[4], time_str[5], 0};

            int hour_int    = atoi(hours);
            int minute_int  = atoi(minutes);
            int second_int  = atoi(seconds);
            
            
            // Parse date DDMMYY
            char day[3] = {date_str[0], date_str[1], 0};
            char month[3] = {date_str[2], date_str[3], 0};
            char year[3] = {date_str[4], date_str[5], 0};
            int day_int   = atoi(day);
            int month_int = atoi(month);
            int year_int  = atoi(year) + 2000;


            convert_utc_to_local_date_time(hour_int, minute_int, second_int, day_int, month_int, year_int, 5, 30, local_time, local_date);


            gps_datetime_valid = true;
            mutex_exit(&gps_mutex);
        }
        
        return true;
    }
    
    safe_uart_puts("Invalid fix\r\n");
    return false;
}

// Convert NMEA coordinate format to decimal degrees
float nmea_to_decimal_degrees(float nmea_coord, char direction) {
    // Extract degrees (before decimal) and minutes (after decimal)
    int degrees = (int)(nmea_coord / 100.0f);  // First 2 or 3 digits
    float minutes = nmea_coord - (degrees * 100.0f);  // Remaining digits
    float decimal = degrees + (minutes / 60.0f);  // Convert minutes to decimal degrees    
    return (direction == 'S' || direction == 'W') ? -decimal : decimal;
}

void convert_utc_to_local_date_time(int utc_hour, int utc_min, int utc_sec, int utc_day, int utc_month, int utc_year, int offset_hour, int offset_min, char* out_time_str, char* out_date_str) {
    struct tm t = {
        .tm_sec = utc_sec,
        .tm_min = utc_min,
        .tm_hour = utc_hour,
        .tm_mday = utc_day,
        .tm_mon = utc_month - 1, // tm_mon is 0-based
        .tm_year = utc_year - 1900  // tm_year is years since 1900
    };

    // Convert to time_t
    time_t raw_time = mktime(&t);

    // Add time zone offset (in seconds)
    raw_time += offset_hour * 3600 + offset_min * 60;

    // Convert back to struct tm
    struct tm *local = gmtime(&raw_time);

    // Format time and date
    snprintf(out_time_str, 16, "%02d:%02d:%02d", local->tm_hour, local->tm_min, local->tm_sec);
    snprintf(out_date_str, 16, "%02d/%02d/%04d", local->tm_mday, local->tm_mon + 1, local->tm_year + 1900);
}


// Helper function to safely get GPS date/time
void get_gps_datetime(char* date_out, char* time_out, bool* valid_out) {
    mutex_enter_blocking(&gps_mutex);
    strcpy(date_out, local_date);
    strcpy(time_out, local_time);
    *valid_out = gps_datetime_valid;
    mutex_exit(&gps_mutex);
}
////------------ GPS Functions END ------------////

// Initialize mutex for printf synchronization
void init_sync(void) {
    mutex_init(&printf_mutex);
}

// Initialize UART for FTDI communication
void init_uart(void) {
    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
    uart_set_hw_flow(UART_ID, false, false);
    uart_set_format(UART_ID, DATA_BITS, STOP_BITS, PARITY);
    uart_set_fifo_enabled(UART_ID, false);
    uart_puts(UART_ID, "\r\n\nUART Successful\r\n");
}

// Safe UART output function
void safe_uart_puts(const char* str) {
    mutex_enter_blocking(&printf_mutex);
    uart_puts(UART_ID, str);
    mutex_exit(&printf_mutex);
}

// Safe printf function
void safe_printf(const char* format, ...) {
    mutex_enter_blocking(&printf_mutex);
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    mutex_exit(&printf_mutex);
}

int main() {
    stdio_init_all();
    display_boot_logo(); // Display boot logo if enabled
    init_sync();
    init_uart();
    sleep_ms(100);
    gpio_init(25); // led

    // Initialize power control first to ensure system stays powered
    if (!init_power_control()) {
        // If power control fails, try to indicate error with LED
        gpio_init(LED_INBUILT_IN);
        gpio_set_dir(LED_INBUILT_IN, GPIO_OUT);
        while(1) {
            gpio_put(LED_INBUILT_IN, !gpio_get(LED_INBUILT_IN));
            sleep_ms(100); // Fast blink to indicate error
        }
    }
    
    // Initialize LED for visual debugging
    gpio_init(LED_INBUILT_IN);
    gpio_set_dir(LED_INBUILT_IN, GPIO_OUT);
    gpio_put(LED_INBUILT_IN, true);
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, false); // Start with LED off


    init_gps();
    init_nrf24();
    
    safe_uart_puts("\r\n=== INMP441 + ML Gunshot Detection System (Dual Core) ===\r\n");
    safe_uart_puts("Core 0: Microphone initialization and LED control\r\n");
    safe_uart_puts("Core 1: ML processing and gunshot detection\r\n");
    
    // Initialize microphone on core 0
    safe_uart_puts("Initializing microphone on Core 0...\r\n");
    
    safe_uart_puts("Microphone initialized successfully on Core 0!\r\n");
    
    // Start Core 1 for ML processing
    safe_uart_puts("Starting Core 1 for ML processing...\r\n");
    multicore_launch_core1(core1_entry);
    
    // Wait for Core 1 to initialize
    safe_uart_puts("Waiting for Core 1 to initialize...\r\n");
    while (!core1_ready) {
        sleep_ms(100);
    }
    safe_uart_puts("Core 1 ready!\r\n");
    
    // Core 0 main loop - LED blinking based on gunshot detection
    safe_uart_puts("Core 0: Starting LED control loop...\r\n");
    uint32_t last_report = 0;
    uint32_t last_status = 0;
    uint32_t gunshot_count = 0;
    bool led_on_new = false;
    
    while (true) {  
        if (led_on) {
            last_report = to_ms_since_boot(get_absolute_time());
            led_on_new = true; // Turn on LED immediately
            gpio_put(LED_PIN, true);
            led_on = false; // Reset flag
        }
        if (led_on_new) {
            // LED on for 1 second, then off
            if (to_ms_since_boot(get_absolute_time()) - last_report >= 1000) {
                gpio_put(LED_PIN, false);
                led_on_new = false; // Turn off LED after 1 second
            }
        }

        // Check if gunshot was detected by Core 1
        if (gunshot_detected) {
            gunshot_count++;

            #if ENABLE_NRF24
                // Prepare and send message via NRF24L01
                gunshot_message_t msg;
                strncpy(msg.node_id, NODE_ID, sizeof(msg.node_id) - 1);
                msg.node_id[sizeof(msg.node_id) - 1] = '\0';  // Ensure null termination
                msg.msg_id = g_gunshot_msg_id++; // Assign unique message ID
                
                // Get GPS date/time if available, fallback to system time
                char gps_date_str[11];
                char gps_time_str[9];
                bool datetime_valid;
                get_gps_datetime(gps_date_str, gps_time_str, &datetime_valid);
                
                if (datetime_valid) {
                    // Use GPS date/time
                    strncpy(msg.date, gps_date_str, sizeof(msg.date) - 1);
                    strncpy(msg.time, gps_time_str, sizeof(msg.time) - 1);
                    msg.date[sizeof(msg.date) - 1] = '\0';
                    msg.time[sizeof(msg.time) - 1] = '\0';
                    printf("[EVENT] Using GPS date/time: %s %s\n", msg.date, msg.time);
                } else {
                    // Fallback to system time
                    time_t now = time(NULL);
                    struct tm *t = localtime(&now);
                    strftime(msg.date, sizeof(msg.date), "%d/%m/%Y", t);
                    strftime(msg.time, sizeof(msg.time), "%H:%M:%S", t);
                    printf("[EVENT] Using system date/time: %s %s\n", msg.date, msg.time);
                }
        
                snprintf(msg.latitude, sizeof(msg.latitude), "%.6f", latitude); // Set latitude, longitude, and confidence
                snprintf(msg.longitude, sizeof(msg.longitude), "%.6f", longitude);
                msg.confidence = latest_confidence;
                
                char json_buffer[1024];
                create_json_message(json_buffer, sizeof(json_buffer), &msg);// Create JSON message
                #if ENABLE_NRF24
                nrf24l01_send_message(json_buffer); // Send message with fragmentation
                led_on = true; // Set LED on flag
                #endif
            #endif
            gunshot_detected = false;// Reset the detection flag
        }
        
        // Print status every 30 seconds
        // uint32_t now = to_ms_since_boot(get_absolute_time());
        // if (now - last_status > 30000) {
        //     char buf[128];
        //     snprintf(buf, sizeof(buf), "Core 0: System status - Gunshots detected: %lu, GPS valid: %s\r\n", 
        //              gunshot_count, g_current_location.valid ? "YES" : "NO");
        //     safe_uart_puts(buf);
        //     last_status = now;
        // }
        
        sleep_ms(50); // Check for events every 50ms
    }

    return 0;
}

// Initialize ML model for gunshot detection
bool init_ml_model() {
    safe_uart_puts("Initializing ML model...\r\n");
    
    return true;
}

// Function to get raw feature data from the firmware pointer
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    // for (size_t i = 0; i < length; i++) {
    //     // >> 8 removes least-significant 8 bits from 24-bit INMP441 data
    //     int16_t sample = (int16_t)(fw_ptr[offset + i] >> 8);
    //     out_ptr[i] = (float)sample / 32768.0f; // Normalize to -1.0 to 1.0
    // }
    // return 0;

    numpy::int16_to_float(fw_ptr+offset, out_ptr, length);
    return 0;
}

// Core 1 entry point - handles all ML processing
void core1_entry() {
    safe_uart_puts("Core 1: Starting ML processing...\r\n");
    run_classifier_init();
    EI_IMPULSE_ERROR res;
    ei_impulse_result_t result = {nullptr};
    signal_t features_signal;
    features_signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    features_signal.get_data = &raw_feature_get_data;
    inmp441_pio_init(INMP441_pio, INMP441_SM, INMP441_SD, INMP441_SCK, INMP441_SAMPLE_RATE);
    safe_uart_puts("Core 1: Ready for ML processing!\r\n");
    // Core 1 main loop - ML processing
    uint8_t f_idx=0;
    core1_ready = true;  // Signal core0 even on failure so it doesn't hang
    while (true) {
        if (new_read_data) {
            new_read_data = false;

            if (++f_idx >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
                res = run_classifier_continuous(&features_signal, &result, false, true);
                if (res != EI_IMPULSE_OK) ei_printf("ERR: Failed to run classifier (%d)\n", res);
                ei_printf("%s:%.5f, ", ei_classifier_inferencing_categories[0], result.classification[0].value);
                ei_printf("%s:%.5f\r\n", ei_classifier_inferencing_categories[1], result.classification[1].value);
                if((ei_classifier_inferencing_categories[1] == "gunshot") && (result.classification[1].value > Confidence_THRESHOLD)) {
                    latest_confidence = result.classification[1].value;
                    gunshot_detected = true;
                    printf("Gunshot detected!\n");
                }
                f_idx=0;
            }
        }
    sleep_ms(10);
    }
}



