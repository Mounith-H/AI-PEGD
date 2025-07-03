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
// - Key -> GP17 (Pin 22 on Pico) (PWM) [Very importent, used to power to Pico]

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>  // For strncmp, memcpy, memset
#include <math.h>    // For fabs
#include <time.h>    // For time functions
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/mutex.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/clocks.h"
#include "hardware/uart.h"
#include "hardware/spi.h"
#include "inmp441.pio.h"
#include "lib/ml_model/model_handler.h" 
#include "lib/ml_model/gunshot_detection_model.h" 
#include "power_control.pio.h"  // This will be generated from power_control.pio
#include "NRF24.h"

///////////////////////////////////////////////////////////////////////////////////////
#define CONFIDENCE_THRESHOLD 0.5f
///////////////////////////////////////////////////////////////////////////////////////

#define NRF24_SCK 2
#define NRF24_MOSI 3
#define NRF24_MISO 4
#define NRF24_CSN 5
#define NRF24_CE 6
#define NRF24_IRQ 7

// Configuration defines for UART and GPIO
#define UART_TX_PIN 0
#define UART_RX_PIN 1
#define UART_ID uart0
#define BAUD_RATE 115200
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY    UART_PARITY_NONE

// GPS Configuration
#define GPS_UART_ID uart1
#define GPS_UART_TX_PIN 8
#define GPS_UART_RX_PIN 9
#define GPS_BAUD_RATE 9600
#define GPS_DATA_BITS 8
#define GPS_STOP_BITS 1
#define GPS_PARITY    UART_PARITY_NONE
#define GPS_BUFFER_SIZE 256

// GPIO Configuration
#define INMP441_SCK_PIN 18  // BCLK
#define INMP441_WS_PIN  19  // LRCLK/WS
#define INMP441_SD_PIN  20  // DATA

// LED Configuration
#define LED_INBUILT_IN 25   // Built-in LED
#define LED_PIN 13         // External LED for gunshot indication

// Audio Configuration
#define SAMPLE_RATE 16000
#define SAMPLE_BUFFER_SIZE 1024
#define DMA_CHANNEL 0
#define INMP441_PIO pio0
#define INMP441_SM  0
#define DEBUG_SAMPLES 32
#define MIC_TEST_DURATION_MS 5000

// MD-CD42 Power Control Configuration
#define POWER_CONTROL_PIN 17   // Key pin for MH-CD42 module
#define POWER_CONTROL_PIO pio1 // Using pio1 to avoid conflicts
#define POWER_CONTROL_SM 1     // State machine 1
#define POWER_FREQ 2           //  frequency
#define POWER_DUTY_CYCLE 30    // 50% duty cycle

// NRF24L01 Configuration
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

// Register Read/Write commands
#define NRF_R_REGISTER    0x00
#define NRF_W_REGISTER    0x20
#define NRF_NOP           0xFF

// Data rates (RF_SETUP register values)
#define RF_DR_2MBPS     0x08  // 2 Mbps mode (default)
#define RF_DR_1MBPS     0x00  // 1 Mbps mode
#define RF_DR_250KBPS   0x20  // 250 kbps mode

// NRF24L01 Power Levels
#define RF_PWR_0DBM    0x06  // Maximum power: 0dBm
#define RF_PWR_NEG_6DBM  0x04  // -6dBm
#define RF_PWR_NEG_12DBM 0x02  // -12dBm
#define RF_PWR_NEG_18DBM 0x00  // Minimum power: -18dBm

#define NRF_CHANNEL 2  // Default channel for NRF24L01 (2.4GHz band)
#define NODE_ID "01"    // Node identifier
#define MAX_PAYLOAD_SIZE 10  // NRF24L01 max payload size
uint8_t address[5] = {'g','y','r','o','c'};  // 5-byte address gyroc
// GPS location structure
struct gps_location_t {
    bool valid;
    float latitude;
    float longitude;
} g_current_location = {false, 0.0f, 0.0f};

// Message structure for transmitting data
typedef struct {
    char node_id[3];        // 2 chars + null terminator
    char date[11];          // DD/MM/YYYY + null terminator
    char time[9];           // HH:MM:SS + null terminator
    char latitude[12];      // Including decimal point and sign
    char longitude[12];     // Including decimal point and sign
    float confidence;
} gunshot_message_t;


// GPS Message types we're interested in
#define NMEA_GPRMC "$GPRMC"
#define NMEA_GPGGA "$GPGGA"

// Global variables and mutex
static mutex_t printf_mutex;
static mutex_t audio_mutex;  // Mutex for audio data sharing between cores
volatile bool core1_ready = false;
volatile bool gunshot_detected = false;
volatile float latest_confidence = 0.0f;  // Latest confidence from ML model

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
static bool gps_datetime_valid = false;

// Audio buffer
static int32_t audio_buffer[SAMPLE_BUFFER_SIZE];
NRF24 nrf = NRF24(spi0, NRF_SCK_PIN, NRF_MOSI_PIN, NRF_MISO_PIN, NRF_CSN_PIN, NRF_CE_PIN, NRF_IRQ_PIN);

void init_sync(void);
void init_uart(void);
void safe_uart_puts(const char* str);
bool init_power_control(void);
void update_power_control(float freq, float duty_cycle);
void init_gps(void);
bool wait_for_gps_fix(uint32_t timeout_ms);
void process_gps(void);
void safe_printf(const char* format, ...);
bool parse_gprmc(const char* sentence, float* out_lat, float* out_lon, bool* out_valid);
float nmea_to_decimal_degrees(float nmea_coord, char direction);
void gps_uart_irq_handler();
void core1_entry(void);
void get_gps_datetime(char* date_out, char* time_out, bool* valid_out);
bool test_microphone();
void nrf24l01_send_message(const char* message);
void create_json_message(char* buffer, size_t buffer_size, const gunshot_message_t* msg);
bool init_microphone();
static inline void configure_dma(void);
void test_fragmentation(void);

// Initialize power control
bool init_power_control(void) {
    safe_uart_puts("Initializing power control...\n");
    
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

// Initialize mutex for printf synchronization
void init_sync(void) {
    mutex_init(&printf_mutex);
    mutex_init(&audio_mutex);  // Initialize audio data mutex
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
    uart_puts(UART_ID, "MEL!\r\n");

    // Initialize model handler
    int init_status = model_init();
    if (init_status != 0) {
        safe_uart_puts("ERROR: Failed to initialize model handler\r\n");
    } else {
        safe_uart_puts("Model handler initialized successfully\r\n");
    }
}

// Safe UART output function
void safe_uart_puts(const char* str) {
    mutex_enter_blocking(&printf_mutex);
    uart_puts(UART_ID, str);
    mutex_exit(&printf_mutex);
}

void init_gps(void) {
    safe_uart_puts("\r\n=== GPS Initialization Starting ===\r\n");
    
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
    // if (now - last_status >= 30000) {  // Changed from 5000 to 30000 (30 seconds)
    //     safe_printf("\r\n[GPS] Status: Sentences: %lu, Last valid: %lu ms ago\r\n", 
    //                 sentence_count,
    //                 now - last_valid_sentence);
    //     last_status = now;
    // }
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
            snprintf(gps_time, sizeof(gps_time), "%s:%s:%s", hours, minutes, seconds);
            
            // Parse date DDMMYY
            char day[3] = {date_str[0], date_str[1], 0};
            char month[3] = {date_str[2], date_str[3], 0};
            char year[3] = {date_str[4], date_str[5], 0};
            int year_int = atoi(year) + 2000;  // Convert YY to 20YY
            snprintf(gps_date, sizeof(gps_date), "%s/%s/%04d", day, month, year_int);
            
            gps_datetime_valid = true;
            printf("[GPS] Updated date/time: %s %s\n", gps_date, gps_time);
            
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

// Helper function to safely get GPS date/time
void get_gps_datetime(char* date_out, char* time_out, bool* valid_out) {
    mutex_enter_blocking(&gps_mutex);
    strcpy(date_out, gps_date);
    strcpy(time_out, gps_time);
    *valid_out = gps_datetime_valid;
    mutex_exit(&gps_mutex);
}

// Message fragment structure for reliable transmission
typedef struct {
    uint8_t msg_id;        // Message ID (0-255)
    uint8_t fragment_num;  // Current fragment number (0-based)
    uint8_t total_fragments; // Total number of fragments
    uint8_t data_length;   // Length of data in this fragment
    uint8_t data[6];       // Data payload (10 - 4 header bytes = 6 data bytes)
} fragment_packet_t;

// Global message ID counter
static uint8_t g_message_id = 0;
// Send message via NRF24L01 with fragmentation support
void nrf24l01_send_message(const char* message) {
    if (!message) return;
    
    size_t message_length = strlen(message);
    const uint8_t MAX_DATA_PER_FRAGMENT = 6; // 10 byte payload - 4 byte header
    
    // Calculate number of fragments needed
    size_t total_fragments = (message_length + MAX_DATA_PER_FRAGMENT - 1) / MAX_DATA_PER_FRAGMENT;
    
    // Limit to 255 fragments (uint8_t limit)
    if (total_fragments > 255) {
        // safe_uart_puts("ERROR: Message too long for fragmentation\r\n");
        return;
    }
    
    // Assign unique message ID
    uint8_t current_msg_id = g_message_id++;
    
    // char debug_buf[256];
    // snprintf(debug_buf, sizeof(debug_buf), "\r\n=== SENDING MESSAGE ===\r\n");
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Original Message: \"%s\"\r\n", message);
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Message Length: %zu bytes\r\n", message_length);
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Message ID: %d\r\n", current_msg_id);
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Total Fragments: %zu\r\n", total_fragments);
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "----------------------\r\n");
    // safe_uart_puts(debug_buf);
    
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
        
        // Display fragment details
        // snprintf(debug_buf, sizeof(debug_buf), "Fragment %zu/%zu:\r\n", fragment_num + 1, total_fragments);
        // safe_uart_puts(debug_buf);
        // snprintf(debug_buf, sizeof(debug_buf), "  Header: [%d][%d][%d][%d]\r\n", 
        //          packet.msg_id, packet.fragment_num, packet.total_fragments, packet.data_length);
        // safe_uart_puts(debug_buf);
        
        // Display data content as characters
        // snprintf(debug_buf, sizeof(debug_buf), "  Data: \"");
        // safe_uart_puts(debug_buf);
        // for (int i = 0; i < data_length; i++) {
        //     char char_buf[2];
        //     char_buf[0] = packet.data[i];
        //     char_buf[1] = '\0';
        //     safe_uart_puts(char_buf);
        // }
        // safe_uart_puts("\"\r\n");
        
        // Display data content as hex
        // snprintf(debug_buf, sizeof(debug_buf), "  Hex: ");
        // safe_uart_puts(debug_buf);
        // for (int i = 0; i < 10; i++) { // Always show all 10 bytes of the packet
        //     if (i < 4) {
        //         // Header bytes
        //         uint8_t header_byte;
        //         switch(i) {
        //             case 0: header_byte = packet.msg_id; break;
        //             case 1: header_byte = packet.fragment_num; break;
        //             case 2: header_byte = packet.total_fragments; break;
        //             case 3: header_byte = packet.data_length; break;
        //         }
        //         snprintf(debug_buf, sizeof(debug_buf), "%02X ", header_byte);
        //     } else {
        //         // Data bytes
        //         snprintf(debug_buf, sizeof(debug_buf), "%02X ", packet.data[i-4]);
        //     }
        //     safe_uart_puts(debug_buf);
        // }
        // safe_uart_puts("\r\n");
        
        // Send fragment
        nrf.sendMessage((uint8_t*)&packet);
        
        // snprintf(debug_buf, sizeof(debug_buf), "  Status: SENT\r\n");
        // safe_uart_puts(debug_buf);
        
        // Small delay between fragments
        sleep_ms(50);
    }
    
    // snprintf(debug_buf, sizeof(debug_buf), "======================\r\n");
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Message transmission complete!\r\n");
    // safe_uart_puts(debug_buf);
    // snprintf(debug_buf, sizeof(debug_buf), "Total fragments sent: %zu\r\n\r\n", total_fragments);
    // safe_uart_puts(debug_buf);
}

// Create JSON message from data
void create_json_message(char* buffer, size_t buffer_size, const gunshot_message_t* msg) {
    snprintf(buffer, buffer_size, 
            "{\"node_id\":\"%s\",\"date\":\"%s\",\"time\":\"%s\",\"latitude\":\"%s\",\"longitude\":\"%s\",\"confidence\":%.2f}",
            msg->node_id, msg->date, msg->time, msg->latitude, msg->longitude, msg->confidence);
}

// Initialize INMP441 microphone
bool init_microphone() {
    // First run the test
        safe_uart_puts("\r\nStarting microphone test...\r\n");
    safe_uart_puts("\r\nTesting INMP441 Microphone:\r\n");
    safe_uart_puts("1. Checking PIO configuration...\r\n");
    safe_uart_puts("Checking PIO program size...\r\n");
    
    if (!pio_can_add_program(INMP441_PIO, &inmp441_program)) {
        safe_uart_puts("ERROR: Cannot load PIO program - insufficient space\r\n");
        safe_uart_puts("\r\nMicrophone initialization failed! Check connections and try again.\r\n");
        return false;
    }
    
    safe_uart_puts("2. Loading PIO program...\r\n");
    uint offset = pio_add_program(INMP441_PIO, &inmp441_program);
    
    char buf[128];
    snprintf(buf, sizeof(buf), "   Program loaded at offset %u\r\n", offset);
    safe_uart_puts(buf);
    
    safe_uart_puts("3. Initializing GPIO pins...\r\n");
    snprintf(buf, sizeof(buf), "   SCK (BCLK): GPIO %d\r\n", INMP441_SCK_PIN);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "   WS (LRCLK): GPIO %d\r\n", INMP441_WS_PIN);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "   SD (DATA):  GPIO %d\r\n", INMP441_SD_PIN);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "   Sample Rate: %d Hz\r\n", SAMPLE_RATE);
    safe_uart_puts(buf);
    
    // Initialize the PIO state machine
    snprintf(buf, sizeof(buf), "   Initializing PIO state machine %d on PIO%d\r\n", 
                INMP441_SM, (INMP441_PIO == pio0) ? 0 : 1);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "   Clock frequency: %lu Hz\r\n", clock_get_hz(clk_sys));
    safe_uart_puts(buf);
    
    inmp441_program_init(INMP441_PIO, INMP441_SM, offset, SAMPLE_RATE, INMP441_SD_PIN, INMP441_SCK_PIN);
    safe_uart_puts("   PIO initialization complete\r\n");
    
    safe_uart_puts("4. Configuring DMA...\r\n");
    snprintf(buf, sizeof(buf), "   DMA Channel: %d\r\n", DMA_CHANNEL);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "   Buffer size: %d samples\r\n", SAMPLE_BUFFER_SIZE);
    safe_uart_puts(buf);
    configure_dma();
    safe_uart_puts("   DMA configuration complete\r\n");
    
    safe_uart_puts("5. Starting audio capture test...\r\n");
    safe_uart_puts("   Please make some noise near the microphone!\r\n\r\n");
    
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    bool received_audio = false;
    int32_t min_sample = 0x7FFFFFFF;
    int32_t max_sample = -0x7FFFFFFF;
    
    safe_uart_puts("   Waiting for DMA transfer...\r\n");

    while (to_ms_since_boot(get_absolute_time()) - start_time < MIC_TEST_DURATION_MS) {
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        
        
        // Process samples
        for (int i = 0; i < DEBUG_SAMPLES; i++) {
            int32_t sample = audio_buffer[i] >> 8;  // Convert to 24-bit
            if (sample != 0) {
                received_audio = true;
            }
            if (sample < min_sample) min_sample = sample;
            if (sample > max_sample) max_sample = sample;
        }
        
        // Restart DMA
        dma_channel_set_write_addr(DMA_CHANNEL, audio_buffer, true);
        
        if (received_audio) {
            safe_uart_puts("   DMA transfer complete\r\n");
            break;  // Got some samples, no need to wait longer
        }
    }
    
    safe_uart_puts("\r\nMicrophone Test Results:\r\n");
    snprintf(buf, sizeof(buf), "- Audio data received: %s\r\n", received_audio ? "YES" : "NO");
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Min sample value: %ld\r\n", min_sample);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Max sample value: %ld\r\n", max_sample);
    safe_uart_puts(buf);
    snprintf(buf, sizeof(buf), "- Sample range: %ld\r\n", max_sample - min_sample);
    safe_uart_puts(buf);
    
    if (!received_audio) {
        safe_uart_puts("\r\nERROR: No audio data received. Please check:\r\n");
        safe_uart_puts("1. All connections are secure\r\n");
        safe_uart_puts("2. VDD and GND are properly connected\r\n");
        safe_uart_puts("3. L/R pin is connected to GND\r\n");
        safe_uart_puts("\r\nMicrophone initialization failed! Check connections and try again.\r\n");
        return false;
    }
    
    if (max_sample - min_sample < 1000) {
        safe_uart_puts("\r\nWARNING: Very low audio range detected.\r\n");
        safe_uart_puts("Try making louder sounds near the microphone.\r\n");
    } else {
        safe_uart_puts("\r\nMicrophone test PASSED! ✓\r\n");
    }

    // No need to reconfigure since test_microphone already set everything up
    return true;
}

// DMA configuration
static inline void configure_dma(void) {
    safe_uart_puts("   Starting DMA configuration...\r\n");
    
    // Get default channel configuration
    dma_channel_config c = dma_channel_get_default_config(DMA_CHANNEL);
    
    char buf[64];
    snprintf(buf, sizeof(buf), "   Got default DMA config for channel %d\r\n", DMA_CHANNEL);
    safe_uart_puts(buf);
    
    // Configure channel
    channel_config_set_read_increment(&c, false);  // Don't increment read address (reading from same PIO FIFO)
    channel_config_set_write_increment(&c, true);  // Do increment write address (writing to buffer)
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);  // Transfer 32-bit words
    channel_config_set_dreq(&c, pio_get_dreq(INMP441_PIO, INMP441_SM, false));  // Pace transfers based on PIO

    dma_channel_configure(
        DMA_CHANNEL,
        &c,
        audio_buffer,                    // Destination pointer
        &INMP441_PIO->rxf[INMP441_SM],  // Source pointer
        SAMPLE_BUFFER_SIZE,              // Number of transfers
        true                            // Start immediately
    );
}

// Core 1 entry point - DEDICATED TO ML MODEL AND AUDIO PROCESSING
void core1_entry() {
    safe_uart_puts("\r\n=== CORE 1: ML MODEL PROCESSOR ===\r\n");
    safe_uart_puts("Core 1: Dedicated to gunshot detection ML model\r\n");
    sleep_ms(100); // Give time for UART message
    
    safe_uart_puts("Core 1: Initializing microphone...\r\n");
    sleep_ms(100); // Give time for UART message
    
    // Initialize microphone
    safe_uart_puts("Core 1: Starting microphone initialization...\r\n");
    bool init_success = init_microphone();
    safe_uart_puts("Core 1: Microphone initialization attempt complete\r\n");
    
    if (!init_success) {
        safe_uart_puts("Core 1: Microphone initialization failed\r\n");
        safe_uart_puts("Core 1: Check hardware connections and try again\r\n");
        core1_ready = true;  // Signal core0 even on failure so it doesn't hang
        while (true) {
            gpio_put(LED_PIN, true);   // Error indication pattern
            sleep_ms(100);
            gpio_put(LED_PIN, false);
            sleep_ms(900);             // Blink once per second to show core is alive
        }
    }
    
    safe_uart_puts("Core 1: Microphone initialized successfully\r\n");
    safe_uart_puts("Core 1: Initializing ML model...\r\n");
    
    // Initialize ML model
    int model_status = model_init();
    if (model_status != 0) {
        safe_uart_puts("Core 1: ERROR - ML model initialization failed!\r\n");
        core1_ready = true;
        while (true) {
            gpio_put(LED_PIN, true);   // Fast error blink
            sleep_ms(50);
            gpio_put(LED_PIN, false);
            sleep_ms(50);
        }
    }
    
    safe_uart_puts("Core 1: ML model initialized successfully\r\n");
    safe_uart_puts("Core 1: Starting audio processing loop...\r\n");
    core1_ready = true;  // Signal core0 that we're ready
    
    // ML processing variables
    uint32_t processing_count = 0;
    uint32_t last_status_update = 0;
    uint32_t last_debug_update = 0;
    
    while (true) {
        // Wait for DMA transfer to complete
        dma_channel_wait_for_finish_blocking(DMA_CHANNEL);
        
        // Simple audio analysis for debugging
        int32_t max_sample = 0;
        int32_t min_sample = 0;
        for (int i = 0; i < SAMPLE_BUFFER_SIZE; i++) {
            int32_t sample = audio_buffer[i] >> 8;  // Convert to 24-bit
            if (sample > max_sample) max_sample = sample;
            if (sample < min_sample) min_sample = sample;
        }
        int32_t amplitude_range = max_sample - min_sample;
        
        // Process audio data using the ML model
        bool detection = model_process_audio(audio_buffer, SAMPLE_BUFFER_SIZE, CONFIDENCE_THRESHOLD);
        float confidence = model_get_confidence();
        
        // Update shared confidence value
        mutex_enter_blocking(&audio_mutex);
        latest_confidence = confidence;
        mutex_exit(&audio_mutex);
        
        processing_count++;
        
        // Print detailed debug every 10 seconds when confidence is detected
        uint32_t now = to_ms_since_boot(get_absolute_time());
        if (now - last_debug_update > 10000 || confidence > 0.1f) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Core 1 Debug: Frame %lu, Range: %ld, Max: %ld, Min: %ld, Confidence: %.4f\r\n", 
                     processing_count, amplitude_range, max_sample, min_sample, confidence);
            safe_uart_puts(buf);
            last_debug_update = now;
        }
        
        // Print status every 5 seconds (less frequent to reduce UART traffic)
        // if (now - last_status_update > 5000) {
        //     char buf[128];
        //     snprintf(buf, sizeof(buf), "Core 1: Processed %lu audio frames, Latest confidence: %.3f\r\n", 
        //              processing_count, confidence);
        //     safe_uart_puts(buf);
        //     last_status_update = now;
        // }

        // Control LED based on confidence level
        if (confidence > 0.5f) {
            gpio_put(LED_PIN, true);   // High confidence - LED on
        } else if (confidence > 0.2f) {
            // Medium confidence - pulse LED
            gpio_put(LED_PIN, (processing_count % 10) < 5);
        } else {
            gpio_put(LED_PIN, false);  // Low confidence - LED off
        }
        
        // If gunshot detected, signal Core 0
        if (detection) {
            gunshot_detected = true;
            // char buf[128];
            // snprintf(buf, sizeof(buf), "Core 1: *** GUNSHOT DETECTED! *** Confidence: %.3f\r\n", confidence);
            // safe_uart_puts(buf);
            
            // Keep LED on for 2 seconds after detection
            for (int i = 0; i < 40; i++) {
                gpio_put(LED_PIN, true);
                sleep_ms(50);
            }
        }
        
        // Restart DMA transfer for next audio frame
        dma_channel_set_write_addr(DMA_CHANNEL, audio_buffer, true);
    }
}


int main() {
    stdio_init_all();
    init_sync();
    init_uart();
    sleep_ms(20);
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
    gpio_set_dir(LED_PIN, GPIO_OUT);  // Turn off LED initially

    safe_uart_puts("\r\n=== Program Starting (GPS Debug Mode) ===\r\n");
    safe_uart_puts("Initializing GPS...\r\n");
    init_gps();
    safe_uart_puts("Starting GPS monitoring...\r\n");
    // Wait for initial GPS fix with timeout
    safe_uart_puts("Waiting for initial GPS fix (timeout: 60 seconds)...\r\n");
    if (wait_for_gps_fix(60000)) {  // 60 second timeout for first fix
        safe_uart_puts("GPS fix acquired successfully!\r\n");
    } else {
        safe_uart_puts("GPS fix timeout - continuing without initial fix.\r\n");
    }


    sleep_ms(100); // give time for things to settle.
    
    // Initialize NRF24L01
    nrf.config((uint8_t*)"gyroc", 2, 10); // Name=gyroc, channel=2,messagSize=10
    nrf.modeTX(); // <---- Set as transmitter.
    
    safe_uart_puts("NRF24L01 initialized successfully\r\n");


    // Launch Core 1 for ML processing
    safe_uart_puts("Starting Core 1 (ML Processor)...\r\n");
    multicore_launch_core1(core1_entry);

    // Wait for Core 1 to be ready
    safe_uart_puts("Waiting for Core 1 to initialize...\r\n");
    while (!core1_ready) {
        sleep_ms(100);
        gpio_put(LED_INBUILT_IN, !gpio_get(LED_INBUILT_IN)); // Blink while waiting
    }
    gpio_put(LED_INBUILT_IN, true); // Solid on when ready
    safe_uart_puts("Core 1 is ready!\r\n");

    // Main loop - Core 0 handles GPS, NRF24L01, and gunshot event processing
    safe_uart_puts("\r\n=== CORE 0: COMMUNICATION & GPS HANDLER ===\r\n");
    safe_uart_puts("Core 0: Monitoring for gunshot events from Core 1...\r\n\r\n");
    
    uint32_t last_status = 0;
    uint32_t gunshot_count = 0;

    while (1) {
        // Check if gunshot was detected by Core 1
        if (gunshot_detected) {
            gunshot_count++;
            
            // Get current confidence from Core 1
            float confidence;
            mutex_enter_blocking(&audio_mutex);
            confidence = latest_confidence;
            mutex_exit(&audio_mutex);
            
            // char buf[128];
            // snprintf(buf, sizeof(buf), "\r\nCore 0: Processing gunshot event #%lu\r\n", gunshot_count);
            // safe_uart_puts(buf);
            // snprintf(buf, sizeof(buf), "Core 0: Confidence: %.3f\r\n", confidence);
            // safe_uart_puts(buf);
            // snprintf(buf, sizeof(buf), "Core 0: Location - Lat: %.6f, Lon: %.6f\r\n", latitude, longitude);
            // safe_uart_puts(buf);
            
            // Prepare and send message via NRF24L01
            gunshot_message_t msg;
            strncpy(msg.node_id, NODE_ID, sizeof(msg.node_id) - 1);
            msg.node_id[sizeof(msg.node_id) - 1] = '\0';  // Ensure null termination
            
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
            
            // Set latitude, longitude, and confidence
            snprintf(msg.latitude, sizeof(msg.latitude), "%.6f", latitude);
            snprintf(msg.longitude, sizeof(msg.longitude), "%.6f", longitude);
            msg.confidence = confidence;
            
            // Create JSON message
            char json_buffer[1024];
            create_json_message(json_buffer, sizeof(json_buffer), &msg);
            
            // safe_uart_puts("Core 0: Sending gunshot alert via NRF24L01...\r\n");
            
            // Send message with fragmentation
            nrf24l01_send_message(json_buffer);
            
            // safe_uart_puts("Core 0: Gunshot alert sent successfully\r\n\r\n");
            
            // Reset the detection flag
            gunshot_detected = false;
        }
        
        // Print status every 30 seconds
        uint32_t now = to_ms_since_boot(get_absolute_time());
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

// Test fragmentation function
void test_fragmentation() {
    safe_uart_puts("\r\n=== Testing NRF24L01 Message Fragmentation ===\r\n");
    
    // Test 1: Short message (fits in one fragment)
    safe_uart_puts("Test 1: Short message\r\n");
    nrf24l01_send_message("Hello");
    
    sleep_ms(500);
    
    // Test 2: Medium message (needs 2-3 fragments)
    safe_uart_puts("\r\nTest 2: Medium message\r\n");
    nrf24l01_send_message("This is a longer message that will need fragmentation");
    
    sleep_ms(500);
    
    // Test 3: JSON message (typical gunshot detection message)
    safe_uart_puts("\r\nTest 3: JSON gunshot message\r\n");
    char test_json[] = "{\"node_id\":\"01\",\"date\":\"03/07/2025\",\"time\":\"12:34:56\",\"latitude\":\"12.345678\",\"longitude\":\"98.765432\",\"confidence\":0.95}";
    nrf24l01_send_message(test_json);
    
    safe_uart_puts("\r\n=== Fragmentation Test Complete ===\r\n");
}

