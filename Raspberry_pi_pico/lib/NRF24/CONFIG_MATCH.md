# NRF24L01 Configuration Match - Pico Transmitter ↔ STM32 Receiver

## ✅ MATCHED CONFIGURATIONS

### STM32 BluePill Receiver (Your working code)
```cpp
// Pin connections
CE  -> PA4
CSN -> PA3
SCK -> PA5
MOSI-> PA7
MISO-> PA6
IRQ -> PA2 (not used)

// Radio configuration
radio.setChannel(2);                    // Channel 2
radio.setDataRate(RF24_2MBPS);          // 2 Mbps
radio.setPALevel(RF24_PA_LOW);          // Low power
radio.setCRCLength(RF24_CRC_16);        // 16-bit CRC
radio.setAutoAck(true);                 // Auto-ack ENABLED
radio.enableDynamicPayloads();          // Dynamic payloads enabled
radio.openReadingPipe(0, "gyroc");      // Address: "gyroc"
```

### Raspberry Pi Pico Transmitter (Updated to match)
```cpp
// Pin connections
CE  -> GP6
CSN -> GP5
SCK -> GP2
MOSI-> GP3
MISO-> GP4
IRQ -> GP7 (not used)

// Radio configuration
nrf.setChannel(2);                      // Channel 2 ✓
nrf.setDataRate(NRF24_DATA_RATE_2MBPS); // 2 Mbps ✓
nrf.setPowerLevel(NRF24_POWER_LEVEL_NEG12DBM); // Low power ✓
nrf.setCRCLength(NRF24_CRC_16BIT);      // 16-bit CRC ✓
nrf.setAutoAck(true);                   // Auto-ack ENABLED ✓
nrf.setRetries(5, 15);                  // 5 retries, 4ms delay
nrf.enableDynamicPayloads();            // Dynamic payloads enabled ✓
nrf.openWritingPipe("gyroc");           // Address: "gyroc" ✓
```

## Expected Communication Flow

### Pico Transmitter Output:
```
NRF24L01 Transmitter Test
Pin Configuration:
CE  -> GP6
CSN -> GP5
SCK -> GP2
MOSI-> GP3
MISO-> GP4
IRQ -> GP7 (not used)

NRF24L01 initialized successfully!
Radio Configuration:
Channel: 2
Data Rate: 2Mbps
Power Level: Low
CRC Length: 16-bit
Auto Ack: Enabled
Dynamic Payloads: Enabled
TX Address: gyroc
Transmitting data...
Sending messages to STM32 receiver...

✓ Sent (3ms): Message #1
✓ Sent (2ms): Message #2
✓ Sent (2ms): Message #3
```

### STM32 Receiver Output:
```
NRF24L01 Receiver Test
Pin Configuration:
CE  -> PA4
CSN -> PA3
SCK -> PA5
MOSI-> PA7
MISO-> PA6
IRQ -> PA2 (not used)

Radio Configuration:
Channel: 2
Data Rate: 2Mbps
Power Level: Low
CRC Length: 16-bit
Auto Ack: Enabled
Dynamic Payloads: Enabled
Listening for data...
Waiting for incoming transmissions...

Received: Message #1 || [4D 65 73 73 61 67 65 20 23 31]
Received: Message #2 || [4D 65 73 73 61 67 65 20 23 32]  
Received: Message #3 || [4D 65 73 73 61 67 65 20 23 33]
```

## Key Changes Made

### ❌ Previous Issues:
- Channel mismatch: Transmitter=72, Receiver=2
- Auto-ack mismatch: Transmitter=disabled, Receiver=enabled

### ✅ Fixed Configuration:
- Channel: Both use 2
- Auto-ack: Both enabled
- Retry mechanism: Added to transmitter
- All other settings already matched

## Testing Steps

1. **Upload corrected Pico code** (already compiled ✓)
2. **Power on STM32 receiver first**
3. **Power on Pico transmitter second**
4. **Observe both serial outputs**

## What You Should See

### Success Indicators:
- Pico shows "✓ Sent" messages
- STM32 shows "Received:" messages with data
- LED1 on STM32 toggles on each reception
- Pico LED blinks briefly on successful transmission

### Failure Indicators:
- Pico shows "✗ Failed to send" messages
- STM32 shows no "Received:" messages
- No LED activity on either device

## Troubleshooting (if still no communication)

### 1. Power Supply Check
- Ensure both modules have stable 3.3V supply
- Add decoupling capacitors (10µF + 100nF) near each module

### 2. Distance Test
- Start with modules very close together (< 50cm)
- Ensure line of sight between antennas

### 3. Hardware Verification
- Double-check all SPI connections
- Verify CE and CSN pins are correctly connected
- Test continuity with multimeter

### 4. Alternative Test
If still no communication, try these simpler settings:
```cpp
// On both devices, use more robust settings
Channel: 76 (less WiFi interference)
Data Rate: 250kbps (more reliable)
Power: Maximum
Auto-ack: Enabled
Fixed payload: 32 bytes (disable dynamic)
```

## Expected Result
With these matched configurations, your Pico should now successfully communicate with your STM32 receiver. The main issues (channel and auto-ack mismatch) have been resolved.

Upload the updated Pico code and test! 🚀
