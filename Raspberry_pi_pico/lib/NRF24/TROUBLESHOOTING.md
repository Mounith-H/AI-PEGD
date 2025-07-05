# NRF24L01 Communication Troubleshooting Guide

## Current Configuration Status

### Transmitter (Your Pico)
- Channel: 72 ✓
- Data Rate: 2Mbps ✓
- Power Level: Low ✓
- CRC Length: 16-bit ✓
- Auto Ack: Disabled ✓
- Dynamic Payloads: Enabled ✓
- Address: "gyroc" ✓

### Receiver (Other Device)
- Channel: 72 ✓
- Data Rate: 2Mbps ✓
- Power Level: Low ✓
- CRC Length: 16-bit ✓
- Auto Ack: Disabled ✓
- Dynamic Payloads: Enabled ✓
- Address: **UNKNOWN** ❓

## Troubleshooting Steps

### 1. Address Mismatch (Most Common Issue)
**Problem**: Transmitter and receiver must use the same address
**Solution**: 
- Check if your receiver is using address "gyroc" 
- OR modify transmitter to use the same address as receiver

### 2. Hardware Connection Check
**Your Transmitter Pins:**
```
CE  -> GP6
CSN -> GP5
SCK -> GP2
MOSI-> GP3
MISO-> GP4
```

**Your Receiver Pins (from your output):**
```
CE  -> PA4
CSN -> PA3
SCK -> PA5
MOSI-> PA7
MISO-> PA6
```

**Action**: Verify all connections are solid and correct

### 3. Power Supply Issues
**Check:**
- 3.3V supply (NOT 5V)
- Current capacity (min 200mA for PA/LNA modules)
- Stable power (add 10µF + 100nF capacitors)

### 4. Distance and Environment
**Test Steps:**
1. Start with modules very close (< 50cm)
2. Ensure line of sight
3. Avoid metal objects near antennas
4. Test in different locations

### 5. Debugging Commands

**Add these to your transmitter code to debug:**

```cpp
// After nrf.begin()
printf("Module details:\n");
nrf.printDetails();

// In the main loop, add more debug info
printf("Status: 0x%02X\n", nrf.getStatus());
printf("FIFO Status: TX_EMPTY=%s, TX_FULL=%s\n", 
       nrf.isTxFifoEmpty() ? "YES" : "NO",
       nrf.isTxFifoFull() ? "YES" : "NO");
```

**Test carrier detection:**
```cpp
if (nrf.testCarrier()) {
    printf("Carrier detected - channel may be busy\n");
}
```

### 6. Alternative Testing

**Option 1: Test with Known Good Receiver**
Use the `test_receiver.cpp` file I created with your second Pico

**Option 2: Test Different Settings**
Try these more robust settings:

```cpp
// More robust configuration
nrf.setPowerLevel(NRF24_POWER_LEVEL_0DBM);    // Max power
nrf.setDataRate(NRF24_DATA_RATE_250KBPS);     // Slower but more reliable
nrf.setChannel(76);                            // Different channel
nrf.setAutoAck(true);                          // Enable auto-ack
nrf.disableDynamicPayloads();                  // Use fixed payload
nrf.setPayloadSize(32);                        // Fixed size
```

### 7. Step-by-Step Verification

**Step 1: Verify Module Detection**
```cpp
if (!nrf.begin()) {
    printf("Module not detected!\n");
    // Check wiring
}
```

**Step 2: Verify Configuration**
```cpp
printf("Actual channel: %d\n", nrf.getChannel());
printf("Actual data rate: %d\n", nrf.getDataRate());
```

**Step 3: Test Transmission**
```cpp
// Send simple data first
uint8_t test_data[] = "TEST";
bool result = nrf.write(test_data, 4);
printf("Send result: %s\n", result ? "SUCCESS" : "FAILED");
```

## Quick Fixes to Try

### Fix 1: Use Same Address as Receiver
**Find out what address your receiver uses and update transmitter:**
```cpp
// Example: if receiver uses address "HELLO"
uint8_t tx_address[] = {'H', 'E', 'L', 'L', 'O'};
nrf.openWritingPipe(tx_address);
```

### Fix 2: Use Fixed Payload Size
**Both devices should use same payload size:**
```cpp
// Disable dynamic payloads
nrf.disableDynamicPayloads();
nrf.setPayloadSize(32);

// Send fixed size
char message[32] = "Test message";
nrf.write((uint8_t*)message, 32);
```

### Fix 3: Enable Auto-Acknowledgment
**This helps verify communication:**
```cpp
nrf.setAutoAck(true);
nrf.setRetries(5, 15);
```

### Fix 4: Test with Simple Data
**Use basic data first:**
```cpp
uint8_t counter = 0;
while (true) {
    bool result = nrf.write(&counter, 1);
    printf("Sent: %d, Result: %s\n", counter, result ? "OK" : "FAIL");
    counter++;
    sleep_ms(1000);
}
```

## Expected Output When Working

**Transmitter should show:**
```
✓ Sent (2ms): Message #1
✓ Sent (2ms): Message #2
✓ Sent (2ms): Message #3
```

**Receiver should show:**
```
✓ Received from pipe 1 (12 bytes): Message #1
✓ Received from pipe 1 (12 bytes): Message #2
✓ Received from pipe 1 (12 bytes): Message #3
```

## Next Steps

1. **First**: Check if receiver address matches "gyroc"
2. **Second**: Test with two Picos using my code
3. **Third**: Try different channel (76 instead of 72)
4. **Fourth**: Enable auto-ack for better reliability
5. **Fifth**: Use fixed payload size instead of dynamic

Let me know what you discover!
