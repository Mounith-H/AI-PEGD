# AI Powered Edge Based Gunshot Detection for Protecting Wildlife

## Project Overview
This project implements an intelligent gunshot detection system using Raspberry Pi Pico for wildlife protection. The system uses machine learning to detect gunshot sounds and reports their location in real-time, helping authorities respond quickly to potential poaching incidents.

## Current Implementation Status

### Completed Features
1. **Hardware Integration**
   - INMP441 Digital Microphone setup with I2S interface
   - GPS Module (NEO-6M) integration with UART
   - NRF24L01 Radio Module for wireless communication
   - Power Management using IP5306 with PIO-based PWM control:
     - Continuous KEY pin pulsing using PIO state machine
     - 16-bit resolution PWM implementation
     - Automatic power maintenance during operation

2. **Core Functionality**
   - Real-time audio sampling using PIO and DMA
   - GPS location tracking with NMEA parsing
   - Wireless data transmission with packet fragmentation
   - Power management with robust PIO implementation:
     - Efficient KEY pin control
     - Reliable power maintenance
     - Clean initialization sequence

3. **System Architecture**
   - Dual-core operation:
     - Core 0: System management, GPS, and radio communication
     - Core 1: Audio processing and gunshot detection
   - PIO-based peripheral management:
     - Audio sampling
     - Power control
     - Communication protocols

4. **Reliability Features**
   - Robust power management with PWM-based IP5306 control:
     - Continuous operation guarantee
     - Power stability monitoring
     - Automatic recovery on power events
   - Error handling and recovery mechanisms
   - Packet fragmentation for reliable data transmission
   - Automatic retry mechanisms for radio communication

### Work in Progress
1. **Machine Learning Integration**
   - TensorFlow Lite model integration
   - Model optimization for Pico
   - Real-time inference implementation

2. **Communication Protocol**
   - Mesh network implementation
   - Data encryption
   - Acknowledgment system

### Future Work
1. **System Features**
   - Web interface for monitoring
   - Mobile app development
   - Multiple node coordination
   - Over-the-air updates

2. **ML Improvements**
   - Model retraining with field data
   - False positive reduction
   - Sound classification expansion

3. **Hardware Improvements**
   - Weather-proof enclosure design
   - Solar charging integration
   - Extended range communication

## Hardware Setup
### Components
- Raspberry Pi Pico
- INMP441 Digital Microphone
- NEO-6M GPS Module
- NRF24L01 Radio Module
- IP5306 Power Management IC

- **Power Control**:
  - Power control implemented through PIO state machine
  - Continuous PWM signal maintains power module operation

## Building and Running
### Prerequisites
1. Install CMake (version 3.12 or higher)
2. Install Raspberry Pi Pico SDK
3. Set PICO_SDK_PATH environment variable

### Build Instructions
1. Create build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Generate build files:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   make
   ```

4. Flash the binary:
   - Hold BOOTSEL button while connecting Pico
   - Copy generated .uf2 file to mounted drive

### Power Management Notes
- The system uses PIO-based power control for the IP5306
- KEY pin requires continuous pulsing to maintain power
- PWM implementation ensures reliable operation
- No manual intervention needed after initial power-up

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
