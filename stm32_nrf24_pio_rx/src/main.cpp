#include <Arduino.h>
#include <SPI.h>
#include "printf.h"
#include "RF24.h"

RF24 radio(PA4, PA3);  // using pin PA4 for the CE pin, and pin PA3 for the CSN pin
// IRQ pin is connected to PA2 but not used in this basic example

// Debug serial on PA9 (TX) and PA10 (RX) - Hardware Serial2
HardwareSerial DebugSerial(PA10, PA9); // RX, TX pins

// Data reconstruction variables
String completeMessage = "";
unsigned long lastPacketTime = 0;
const unsigned long PACKET_TIMEOUT = 5000; // 5 seconds timeout between packets
int fragmentCount = 0;
int expectedFragments = 0; // Will be set from header
int currentMessageId = -1;

void parseAndDisplayData(String jsonData);

void setup() {
  Serial.begin(115200);      // USB Serial - for JSON output only
  DebugSerial.begin(115200); // Debug Serial on PA9/PA10 - for debug messages
  
  pinMode(PC13, OUTPUT); // LED pin for indication
  
  // Initialize radio
  if (!radio.begin()) {
    DebugSerial.println("Radio hardware not responding!");
    while(1); // Hold in infinite loop
  }
  // Configuration settings.
  radio.setChannel(2);
  radio.setDataRate(RF24_2MBPS); // 2 Mbps
  radio.setPALevel(RF24_PA_LOW); // Set power level
  radio.openReadingPipe(0, (uint8_t*)"gyroc");  // using pipe 0
  radio.setCRCLength(RF24_CRC_16); // Set CRC length
  radio.setAutoAck(true); // enable AutoAck.
  radio.enableDynamicPayloads(); // Enable dynamic payloads
  // END Configuration settings.
  
  radio.startListening();  // put radio in RX mode
  delay(1000); // Allow some time for the radio to initialize
  
  DebugSerial.println("NRF24L01 Receiver Test");
  DebugSerial.println("Pin Configuration:");
  DebugSerial.println("CE  -> PA4");
  DebugSerial.println("CSN -> PA3"); 
  DebugSerial.println("SCK -> PA5");
  DebugSerial.println("MOSI-> PA7");
  DebugSerial.println("MISO-> PA6");
  DebugSerial.println("IRQ -> PA2 (not used)");
  DebugSerial.println("DEBUG TX -> PA9");
  DebugSerial.println("DEBUG RX -> PA10");
  
  // Print radio configuration details
  DebugSerial.println("\nRadio Configuration:");
  DebugSerial.print("Channel: "); DebugSerial.println(radio.getChannel());
  DebugSerial.print("Data Rate: ");
  if (radio.getDataRate() == RF24_2MBPS) DebugSerial.println("2Mbps");
  else if (radio.getDataRate() == RF24_1MBPS) DebugSerial.println("1Mbps");
  else DebugSerial.println("250kbps");
  DebugSerial.print("Power Level: ");
  if (radio.getPALevel() == RF24_PA_LOW) DebugSerial.println("Low");
  else if (radio.getPALevel() == RF24_PA_HIGH) DebugSerial.println("High");
  else if (radio.getPALevel() == RF24_PA_MAX) DebugSerial.println("Max");
  else DebugSerial.println("Min");
  DebugSerial.print("CRC Length: ");
  if (radio.getCRCLength() == RF24_CRC_16) DebugSerial.println("16-bit");
  else DebugSerial.println("8-bit");
  DebugSerial.print("Auto Ack: "); DebugSerial.println(radio.isAckPayloadAvailable() ? "Enabled" : "Disabled");
  DebugSerial.print("Dynamic Payloads: "); DebugSerial.println("Enabled");
  
  DebugSerial.println("Listening for data...");
  DebugSerial.println("Waiting for incoming transmissions...");
  DebugSerial.println("Expecting messages with 4-byte header + 20-byte data format");
  DebugSerial.println("USB Serial: JSON output only");
  DebugSerial.println("Debug Serial (PA9/PA10): Debug messages");
}  // setup

void loop() {
  static uint8_t buffer[25]; // Increased buffer size for 24 bytes + null terminator
  static char msg[150] = {0}; // Increased message buffer
  
  if (radio.available()) {
      uint8_t len = radio.getDynamicPayloadSize(); // Get payload size
      if (len >= 4 && len <= 24) { // Validate payload size (at least header, max 24 bytes)
        radio.read(buffer, len);  // Read message into buffer.
        
        // Parse header: [msg_id][fragment_num][total_fragments][data_length]
        uint8_t messageId = buffer[0];
        uint8_t fragmentNum = buffer[1];
        uint8_t totalFragments = buffer[2];
        uint8_t dataLength = buffer[3];
        
        // Extract data portion (after 4-byte header)
        char dataBuffer[21] = {0}; // Max 20 bytes data + null terminator
        if (dataLength > 0 && dataLength <= 20 && (4 + dataLength) <= len) {
          memcpy(dataBuffer, &buffer[4], dataLength);
          dataBuffer[dataLength] = '\0';
        }
        
        // Show detailed packet info on debug serial
        sprintf(msg, "Msg:%d Frag:%d/%d DataLen:%d Data:'%s'", 
                messageId, fragmentNum + 1, totalFragments, dataLength, dataBuffer);
        DebugSerial.println(msg);
        
        // Check if this is a new message
        if (currentMessageId != messageId) {
          if (completeMessage.length() > 0) {
            DebugSerial.println("New message started - clearing previous incomplete message");
          }
          completeMessage = "";
          fragmentCount = 0;
          currentMessageId = messageId;
          expectedFragments = totalFragments;
        }
        
        // Add data to complete message
        completeMessage += String(dataBuffer);
        lastPacketTime = millis();
        fragmentCount++;
        
        // Blink LED to indicate reception
        digitalWrite(PC13, LOW);  // Turn on LED (active low)
        delay(10);
        digitalWrite(PC13, HIGH); // Turn off LED
        
        // Check if we've received all expected fragments
        if (fragmentCount >= expectedFragments) {
          DebugSerial.println("\n=== COMPLETE MESSAGE RECEIVED ===");
          DebugSerial.println("Message ID: " + String(messageId));
          DebugSerial.println("Total fragments: " + String(fragmentCount));
          DebugSerial.println("Complete message: " + completeMessage);
          
          // Try to parse as JSON
          if (completeMessage.indexOf('{') != -1 && completeMessage.indexOf('}') != -1) {
            int startPos = completeMessage.indexOf('{');
            int endPos = completeMessage.lastIndexOf('}') + 1;
            
            if (startPos != -1 && endPos > startPos) {
              String jsonMessage = completeMessage.substring(startPos, endPos);
              
              // Send JSON to USB Serial ONLY
              Serial.println(jsonMessage);
              
              // Send parsed data to debug serial
              parseAndDisplayData(jsonMessage);
            }
          } else {
            DebugSerial.println("Message doesn't appear to be JSON format");
          }
          
          DebugSerial.println("=================================\n");
          
          // Clear the buffer for next message
          completeMessage = "";
          fragmentCount = 0;
          currentMessageId = -1;
          expectedFragments = 0;
        }
      }
  }
  
  // Check for timeout - clear incomplete message if too much time has passed
  if (completeMessage.length() > 0 && (millis() - lastPacketTime) > PACKET_TIMEOUT) {
    DebugSerial.println("Message timeout - clearing incomplete data");
    DebugSerial.println("Message ID: " + String(currentMessageId));
    DebugSerial.println("Fragments received before timeout: " + String(fragmentCount) + "/" + String(expectedFragments));
    DebugSerial.println("Partial message: " + completeMessage);
    completeMessage = "";
    fragmentCount = 0;
    currentMessageId = -1;
    expectedFragments = 0;
  }
  
  // Add a small delay to prevent overwhelming the serial output
  delay(10);
}

void parseAndDisplayData(String jsonData) {
  DebugSerial.println("=== PARSED DATA ===");
  
  // Extract Node ID
  int nodeIdStart = jsonData.indexOf("\"node_id\":\"") + 11;
  int nodeIdEnd = jsonData.indexOf("\"", nodeIdStart);
  if (nodeIdStart > 10 && nodeIdEnd > nodeIdStart) {
    String nodeId = jsonData.substring(nodeIdStart, nodeIdEnd);
    DebugSerial.println("Node ID: " + nodeId);
  }
  
  // Extract Message ID
  int msgIdStart = jsonData.indexOf("\"msg\":\"") + 7;
  int msgIdEnd = jsonData.indexOf("\"", msgIdStart);
  if (msgIdStart > 6 && msgIdEnd > msgIdStart) {
    String msgId = jsonData.substring(msgIdStart, msgIdEnd);
    DebugSerial.println("Message ID: " + msgId);
  }
  
  // Extract ID (if present)
  int idStart = jsonData.indexOf("\"_id\":\"") + 7;
  int idEnd = jsonData.indexOf("\"", idStart);
  if (idStart > 6 && idEnd > idStart) {
    String id = jsonData.substring(idStart, idEnd);
    DebugSerial.println("ID: " + id);
  }
  
  // Extract Date
  int dateStart = jsonData.indexOf("\"date\":\"") + 8;
  int dateEnd = jsonData.indexOf("\"", dateStart);
  if (dateStart > 7 && dateEnd > dateStart) {
    String date = jsonData.substring(dateStart, dateEnd);
    DebugSerial.println("Date: " + date);
  }
  
  // Extract Time
  int timeStart = jsonData.indexOf("\"time\":\"") + 8;
  int timeEnd = jsonData.indexOf("\"", timeStart);
  if (timeStart > 7 && timeEnd > timeStart) {
    String time = jsonData.substring(timeStart, timeEnd);
    DebugSerial.println("Time: " + time);
  }
  
  // Extract Latitude
  int latStart = jsonData.indexOf("\"latitude\":\"") + 12;
  int latEnd = jsonData.indexOf("\"", latStart);
  if (latStart > 11 && latEnd > latStart) {
    String latitude = jsonData.substring(latStart, latEnd);
    DebugSerial.println("Latitude: " + latitude);
  }
  
  // Extract Longitude
  int lonStart = jsonData.indexOf("\"longitude\":\"") + 13;
  int lonEnd = jsonData.indexOf("\"", lonStart);
  if (lonStart > 12 && lonEnd > lonStart) {
    String longitude = jsonData.substring(lonStart, lonEnd);
    DebugSerial.println("Longitude: " + longitude);
  }
  
  // Extract Confidence
  int confStart = jsonData.indexOf("\"confidence\":") + 13;
  int confEnd = jsonData.indexOf("}", confStart);
  if (confStart > 12 && confEnd > confStart) {
    String confidence = jsonData.substring(confStart, confEnd);
    DebugSerial.println("Confidence: " + confidence);
  }
  
  DebugSerial.println("==================");
}