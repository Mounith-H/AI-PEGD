#include <Arduino.h>
#include <SPI.h>
#include "printf.h"
#include "RF24.h"

RF24 radio(PA4, PA3);  // using pin PA4 for the CE pin, and pin PA3 for the CSN pin
// IRQ pin is connected to PA2 but not used in this basic example

// Data reconstruction variables
String completeMessage = "";
unsigned long lastPacketTime = 0;
const unsigned long PACKET_TIMEOUT = 5000; // 5 seconds timeout between packets
int fragmentCount = 0;
int expectedFragments = 0; // Will be set from header
int currentMessageId = -1;

void parseAndDisplayData(String jsonData);

void setup() {
  Serial.begin(115200);
  
  pinMode(PC13, OUTPUT); // LED pin for indication
  
  // Initialize radio
  if (!radio.begin()) {
    Serial.println("Radio hardware not responding!");
    while(1); // Hold in infinite loop
  }

  // Configuration settings.
  radio.setChannel(2);
  radio.setPayloadSize(10);
  radio.setDataRate(RF24_2MBPS); // 2 Mbps
  radio.setPALevel(RF24_PA_LOW); // Set power level
  radio.openReadingPipe(0, (uint8_t*)"gyroc");  // using pipe 0
  radio.setAutoAck(false); // disable AutoAck.
  // END Configuration settings.
  
  radio.startListening();  // put radio in RX mode
  delay(1000); // Allow some time for the radio to initialize
  
  Serial.println("NRF24L01 Receiver Test");
  Serial.println("Pin Configuration:");
  Serial.println("CE  -> PA4");
  Serial.println("CSN -> PA3"); 
  Serial.println("SCK -> PA5");
  Serial.println("MOSI-> PA7");
  Serial.println("MISO-> PA6");
  Serial.println("IRQ -> PA2 (not used)");
  Serial.println("Listening for data...");
  Serial.println("Expecting messages with 4-byte header + 6-byte data format");
}  // setup

void loop() {
  static uint8_t buffer[11]; // Increased buffer size to account for null terminator
  static char msg[150] = {0}; // Increased message buffer
  
  if (radio.available()) {
      uint8_t len = radio.getDynamicPayloadSize(); // Get payload size
      if (len >= 4 && len <= 10) { // Validate payload size (at least header)
        radio.read(buffer, len);  // Read message into buffer.
        
        // Parse header: [msg_id][fragment_num][total_fragments][data_length]
        uint8_t messageId = buffer[0];
        uint8_t fragmentNum = buffer[1];
        uint8_t totalFragments = buffer[2];
        uint8_t dataLength = buffer[3];
        
        // Extract data portion (after 4-byte header)
        char dataBuffer[7] = {0}; // Max 6 bytes data + null terminator
        if (dataLength > 0 && dataLength <= 6 && (4 + dataLength) <= len) {
          memcpy(dataBuffer, &buffer[4], dataLength);
          dataBuffer[dataLength] = '\0';
        }
        
        // Show detailed packet info
        sprintf(msg, "Msg:%d Frag:%d/%d DataLen:%d Data:'%s'", 
                messageId, fragmentNum + 1, totalFragments, dataLength, dataBuffer);
        Serial.println(msg);
        
        // Check if this is a new message
        if (currentMessageId != messageId) {
          if (completeMessage.length() > 0) {
            Serial.println("New message started - clearing previous incomplete message");
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
          Serial.println("\n=== COMPLETE MESSAGE RECEIVED ===");
          Serial.println("Message ID: " + String(messageId));
          Serial.println("Total fragments: " + String(fragmentCount));
          Serial.println("Complete message: " + completeMessage);
          
          // Try to parse as JSON
          if (completeMessage.indexOf('{') != -1 && completeMessage.indexOf('}') != -1) {
            int startPos = completeMessage.indexOf('{');
            int endPos = completeMessage.lastIndexOf('}') + 1;
            
            if (startPos != -1 && endPos > startPos) {
              String jsonMessage = completeMessage.substring(startPos, endPos);
              parseAndDisplayData(jsonMessage);
            }
          } else {
            Serial.println("Message doesn't appear to be JSON format");
          }
          
          Serial.println("=================================\n");
          
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
    Serial.println("Message timeout - clearing incomplete data");
    Serial.println("Message ID: " + String(currentMessageId));
    Serial.println("Fragments received before timeout: " + String(fragmentCount) + "/" + String(expectedFragments));
    Serial.println("Partial message: " + completeMessage);
    completeMessage = "";
    fragmentCount = 0;
    currentMessageId = -1;
    expectedFragments = 0;
  }
  
  // Add a small delay to prevent overwhelming the serial output
  delay(10);
}

void parseAndDisplayData(String jsonData) {
  Serial.println("=== PARSED DATA ===");
  
  // Extract ID
  int idStart = jsonData.indexOf("\"_id\":\"") + 7;
  int idEnd = jsonData.indexOf("\"", idStart);
  if (idStart > 6 && idEnd > idStart) {
    String id = jsonData.substring(idStart, idEnd);
    Serial.println("ID: " + id);
  }
  
  // Extract Date
  int dateStart = jsonData.indexOf("\"date\":\"") + 8;
  int dateEnd = jsonData.indexOf("\"", dateStart);
  if (dateStart > 7 && dateEnd > dateStart) {
    String date = jsonData.substring(dateStart, dateEnd);
    Serial.println("Date: " + date);
  }
  
  // Extract Time
  int timeStart = jsonData.indexOf("\"time\":\"") + 8;
  int timeEnd = jsonData.indexOf("\"", timeStart);
  if (timeStart > 7 && timeEnd > timeStart) {
    String time = jsonData.substring(timeStart, timeEnd);
    Serial.println("Time: " + time);
  }
  
  // Extract Latitude
  int latStart = jsonData.indexOf("\"latitude\":\"") + 12;
  int latEnd = jsonData.indexOf("\"", latStart);
  if (latStart > 11 && latEnd > latStart) {
    String latitude = jsonData.substring(latStart, latEnd);
    Serial.println("Latitude: " + latitude);
  }
  
  // Extract Longitude
  int lonStart = jsonData.indexOf("\"longitude\":\"") + 13;
  int lonEnd = jsonData.indexOf("\"", lonStart);
  if (lonStart > 12 && lonEnd > lonStart) {
    String longitude = jsonData.substring(lonStart, lonEnd);
    Serial.println("Longitude: " + longitude);
  }
  
  // Extract Confidence
  int confStart = jsonData.indexOf("\"confidence\":") + 13;
  int confEnd = jsonData.indexOf("}", confStart);
  if (confStart > 12 && confEnd > confStart) {
    String confidence = jsonData.substring(confStart, confEnd);
    Serial.println("Confidence: " + confidence);
  }
  
  Serial.println("==================");
}