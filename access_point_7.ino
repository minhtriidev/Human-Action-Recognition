#include <SoftwareSerial.h>
#include <ESP8266WiFi.h>

SoftwareSerial esp8266Serial(13, 15); // RX, TX

const char *ssid = "ESP8266-AP";
const char *password = "12345678";
WiFiServer server(80);

WiFiClient client;

const unsigned long waitTimeNoClient = 5 * 60 * 1000;  // 10 phút (5 * 60 * 1000 milliseconds)
const unsigned long waitTimeAfterClientDisconnect = 5 * 60 * 1000;  // 5 phút
unsigned long lastClientDisconnectTime = 0;

void setup() {
  Serial.begin(9600);
  esp8266Serial.begin(9600);

  WiFi.softAP(ssid, password);
  server.begin();

  Serial.print("Access Point IP Address: ");
  Serial.println(WiFi.softAPIP());
}

void loop() {
  client = server.available();

  if (client) {

    while (client.connected()) {
      if (client.available()) {
        String request = client.readStringUntil('\r');
        Serial.println(request);
        esp8266Serial.println(request);
      }

      if (esp8266Serial.available() > 0) {
        String received_data = esp8266Serial.readStringUntil('\n');
        Serial.print("Received from Orange Pi: ");
        Serial.println(received_data);

        // Gửi lại dữ liệu nhận được từ Orange Pi cho thiết bị kết nối
        client.println(received_data);
      }
    }

    client.stop();
    Serial.println("Client disconnected");
    lastClientDisconnectTime = millis();  // Lưu thời điểm client ngắt kết nối
  }

  unsigned long currentTime = millis();
  unsigned long elapsedTime = currentTime - lastClientDisconnectTime;

  // Kiểm tra xem đã đủ thời gian chờ sau khi client ngắt kết nối chưa
  if (elapsedTime > waitTimeAfterClientDisconnect && !client.connected()) {
    turnOffAccessPoint();
  }

  // Kiểm tra xem đã đủ thời gian chờ sau khi không có client kết nối chưa
  if (elapsedTime > waitTimeNoClient && !client.connected()) {
    turnOffAccessPoint();
  }

  delay(1000);
}

void turnOffAccessPoint() {
  Serial.println("Turning off Access Point");
  WiFi.softAPdisconnect(true);
}
