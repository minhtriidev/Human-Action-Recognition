import subprocess
import serial
import time

# Cấu hình cổng UART trên Orange Pi
orange_pi_uart = serial.Serial(
    port='/dev/ttyS0',  # Thay đổi tùy thuộc vào Orange Pi của bạn
    baudrate=9600,
    timeout=1
)

wifi_name = ""
wifi_pass = ""

def process_uart_data(data):
    global wifi_name, wifi_pass

    if data.startswith("ssid"):
        wifi_name = data.split(" ", 1)[1]
        print(f"WiFi SSID set to: {wifi_name}")

    elif data.startswith("pass"):
        wifi_pass = data.split(" ", 1)[1]
        print(f"WiFi Password set to: {wifi_pass}")

        # scan mang wifi
        wifi_networks = scan_wifi_networks()
        print(f"{wifi_networks}")
        # Kết nối đến mạng Wi-Fi chỉ định
        connect_to_wifi(wifi_name, wifi_pass)


def scan_wifi_networks():
    try:
        # Sử dụng lệnh nmcli để quét các mạng Wi-Fi
        result = subprocess.run(['nmcli', 'device', 'wifi', 'list'], capture_output=True, text=True, check=True)
        networks = result.stdout
        return networks
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def connect_to_wifi(ssid, password):
    try:
        # Xây dựng lệnh để kết nối vào mạng Wi-Fi chỉ định
        command = [
            'sudo',
            'nmcli',
            'device',
            'wifi',
            'connect',
            ssid,
            'password',
            password
        ]

        # Gọi lệnh
        subprocess.run(command, check=True)

        print(f"Connected to Wi-Fi: {ssid}")

        # Phản hồi lại dữ liệu
        response_data = f"Connected to Wi-Fi: {ssid}"
        orange_pi_uart.write(response_data.encode('utf-8'))

    except subprocess.CalledProcessError as e:
        error_message = f"Error: {e}"
        print(error_message)
        # Phản hồi lỗi qua UART
        orange_pi_uart.write(error_message.encode('utf-8'))

try:
    while True:
        # Đọc dữ liệu từ UART
        received_data = orange_pi_uart.readline().decode('utf-8').strip()

        if received_data:
            print(f"Received: {received_data}")
            process_uart_data(received_data)

except KeyboardInterrupt:
    orange_pi_uart.close()
    print("UART communication closed.")
