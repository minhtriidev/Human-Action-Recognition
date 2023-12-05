import subprocess
import serial
import threading

# Cấu hình cổng UART trên Orange Pi
uart_port = '/dev/ttyS0'  # Thay đổi tùy thuộc vào Orange Pi của bạn
orange_pi_uart = serial.Serial(
    port=uart_port,
    baudrate=9600,
    timeout=1
)

wifi_name = ""
wifi_pass = ""
wifi_networks = ""

# Sử dụng Lock để tránh xung đột dữ liệu khi đọc và ghi wifi_name, wifi_pass, wifi_networks
data_lock = threading.Lock()

def process_uart_data(data):
    global wifi_name, wifi_pass, wifi_networks

    if data.startswith("ssid"):
        with data_lock:
            wifi_name = data.split(" ", 1)[1]
            print(f"WiFi SSID set to: {wifi_name}")

    elif data.startswith("pass"):
        with data_lock:
            wifi_pass = data.split(" ", 1)[1]
            print(f"WiFi Password set to: {wifi_pass}")

            # Scan mạng wifi
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

        # Kiểm tra kết nối Wi-Fi thành công
        if check_wifi_connection(ssid):
            success_message = f"Connected to Wi-Fi: {ssid}"
            print(success_message)
            # Phản hồi lại dữ liệu về ESP8266 qua UART
            orange_pi_uart.write(success_message.encode('utf-8'))
        else:
            error_message = f"Failed to connect to Wi-Fi: {ssid}"
            print(error_message)
            # Phản hồi lỗi về ESP8266 qua UART
            orange_pi_uart.write(error_message.encode('utf-8'))

    except subprocess.CalledProcessError as e:
        error_message = f"Error: {e}"
        print(error_message)
        # Phản hồi lỗi về ESP8266 qua UART
        orange_pi_uart.write(error_message.encode('utf-8'))

def check_wifi_connection(ssid):
    try:
        # Sử dụng lệnh nmcli để kiểm tra kết nối Wi-Fi
        result = subprocess.run(['nmcli', 'connection', 'show', '--active'], capture_output=True, text=True, check=True)
        return ssid in result.stdout
    except subprocess.CalledProcessError:
        return False

def uart_reader():
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

# Tạo và khởi chạy thread đọc và ghi UART
uart_thread = threading.Thread(target=uart_reader)
uart_thread.start()

# Chờ thread kết thúc (không bao gồm đọc và ghi UART)
uart_thread.join()
