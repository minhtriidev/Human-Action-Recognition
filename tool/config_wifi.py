import subprocess

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
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# Quét và hiển thị danh sách các mạng Wi-Fi
wifi_networks = scan_wifi_networks()
if wifi_networks:
    print("Available Wi-Fi Networks:")
    print(wifi_networks)

    # Thay thế bằng tên mạng Wi-Fi và mật khẩu thực tế của bạn
    target_ssid = 'tri'
    target_password = '12345678'

    # Kết nối vào mạng Wi-Fi chỉ định
    connect_to_wifi(target_ssid, target_password)
else:
    print("Failed to scan Wi-Fi networks.")
