import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)

# def disconnect_wifi(interface):
#     try:
#         # Ngắt kết nối WiFi trên interface được chỉ định
#         subprocess.run(["sudo", "ifconfig", interface, "down"], check=True)
#         logging.info(f"Disconnected from {interface}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error disconnecting from {interface}: {e}")

def start_access_point(ssid, wpa_passphrase):
    try:

        hostapd_config = f"""
#

ssid={ssid}
interface=wlan0
hw_mode=g
channel=5
bridge=br0
driver=nl80211

logger_syslog=0
logger_syslog_level=0
wmm_enabled=1
wpa=2
preamble=1

wpa_psk=66eb31d2b48d19ba216f2e50c6831ee11be98e2fa3a8075e30b866f4a5ccda27
wpa_passphrase={wpa_passphrase}
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
auth_algs=1
macaddr_acl=0

### IEEE 802.11n
#ieee80211n=1
#ht_capab=
#country_code=US
#ieee80211d=1
### IEEE 802.11n

### IEEE 802.11a
#hw_mode=a
### IEEE 802.11a

### IEEE 802.11ac
#ieee80211ac=1
#vht_capab=[MAX-MPDU-11454][SHORT-GI-80][TX-STBC-2BY1][RX-STBC-1][MAX-A-MPDU-LEN-EXP3]
#vht_oper_chwidth=1
#vht_oper_centr_freq_seg0_idx=42
### IEEE 802.11ac

# controlling enabled
ctrl_interface=/var/run/hostapd
ctrl_interface_group=0
"""

        with open("/etc/hostapd.conf", "w") as file:
            file.write(hostapd_config)

        subprocess.run(["sudo", "systemctl", "stop", "hostapd"], capture_output=True, text=True).check_returncode()
        subprocess.run(["sudo", "systemctl", "start", "hostapd"], capture_output=True, text=True).check_returncode()

        # Khởi động dnsmasq
        dnsmasq_config = f"""
interface=wlan0            # Tên của interface Wi-Fi (giống với hostapd.conf)
listen-address=192.168.4.1  # Địa chỉ IP của Access Point
bind-interfaces            # Ràng buộc interfaces
server=8.8.8.8             # DNS server
domain-needed               # Nếu cần domain
bogus-priv                  # Chấp nhận các domain giả mạo
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h  # Phạm vi DHCP
"""

        with open("/etc/dnsmasq.conf", "w") as file:
            file.write(dnsmasq_config)

        # Thiết lập địa chỉ IP tĩnh cho wlan0
        subprocess.run(["sudo", "ifconfig", "wlan0", "192.168.4.1"]).check_returncode()

        subprocess.run(["sudo", "systemctl", "stop", "dnsmasq"], capture_output=True, text=True).check_returncode()
        subprocess.run(["sudo", "systemctl", "start", "dnsmasq"], capture_output=True, text=True).check_returncode()

        # Enable IP forwarding
        subprocess.run(["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"]).check_returncode()

        subprocess.run(["sudo", "systemctl", "restart", "hostapd"], capture_output=True, text=True).check_returncode()
        subprocess.run(["sudo", "systemctl", "restart", "dnsmasq"], capture_output=True, text=True).check_returncode()

        logging.info("Access Point has been successfully set up. Waiting for connections from other devices.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    ssid = "OrangePi"        # Tên của Access Point
    wpa_passphrase = "1122334455"   # Mật khẩu của Access Point

    start_access_point(ssid, wpa_passphrase)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Dừng Access Point.")
