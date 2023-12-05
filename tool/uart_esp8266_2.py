import serial
import time

# Cấu hình cổng UART trên Orange Pi
orange_pi_uart = serial.Serial(
    port='/dev/ttyS0',  # Thay đổi tùy thuộc vào Orange Pi của bạn
    baudrate=9600,
    timeout=1
)

try:
    while True:
        # Đọc dữ liệu từ UART
        received_data = orange_pi_uart.readline().decode('utf-8').strip()
        
        if received_data:
            print(f"Received: {received_data}")

            # Phản hồi lại dữ liệu
            response_data = "Hello from Orange Pi!"
            orange_pi_uart.write(response_data.encode('utf-8'))
            time.sleep(1)  # Đợi để tránh việc gửi quá nhanh
except KeyboardInterrupt:
    orange_pi_uart.close()
    print("UART communication closed.")
