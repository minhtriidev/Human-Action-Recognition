import subprocess
import serial
import cv2
import mediapipe as mp
from Model import ST_GCN
from mediapipe_function_4 import make_landmark_timestep, convert_landmarks_25
import os
import numpy as np
import torch
import asyncio
from telegram import Bot
import threading

# Variable to indicate if Wi-Fi is connected
wifi_connected = False

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

# Constants
C = 3  # Number of channels
V = 25  # Number of pose landmarks (nodes)
N = 1  # realtime dùng 1 chuỗi duy nhất và liên tục
T = 40  # Number of frames
data = np.zeros((N, C, T, V))

action_labels = ['cheer', 'clap', 'fall', 'jump', 'kick', 'run', 'sit', 'stand', 'walk', 'wave']

threshold = 0.8  # Set the probability threshold for action label display

# Load your model
model = ST_GCN(num_classes=10, in_channels=3, t_kernel_size=9, hop_size=2)
model.load_state_dict(torch.load('model/10action_25kp_100_64_0.01.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize camera
cap = cv2.VideoCapture(0)

# Create a Telegram bot instance
bot_token = "6702157131:AAHLopyJhuNij-Qkez5YySW7td__L31zzHM"
chat_id = "5607828483"
bot = Bot(token=bot_token)

# Variables
lm_list = []  # Pose landmarks list
captured_image_path = "picture_fall/captured_fall_image.jpg"

async def send_telegram_message_with_image():
    try:
        if os.path.exists(captured_image_path):
            with open(captured_image_path, "rb") as photo:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption="Fall action detected!")
        else:
            await bot.send_message(chat_id=chat_id, text="Fall action detected, but failed to capture an image.")
    except Exception as e:
        print(f"Error sending image: {str(e)}")

async def process_frames():
    global lm_list

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    while not wifi_connected:
        await asyncio.sleep(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width = frame.shape[:2]
        frame = frame[0:height, 0:int(width/2)].copy()
        frame = cv2.flip(frame, flipCode=1)

        ret, buffer = cv2.imencode('.jpg', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm = np.array(lm).reshape(-1, 4)
            kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)

            if len(lm_list) == T:
                regn_list = lm_list.copy()
                lm_list = lm_list[8:]
                pose_results = np.array(regn_list)
                pose_results = np.transpose(pose_results, (2, 0, 1))
                data[0, :, :, :] = pose_results

                input_data = torch.from_numpy(data).float()
                with torch.no_grad():
                    outputs = model(input_data)
                probabilities = torch.softmax(outputs, dim=1)

                predicted_label = torch.argmax(outputs, dim=1).item()
                current_label = action_labels[predicted_label]
                max_probability = probabilities[0, predicted_label].item()
                print(current_label)
                print(max_probability)

                if max_probability > threshold:
                    current_label = action_labels[predicted_label]
                    if current_label == "fall":
                        cv2.imwrite(captured_image_path, frame)
                        await asyncio.wait_for(send_telegram_message_with_image(), timeout=60)

                else:
                    print("UNKNOWN")

            else:
                lm_list.append(kinect_v2_lm[:, :3])

        else:
            print("No Landmarks Detected")

        await asyncio.sleep(0.01)

async def uart_reader():
    global wifi_connected, wifi_name, wifi_pass, wifi_networks

    try:
        while True:
            received_data = orange_pi_uart.readline().decode('utf-8').strip()

            if received_data:
                print(f"Received: {received_data}")
                if received_data.startswith("ssid"):
                    wifi_name = received_data.split(" ", 1)[1]
                    print(f"WiFi SSID set to: {wifi_name}")
                elif received_data.startswith("pass"):
                    wifi_pass = received_data.split(" ", 1)[1]
                    print(f"WiFi Password set to: {wifi_pass}")

                    wifi_networks = scan_wifi_networks()
                    print(f"{wifi_networks}")

                    connect_to_wifi(wifi_name, wifi_pass)

    except asyncio.CancelledError:
        print("UART communication canceled.")
    except Exception as e:
        print(f"Error in UART communication: {str(e)}")
    except KeyboardInterrupt:
        orange_pi_uart.close()
        print("UART communication closed.")

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
    global wifi_connected
    try:
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
        subprocess.run(command, check=True)

        if check_wifi_connection(ssid):
            success_message = f"Connected to Wi-Fi: {ssid}"
            print(success_message)
            wifi_connected = True
            orange_pi_uart.write(success_message.encode('utf-8'))
        else:
            error_message = f"Failed to connect to Wi-Fi: {ssid}"
            print(error_message)
            wifi_connected = False
            orange_pi_uart.write(error_message.encode('utf-8'))

    except subprocess.CalledProcessError as e:
        error_message = f"Error: {e}"
        print(error_message)
        wifi_connected = False
        orange_pi_uart.write(error_message.encode('utf-8'))

def check_wifi_connection(ssid):
    try:
        # Sử dụng lệnh nmcli để kiểm tra kết nối Wi-Fi
        result = subprocess.run(['nmcli', 'connection', 'show', '--active'], capture_output=True, text=True, check=True)
        return ssid in result.stdout
    except subprocess.CalledProcessError:
        return False

if __name__ == "__main__":
    uart_thread = threading.Thread(target=lambda: asyncio.run(uart_reader()))
    uart_thread.start()

    asyncio.run(process_frames())
