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
import threading
import requests
import json

from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext

os.chdir("/home/orangepi/Documents/Human-Action-Recognition")

# Load config
def load_config():
    try:
        with open('/home/orangepi/Documents/Human-Action-Recognition/config.json', 'r') as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        return None

def save_config(config):
    with open('/home/orangepi/Documents/Human-Action-Recognition/config.json', 'w') as file:
        json.dump(config, file, indent=2)

# Initialize config
config = load_config()

if not config:
    # Default config if not exists
    config = {
        "telegram": {
            "token": "",
            "chat_id": ""
        },
        "wifi": {
            "ssid": "",
            "password": ""
        }
    }

# Update variables
telegram_token = config['telegram']['token']
telegram_chat_id = config['telegram']['chat_id']
wifi_name = config['wifi']['ssid']
wifi_pass = config['wifi']['password']


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
model.load_state_dict(torch.load('/home/orangepi/Documents/Human-Action-Recognition/model/10action_25kp_100_64_0.01.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize camera
cap = cv2.VideoCapture(0)

# Variables
lm_list = []  # Pose landmarks list
captured_image_path = "/home/orangepi/Documents/Human-Action-Recognition/picture_fall/captured_fall_image.jpg"

async def send_telegram_message_with_image(token, chat_id):
    try:
        bot = Bot(token=token)
        
        if os.path.exists(captured_image_path):
            with open(captured_image_path, "rb") as photo:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption="Fall action detected!")
        else:
            await bot.send_message(chat_id=chat_id, text="Fall action detected, but failed to capture an image.")
    except Exception as e:
        print(f"Error sending image: {str(e)}")

async def process_frames():
    global lm_list, telegram_token, telegram_chat_id, wifi_name

    # Initialize pose object outside the loop
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

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
                # print(current_label)
                # print(max_probability)

                if max_probability > threshold:
                    current_label = action_labels[predicted_label]
                    if current_label == "fall":
                        cv2.imwrite(captured_image_path, frame)
                        await asyncio.wait_for(send_telegram_message_with_image(telegram_token, telegram_chat_id), timeout=60)

                else:
                    print("UNKNOWN")

            else:
                lm_list.append(kinect_v2_lm[:, :3])

            await asyncio.sleep(0.01)  # Bạn có thể điều chỉnh thời gian nghỉ tại đây nếu cần

        else:
            print("No Landmarks Detected")

async def uart_reader():
    global wifi_name, wifi_pass, telegram_token, telegram_chat_id

    try:
        wifi_connected = False

        while True:
            received_data = orange_pi_uart.readline().decode('utf-8').strip()

            if received_data:
                print(f"Received: {received_data}")
                if received_data.startswith("ssid"):
                    wifi_name = received_data.split(" ", 1)[1]
                    config['wifi']['ssid'] = wifi_name
                    save_config(config)
                    ssdi_message = f"\nWiFi SSID set to: {wifi_name}"
                    orange_pi_uart.write(ssdi_message.encode('utf-8'))
                    orange_pi_uart.flush()

                elif received_data.startswith("pass"):
                    wifi_pass = received_data.split(" ", 1)[1]
                    config['wifi']['password'] = wifi_pass
                    save_config(config)
                    pass_message = f"\nWiFi Password set to: {wifi_pass}"
                    orange_pi_uart.write(pass_message.encode('utf-8'))
                    orange_pi_uart.flush()

                    wifi_connected = connect_to_wifi(wifi_name, wifi_pass)
                    if wifi_connected:
                        success_message = f"\nConnected to Wi-Fi: {wifi_name}"
                        print(success_message)
                        orange_pi_uart.write(success_message.encode('utf-8'))
                        orange_pi_uart.flush()

                    else:
                        error_message = f"\nError: Failed to connect to Wi-Fi: {wifi_name}"
                        print(error_message)
                        orange_pi_uart.write(error_message.encode('utf-8'))
                        orange_pi_uart.flush()

                elif received_data.startswith("token"):
                    telegram_token = received_data.split(" ", 1)[1]
                    config['telegram']['token'] = telegram_token
                    save_config(config)
                    token_message = f"\nTelegram Token set to: {telegram_token}"
                    orange_pi_uart.write(token_message.encode('utf-8'))
                    orange_pi_uart.flush()

                elif received_data.startswith("id"):
                    telegram_chat_id = received_data.split(" ", 1)[1]
                    config['telegram']['chat_id'] = telegram_chat_id
                    save_config(config)
                    id_message = f"\nTelegram Chat ID set to: {telegram_chat_id}"
                    orange_pi_uart.write(id_message.encode('utf-8'))
                    orange_pi_uart.flush()
                    await check_token_and_chat(telegram_token, telegram_chat_id)

                else:
                    hello_message = "\nHello client!"
                    orange_pi_uart.write(hello_message.encode('utf-8'))
                    orange_pi_uart.flush()

    except asyncio.CancelledError:
        print("UART communication canceled.")
    except Exception as e:
        print(f"Error in UART communication: {str(e)}")
    except KeyboardInterrupt:
        orange_pi_uart.close()
        print("UART communication closed.")

async def check_token_and_chat(telegram_token, telegram_chat_id):
    try:
        # Kiểm tra token và ID chat
        bot = Bot(token=telegram_token)
        chat = await bot.get_chat(chat_id=telegram_chat_id)

        success_message = "\nToken and chat ID are valid and active."
        print(success_message)
        orange_pi_uart.write(success_message.encode('utf-8'))
        orange_pi_uart.flush()

        # print(f"Token and chat ID are valid and active.")
        # print(f"Bot name: {await bot.get_me().username}")
        # print(f"Chat title: {chat.title}")
        return True

    except Exception as e:
        error_message = f"\nError: {e}"
        print(error_message)
        orange_pi_uart.write(error_message.encode('utf-8'))
        orange_pi_uart.flush()
        print(f"Error: {e}")
        # print(f"Token or chat ID might be invalid or the bot is not a member of the channel.")
        return False

def scan_wifi_networks():
    try:
        # Sử dụng lệnh nmcli để quét các mạng Wi-Fi
        result = subprocess.run(['nmcli', 'device', 'wifi', 'list'], capture_output=True, text=True, check=True)
        networks = result.stdout
        print(networks)
        return networks
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def connect_to_wifi(ssid, password):
    try:
        scan_wifi_networks()
        command = [
            # 'sudo',
            'nmcli',
            'device',
            'wifi',
            'connect',
            ssid,
            'password',
            password
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False


if __name__ == "__main__":
    uart_thread = threading.Thread(target=lambda: asyncio.run(uart_reader()))
    uart_thread.start()

    asyncio.run(process_frames())
