import cv2
import io
import mediapipe as mp
from Model import ST_GCN
from mediapipe_function_4 import make_landmark_timestep, convert_landmarks_25
import base64
import os
import numpy as np
import torch
import asyncio
from telegram import Bot

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
frame_count = 0
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
    global current_label
    global captured_frames

    while True:

        success, frame = cap.read()
        if not success:
            break

        # uncomment to use on webcam
        height, width = frame.shape[:2]
        frame = frame[0:height, 0:int(width/2)].copy()
        frame=cv2.flip(frame,flipCode=1)

        ret, buffer = cv2.imencode('.jpg', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        # Phát hiện khung xương con người trong khung hình
        if results.pose_landmarks:

            # Xử lý pose landmarks và thêm vào lm_list
            lm = make_landmark_timestep(results)
            lm = np.array(lm).reshape(-1, 4)
            kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)
            # Remove the oldest frame and append the new frame
            if len(lm_list) == T:
                regn_list = lm_list.copy()
                lm_list = lm_list[8:]
                # This should return 'pose_results' in the shape (C, T, V)
                pose_results = np.array(regn_list)  # Convert lm_list to a NumPy array
                pose_results = np.transpose(pose_results, (2, 0, 1))  # Transpose dimensions (C, T, V)
                data[0, :, :, :] = pose_results

                input_data = torch.from_numpy(data).float()
                with torch.no_grad():
                    outputs = model(input_data)
                probabilities = torch.softmax(outputs, dim=1)
                # Get the predicted label and its corresponding probability
                predicted_label = torch.argmax(outputs, dim=1).item()
                current_label = action_labels[predicted_label]
                max_probability = probabilities[0, predicted_label].item()
                print(current_label)
                print(max_probability)

                if max_probability > threshold:
                    current_label = action_labels[predicted_label]
                    if current_label == "fall":
                        cv2.imwrite(captured_image_path, frame)
                        # Use asyncio.wait_for to add a timeout
                        await asyncio.wait_for(send_telegram_message_with_image(), timeout=60)

                else:
                    print("UNKNOWN")

            else:
                lm_list.append(kinect_v2_lm[:, :3])

        # Khi không phát hiện khung xương người trong khung hình
        else:
            print("No Landmarks Detected")

        await asyncio.sleep(0.01)  # Adjust the sleep interval as needed


if __name__ == "__main__":
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    asyncio.run(process_frames())
