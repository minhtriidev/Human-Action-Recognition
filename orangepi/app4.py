import cv2
import io
from fastapi import FastAPI, Response, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse, HTMLResponse
import torch
import numpy as np
import mediapipe as mp
from Model import ST_GCN
from mediapipe_function_4 import make_landmark_timestep, convert_landmarks_25, plot_3d_landmarks
import matplotlib.pyplot as plt
import base64
import asyncio
import websockets
import json

# Khai báo thông tin model
model = ST_GCN(num_classes=9, in_channels=3, t_kernel_size=9, hop_size=2)
model.load_state_dict(torch.load('9action15_100_64_0.01.pth', map_location=torch.device('cpu')))
model.eval()

# Khai báo các hàm chức năng Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Khai báo các biến toàn cục
frame_count = 0
current_label = None
kinect_v2_landmarks = np.zeros((25, 4))
regn_list = []
# Constants
C = 3  # Number of channels
V = 25  # Number of pose landmarks (nodes)
N = 1  # realtime dùng 1 chuỗi duy nhất và liên tục
T = 40  # Number of frames
data = np.zeros((N, C, T, V))
action_labels = ['CLAP', 'FALL', 'JUMP', 'RUN', 'SIT', 'STAND', 'THROW', 'WALK', 'WAVE']
threshold = 0.7  # Set the probability threshold for action label display

received_stream_url = ""
app = FastAPI()

shared_state = {"current_label": ""}

vid_state = {"current_skeleton": ""}

skeleton_state = {"current_skeleton": ""}

skeleton_state_video = {"current_skeleton": ""}

image_state = {"current_image": ""}

# Danh sách lưu trữ các frame
captured_frames = []

from pydantic import BaseModel
# Danh sách lưu trữ các frame
captured_frames = []

class FrameData(BaseModel):
    frame_number: int
    data: str

class VideoData(BaseModel):
    video_url: str
    frames: list[FrameData]


# Load HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index4.html", {"request": request})

async def process_frames(video_url):
    global lm_list
    global current_label
    global captured_frames
    global frame_count
    frame_count = 0
    camera = cv2.VideoCapture(video_url)  # 0 represents the default camera
    while True:
        if video_url == 0:
            success, frame = cap.read()

            if not success:
                break
            height, width = frame.shape[:2]
            frame = frame[0:height, 0:int(width/2)].copy()
            frame=cv2.flip(frame,flipCode=1)
        else:
            success, frame = camera.read()

            if not success:
                break
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        # Phát hiện khung xương con người trong khung hình
        if results.pose_landmarks:

            # Xử lý pose landmarks và thêm vào lm_list
            lm = make_landmark_timestep(results)
            lm = np.array(lm).reshape(-1, 4)
            kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)
            skeleton_state["current_skeleton"] = kinect_v2_lm
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

                if max_probability > threshold:
                    current_label = action_labels[predicted_label]
                else:
                    current_label = "UNKNOWN"  # Display "Unknown" when probability is below the threshold

            else:
                lm_list.append(kinect_v2_lm[:, :3])

        # Khi không phát hiện khung xương người trong khung hình
        else:
            current_label = "No Landmarks Detected"
        shared_state["current_label"] = current_label
        await asyncio.sleep(0.01)  # Adjust the sleep interval as needed
        frame_bytes = io.BytesIO(buffer)
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.read() + b'\r\n')

global_label = None
async def process_frames1(video_url):
    global lm_list
    lm_list = []
    current_label = None
    global captured_frames
    global frame_count
    global global_label
    print(video_url)
    camera = cv2.VideoCapture(video_url)  # 0 represents the default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        # Phát hiện khung xương con người trong khung hình
        if results.pose_landmarks:

            # Xử lý pose landmarks và thêm vào lm_list
            lm = make_landmark_timestep(results)
            lm = np.array(lm).reshape(-1, 4)
            kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)
            skeleton_state["current_skeleton"] = kinect_v2_lm
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

                if max_probability > threshold:
                    current_label = action_labels[predicted_label]
                else:
                    current_label = "Unknown"  # Display "Unknown" when probability is below the threshold
                # print("hello", current_label)
            else:
                lm_list.append(kinect_v2_lm[:, :3])

        # Khi không phát hiện khung xương người trong khung hình
        else:
            current_label = "no landmarks detected"
        global_label = current_label
        # print(current_label)
        await asyncio.sleep(0.01)  # Adjust the sleep interval as needed

        
@app.post("/detect_action")
async def detect_action():
    print(shared_state["current_label"])
    return {"current_label": shared_state["current_label"]}

@app.post("/set_stream_url")
async def set_stream_url(data: dict):
    global received_stream_url
    received_stream_url = data["video_url"]
    return {"message": "Video URL received and stored"}

def generate_skeleton_image_base64():
    # Generate skeleton visualization using your existing code or methods
    kinect_v2_landmarks = skeleton_state["current_skeleton"]  # Replace with your method to get skeleton data
    elev = 0  # Desired elevation angle in degrees
    azim = 270  # Desired azimuth angle in degrees
    fig = plot_3d_landmarks(kinect_v2_landmarks, elev, azim)  # Assuming you have a function for 3D visualization

    # Convert the Matplotlib figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.read()
    
    # Convert the bytes to a Base64-encoded string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64

def generate_skeleton_video_image_base64():
    # Generate skeleton visualization using your existing code or methods
    kinect_v2_landmarks = skeleton_state["current_skeleton"]  # Replace with your method to get skeleton data
    elev = 0  # Desired elevation angle in degrees
    azim = 270  # Desired azimuth angle in degrees
    fig = plot_3d_landmarks(kinect_v2_landmarks, elev, azim)  # Assuming you have a function for 3D visualization
        
    # Convert the Matplotlib figure to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.read()
    
    # Convert the bytes to a Base64-encoded string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64

connected_clients_skeleton = set()

@app.websocket("/ws_skeleton_image")
async def ws_skeleton_image(websocket: WebSocket):
    global received_stream_url
    await websocket.accept()
    connected_clients_skeleton.add(websocket)
    try:
        while True:
            image_base64 = generate_skeleton_image_base64()
            await websocket.send_json({"image_base64": image_base64})
            if received_stream_url == 0:
                await asyncio.sleep(2)  # Adjust the delay as needed
            else:
                await asyncio.sleep(0.01)
    except websockets.exceptions.ConnectionClosed:
        connected_clients_skeleton.remove(websocket)
        

# Route to stream video feed
@app.get("/video_feed")
async def video_feed():
    global received_stream_url
    received_stream_url = 0
    return StreamingResponse(process_frames(received_stream_url), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture_frames")
async def capture_frames(data1: VideoData):
    video_url = data1.video_url
    asyncio.create_task(process_frames1(video_url))  # Start the frame processing in the background
    return {"message": "Frame capture started"}

# Define the WebSocket connection variable
websocket_connections = []

@app.websocket("/ws_img")
async def websocket_image_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        label_data = {
            "image": image_state["current_image"],
        }
        await websocket.send_text(json.dumps(label_data))
        await asyncio.sleep(0.01)  # Adjust the sleep interval as needed
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global global_label
    global frame_count
    await websocket.accept()
    connected_clients_skeleton.add(websocket)  # Store the WebSocket connection
    try:
        while True:    
            print("hello", global_label, frame_count)       
            await websocket.send_json({"label": global_label, "frame": frame_count})
            await asyncio.sleep(0.01)  # Adjust the sleep interval as needed
    except websockets.exceptions.ConnectionClosed:
        frame_count = 0
        connected_clients_skeleton.remove(websocket)
        
if __name__ == "__main__":
    import uvicorn
    global lm_list
    lm_list = []
    uvicorn.run(app, host="0.0.0.0", port=8000)
