import torch
import cv2
import numpy as np
import mediapipe as mp
from Model import ST_GCN
from mediapipe_function import convert_landmarks, make_landmark_timestep, convert_landmarks_25


# Khai báo thông tin model
model = ST_GCN(num_classes=9, in_channels=3, t_kernel_size=9, hop_size=2)
model.load_state_dict(torch.load('9action10_100_64_0.01.pth', map_location=torch.device('cpu')))
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
lm_list = []
kinect_v2_landmarks = np.zeros((25, 4))
regn_list = []
# Constants
C = 3  # Number of channels
V = 25  # Number of pose landmarks (nodes)
N = 1  # realtime dùng 1 chuỗi duy nhất và liên tục
T = 40  # Number of frames
data = np.zeros((N, C, T, V))
action_labels = ['clap', 'fall', 'jump', 'run', 'sit', 'stand', 'throw', 'walk', 'wave']
threshold = 0.8  # Set the probability threshold for action label display

# Vòng lặp chính thực thi nhận diện
while True:
    # Đọc từng khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi hình ảnh từ BGR sang RGB
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
            print(current_label)
            max_probability = probabilities[0, predicted_label].item()

            if max_probability > threshold:
                current_label = action_labels[predicted_label]
            else:
                current_label = "Unknown"  # Display "Unknown" when probability is below the threshold


        else:
            lm_list.append(kinect_v2_lm[:, :3])

        # Display the frame with label
        cv2.putText(frame, current_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Khi không phát hiện khung xương người trong khung hình
    else:
        print('no landmarks detected')

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
