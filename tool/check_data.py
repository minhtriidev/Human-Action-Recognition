import numpy as np


test_data = np.load('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/25kp/train_data25_9action_6.npy')
print(test_data.shape)
print(test_data[0, 0, 3, :])


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe_function import convert_landmarks_25, make_landmark_timestep


def plot_3d_landmarks(kinect_v2_landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # Plot keypoints
    for i, lm in enumerate(kinect_v2_landmarks):
#        if lm[3] > 0.5:  # Only plot visible keypoints
        x, y, z = lm[0], lm[1], lm[2]
        ax.scatter(x, y, z, color='r')

    # Plot connections
    connections = [
        (0, 1), (1, 20), (20, 2),
        (2, 3), (20, 4), (4, 5),
        (5, 6), (6, 7), (7, 21),
        (6, 22), (20, 8), (8, 9),
        (9, 10), (10, 11), (11, 23),
        (10, 24), (0, 12), (12, 13),
        (13, 14), (14, 15), (0, 16),
        (16, 17), (17, 18), (18, 19),
        # Add more connections here
    ]

    for connection in connections:
        index_1, index_2 = connection
        x1, y1, z1 = kinect_v2_landmarks[index_1][0], kinect_v2_landmarks[index_1][1], kinect_v2_landmarks[index_1][2]
        x2, y2, z2 = kinect_v2_landmarks[index_2][0], kinect_v2_landmarks[index_2][1], kinect_v2_landmarks[index_2][2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')

    ax.set_xlabel('Ox')
    ax.set_ylabel('Oy')
    ax.set_zlabel('Oz')
    plt.show()
    return fig


# Create a directory to store the image frames with landmarks
output_folder = 'output_frames'
os.makedirs(output_folder, exist_ok=True)

video_file = 'D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/test/clap/clap_1.mp4'
cap = cv2.VideoCapture(video_file)

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode = True, smooth_landmarks = True)
mpDraw = mp.solutions.drawing_utils

lm_list = []
no_of_frames = 300


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()

    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        lm = make_landmark_timestep(results)

        lm = np.array(lm).reshape(-1, 4)

        kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)
        lm_list.append(kinect_v2_lm[:,:3])

        fig = plot_3d_landmarks(kinect_v2_lm)
        plt.savefig(os.path.join(output_folder, f"plot_{len(lm_list)}.png"))
        plt.close(fig)
    else:
        # Append a zero-filled frame if no landmarks detected
        lm_list.append(np.zeros((25, 3)))