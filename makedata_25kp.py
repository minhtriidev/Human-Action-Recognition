import os
import os.path as osp
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import glob
from mediapipe_function_4 import convert_landmarks_25, make_landmark_timestep


# Khai báo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Khai báo các hàm chức năng detect và lấy keypoint
def extract_frame(video_path):
    dname = 'temp'
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths


def pose_inference(video_file, frame_paths):
    lm_list = []
    print('Performing Human Pose Estimation for each frame')
    cap = cv2.VideoCapture(video_file)

    for frame_path in tqdm(frame_paths):
        ret, frame = cap.read()
        if not ret:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm = np.array(lm).reshape(-1, 4)
            kinect_v2_lm = convert_landmarks_25(lm, scale_factor=1)
            lm_list.append(kinect_v2_lm[:, :3])
    return lm_list


def ntu_pose_extraction(vid, label):
    frame_paths = extract_frame(vid)
    pose_results = pose_inference(vid, frame_paths)
    pose_results = np.array(pose_results)  # Convert lm_list to a NumPy array
    pose_results = np.transpose(pose_results, (2, 0, 1))  # Transpose dimensions (C, T, V)
    shutil.rmtree(osp.dirname(frame_paths[0]))
    return pose_results


# Constants
C = 3  # Number of channels
T = 40  # Number of frames per video
V = 25  # Number of pose landmarks (nodes)

# Create a dictionary to store video files for each action
video_files_dict = {
    'action0': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/clap/*.mp4'), key=os.path.getmtime),
    'action1': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/fall/*.mp4'), key=os.path.getmtime),
    'action2': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/jump/*.mp4'), key=os.path.getmtime),
    'action3': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/run/*.mp4'), key=os.path.getmtime),
    'action4': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/sit/*.mp4'), key=os.path.getmtime),
    'action5': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/stand/*.mp4'), key=os.path.getmtime),
    'action6': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/throw/*.mp4'), key=os.path.getmtime),
    'action7': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/walk/*.mp4'), key=os.path.getmtime),
    'action8': sorted(glob.glob('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/test_real/wave/*.mp4'), key=os.path.getmtime),
}

# Calculate the total number of videos and assign labels
N = sum(len(video_files) for video_files in video_files_dict.values())
data = np.zeros((N, C, T, V))
label_index = 0  # Initialize label index
video_names = []

for action, video_files in video_files_dict.items():
    num_videos = len(video_files)
    label = np.full((num_videos,), label_index)  # Create an array of labels for the current action

    for index, file in enumerate(video_files):
        print('Processing ' + file)
        video_names.append(file)
        pose_results = ntu_pose_extraction(file, label=label[index])  # Assign label to the pose extraction function

        # Adjust the number of frames to match T
        num_frames = pose_results.shape[1]
        if num_frames > T:
            # Truncate frames to T
            pose_results = pose_results[:, :T, :]
        elif num_frames < T:
            # Pad with the last frame to fill T frames
            last_frame = pose_results[:, -1, :].reshape((pose_results.shape[0], 1, pose_results.shape[2]))
            num_padding = T - num_frames
            pose_results = np.concatenate((pose_results, np.tile(last_frame, (1, num_padding, 1))), axis=1)

        # Assign pose_results to data starting from the appropriate index
        start_index = label_index * (N // len(video_files_dict))

        data[start_index + index, :, :, :] = pose_results

    label_index += 1  # Increment the label index for the next action


# Create test_data.npy
np.save('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/train_data_9action_16.npy', data)

# Create test_label.npy
label = np.repeat(np.arange(len(video_files_dict)), N // len(video_files_dict))
np.save('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/train_label_9action_16.npy', label)

# np.save('D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/video_names_test_9action_16.npy', video_names)
