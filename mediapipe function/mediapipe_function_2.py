import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('Agg')
def convert_landmarks(landmarks, scale_factor):
    kinect_v2_landmarks = np.zeros((26, 4))
    value = np.zeros((1, 4))
    mapping = {11: 4,13: 5, 15: 6,17: 7,12: 8,14: 9,16: 10, 18: 11,23: 12, 25: 13, 27: 14, 31: 15, 24: 16, 26: 17, 28: 18, 32: 19, 19: 21, 21: 22, 20: 23, 22: 24}

    for i, j in mapping.items():
        kinect_v2_landmarks[j, :] = landmarks[i, :]

    unmapped_indices = set(range(26)) - set(mapping.values())
    for index in unmapped_indices:
        if index == 0:
            kinect_v2_landmarks[0, :] = (landmarks[24, :] + landmarks[23, :]) / 2
        elif index == 1:
            kinect_v2_landmarks[1, :] = ((landmarks[11, :] + landmarks[12, :]) / 2 +
                                         (landmarks[24, :] + landmarks[23, :]) / 2) / 2
        elif index == 2:
            kinect_v2_landmarks[2, :] = ((landmarks[9, :] + landmarks[10, :]) / 2 + (landmarks[11, :] + landmarks[12, :]) / 2) / 2
        elif index == 3:
            kinect_v2_landmarks[3, :] = (landmarks[9, :] + landmarks[10, :]) / 2
        elif index == 20:
            kinect_v2_landmarks[20, :] = (landmarks[11, :] + landmarks[12, :]) / 2

    # second_joint is 0,0,0

    second_joint = kinect_v2_landmarks[1, :3]
    for i in range(kinect_v2_landmarks.shape[0]):
        if i != 1:
            kinect_v2_landmarks[i, :3] = kinect_v2_landmarks[i, :3] - second_joint

    kinect_v2_landmarks[1, :3] = [0,0,0]

    # Rotate the joints around the y-axis by 90 degrees
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    kinect_v2_landmarks[:, :3] = np.matmul(rotation_matrix, kinect_v2_landmarks[:, :3].T).T

    # Scale the joints
    kinect_v2_landmarks[:, :3] *= scale_factor

    kinect_v2_landmarks[25, :] = ((landmarks[11, :] + landmarks[12, :]) / 2 + (
                landmarks[24, :] + landmarks[23, :]) / 2) / 2

    kinect_v2_landmarks[kinect_v2_landmarks > 1] = 1
    kinect_v2_landmarks[kinect_v2_landmarks < -1] = -1

    return kinect_v2_landmarks


def convert_landmarks_normalize(landmarks, scale_factor):
    kinect_v2_landmarks = np.zeros((26, 4))
    value = np.zeros((1, 4))
    mapping = {11: 4,13: 5, 15: 6,17: 7,12: 8,14: 9,16: 10, 18: 11,23: 12, 25: 13, 27: 14, 31: 15, 24: 16, 26: 17, 28: 18, 32: 19, 19: 21, 21: 22, 20: 23, 22: 24}

    for i, j in mapping.items():
        kinect_v2_landmarks[j, :] = landmarks[i, :]

    unmapped_indices = set(range(26)) - set(mapping.values())
    for index in unmapped_indices:
        if index == 0:
            kinect_v2_landmarks[0, :] = (landmarks[24, :] + landmarks[23, :]) / 2
        elif index == 1:
            kinect_v2_landmarks[1, :] = ((landmarks[11, :] + landmarks[12, :]) / 2 +
                                         (landmarks[24, :] + landmarks[23, :]) / 2) / 2
        elif index == 2:
            kinect_v2_landmarks[2, :] = ((landmarks[9, :] + landmarks[10, :]) / 2 + (landmarks[11, :] + landmarks[12, :]) / 2) / 2
        elif index == 3:
            kinect_v2_landmarks[3, :] = (landmarks[9, :] + landmarks[10, :]) / 2
        elif index == 20:
            kinect_v2_landmarks[20, :] = (landmarks[11, :] + landmarks[12, :]) / 2
        elif index == 25:
            value[:,:] = ((landmarks[11, :] + landmarks[12, :]) / 2 + (landmarks[24, :] + landmarks[23, :]) / 2) / 2

    # second_joint is 0,0,0

    second_joint = kinect_v2_landmarks[1, :3]
    for i in range(kinect_v2_landmarks.shape[0]):
        if i != 1:
            kinect_v2_landmarks[i, :3] = kinect_v2_landmarks[i, :3] - second_joint

    kinect_v2_landmarks[1, :3] = [0,0,0]

    # First, find the direction vectors of the axes
    oz_axis = kinect_v2_landmarks[1, :3] - kinect_v2_landmarks[0, :3]
    oz_axis /= np.linalg.norm(oz_axis)
    ox_axis = kinect_v2_landmarks[8, :3] - kinect_v2_landmarks[4, :3]
    ox_axis /= np.linalg.norm(ox_axis)

    # Calculate the rotation matrix
    rotation_matrix = np.array([ox_axis, np.cross(ox_axis, oz_axis), oz_axis])

    # Apply the rotation to all joints except the second joint (index 1)
    kinect_v2_landmarks[:, :3] = np.dot(kinect_v2_landmarks[:, :3], rotation_matrix.T)

    # Scale the joints
    kinect_v2_landmarks[:, :3] *= scale_factor
    kinect_v2_landmarks[25, :] = value[:,:]

    return kinect_v2_landmarks

def convert_landmarks_25(landmarks, scale_factor):
    kinect_v2_landmarks = np.zeros((25, 4))
    value = np.zeros((1, 4))
    mapping = {11: 4,13: 5, 15: 6,17: 7,12: 8,14: 9,16: 10, 18: 11,23: 12, 25: 13, 27: 14, 31: 15, 24: 16, 26: 17, 28: 18, 32: 19, 19: 21, 21: 22, 20: 23, 22: 24}

    for i, j in mapping.items():
        kinect_v2_landmarks[j, :] = landmarks[i, :]

    unmapped_indices = set(range(25)) - set(mapping.values())
    for index in unmapped_indices:
        if index == 0:
            kinect_v2_landmarks[0, :] = (landmarks[24, :] + landmarks[23, :]) / 2
        elif index == 1:
            kinect_v2_landmarks[1, :] = ((landmarks[11, :] + landmarks[12, :]) / 2 +
                                         (landmarks[24, :] + landmarks[23, :]) / 2) / 2
        elif index == 2:
            kinect_v2_landmarks[2, :] = ((landmarks[9, :] + landmarks[10, :]) / 2 + (landmarks[11, :] + landmarks[12, :]) / 2) / 2
        elif index == 3:
            kinect_v2_landmarks[3, :] = (landmarks[9, :] + landmarks[10, :]) / 2
        elif index == 20:
            kinect_v2_landmarks[20, :] = (landmarks[11, :] + landmarks[12, :]) / 2

    # second_joint is 0,0,0

    second_joint = kinect_v2_landmarks[1, :3]
    for i in range(kinect_v2_landmarks.shape[0]):
        if i != 1:
            kinect_v2_landmarks[i, :3] = kinect_v2_landmarks[i, :3] - second_joint

    kinect_v2_landmarks[1, :3] = [0,0,0]

    # Rotate the joints around the y-axis by 90 degrees
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    kinect_v2_landmarks[:, :3] = np.matmul(rotation_matrix, kinect_v2_landmarks[:, :3].T).T

    # Scale the joints
    kinect_v2_landmarks[:, :3] *= scale_factor

    kinect_v2_landmarks[kinect_v2_landmarks > 1] = 1
    kinect_v2_landmarks[kinect_v2_landmarks < -1] = -1

    return kinect_v2_landmarks

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_world_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)

    return c_lm

from PIL import Image, ImageDraw
import os
from io import BytesIO
import io
import base64
import asyncio
def plot_3d_landmarks(kinect_v2_landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # Plot keypoints
    for i, lm in enumerate(kinect_v2_landmarks):
        if lm[3] > 0.1:  # Only plot visible keypoints
            x, y, z = lm[0], lm[1], lm[2]
            if i == 15 or i == 19 or i == 3:
                ax.text(x, y, z, f'({x:.2f}, {y:.2f}, {z:.2f})', fontsize=8, color='k')  # Add the label with the index 'i'
            # ax.text(x, y, z, f'({x:.2f}, {y:.2f}, {z:.2f})', fontsize=8, color='k')  # Add the label with the index 'i'
            # ax.text(x, y, z, f'({i})', fontsize=8, color='k')  # Add the label with the index 'i'
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
        if kinect_v2_landmarks[index_1][3] > 0.1 and kinect_v2_landmarks[index_2][3] > 0.1:
            x1, y1, z1 = kinect_v2_landmarks[index_1][0], kinect_v2_landmarks[index_1][1], kinect_v2_landmarks[index_1][2]
            x2, y2, z2 = kinect_v2_landmarks[index_2][0], kinect_v2_landmarks[index_2][1], kinect_v2_landmarks[index_2][2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')
    
    ax.set_xlabel('Ox')
    ax.set_ylabel('Oy')
    ax.set_zlabel('Oz')
    
        
    # Save the plot to a base64 encoded PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('UTF-8')

    plt.close(fig)

    return base64_image

output_folder = "output_folder" 
