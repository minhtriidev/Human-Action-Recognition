import cv2
import os

# Input folder containing video files
input_folder = 'D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/recorded_videos_walk'

# Output folder to save trimmed videos
output_folder = 'D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/output_cut_folder_walk'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
file_list = os.listdir(input_folder)

# Iterate through each file in the folder
for filename in file_list:
    if filename.endswith(".mp4"):  # Adjust the extension as needed
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        
        # Open the video file
        cap = cv2.VideoCapture(input_file)

        # Get the frame rate and frame count
        fps = cap.get(5)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame_number = 40
        end_frame_number = 85


        # Create VideoWriter object to save the trimmed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
        out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        current_frame_number = 0

        while current_frame_number < end_frame_number:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            if current_frame_number >= start_frame_number:
                out.write(frame)

            current_frame_number += 1
        # Release the video objects
        cap.release()
        out.release()

print("Trimming complete. Trimmed videos are saved in the output folder.")
