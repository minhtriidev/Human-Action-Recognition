import cv2
import time
import os

# Create the output folder if it doesn't exist
output_folder = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_16/recorded_videos_walk"
os.makedirs(output_folder, exist_ok=True)

# Define the video capture device (0 for default camera)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_color = (0, 255, 0)  # Green color in BGR
font_thickness = 5
text_position = (80, 120)
# Set the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
output_filename = os.path.join(output_folder, "output.mp4")
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

# Record a new video every 3 seconds
record_interval = 3  # in seconds
retry_counter = 0
start_time = time.time()
counter = 0 
update_timer = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Write the current frame to the video file
    out.write(frame)

    # Check if it's time to start a new video
    if time.time() - start_time >= record_interval:
        counter += 1
        start_time = time.time()
        out.release()  # Release the current video file
        output_filename = os.path.join(output_folder, f"vid_walk_{int(start_time)}.mp4")  # Generate a new filename
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (640, 480))
        retry_counter = 0
    if time.time() - update_timer >= 1:
        update_timer = time.time()
        retry_counter += 1
    cv2.putText(frame, str(retry_counter), text_position, font, font_scale, font_color, font_thickness)
    cv2.imshow('Recording', frame)
    # Press 'q' to stop the recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
