import cv2

# Mở video file
video_path = 'D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_12/recorded_videos_fall/vid_fall_1695283790.mp4'
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Đếm số frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Tổng số frame trong video: {frame_count}")

# Đóng video
cap.release()
