import cv2
import time

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thời điểm bắt đầu
start_time = time.time()

# Số khung hình đã xử lý trong khoảng thời gian
frame_count = 0

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Tăng số lượng khung hình đã xử lý
    frame_count += 1

    # Hiển thị khung hình
    cv2.imshow('Webcam', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tính thời gian đã trôi qua
end_time = time.time()
elapsed_time = end_time - start_time

# Tính tốc độ khung hình (fps)
fps = frame_count / elapsed_time

# Đóng webcam và cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()

# In tốc độ khung hình (fps)
print(f"Tốc độ khung hình (fps): {fps:.2f}")
