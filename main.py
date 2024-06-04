import numpy as np  # Thư viện NumPy để làm việc với các mảng
from ultralytics import YOLO  # Thư viện YOLO cho nhận diện đối tượng
import cv2  # Thư viện OpenCV để xử lý ảnh và video
import cvzone  # Thư viện CVZone cho các thao tác xử lý ảnh nâng cao
import math  # Thư viện math để thực hiện các phép toán
from sort import *  # Thư viện SORT cho theo dõi đối tượng

# Mở video từ file
cap = cv2.VideoCapture("D:/Self - Study/Object Detection/People-counter/assets/Videos/people.mp4")  # Đường dẫn tới video

# Khởi tạo mô hình YOLO với trọng số đã được huấn luyện
model = YOLO('yolov8n.pt')

# Định nghĩa các lớp đối tượng mà mô hình có thể nhận diện
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Đọc ảnh mặt nạ từ file
mask = cv2.imread('../mask-1.png')

# Khởi tạo bộ theo dõi đối tượng SORT với các tham số max_age, min_hits, iou_threshold
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Định nghĩa tọa độ của các đường giới hạn đếm đối tượng
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

# Khởi tạo danh sách chứa các ID của các đối tượng đã được đếm
totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()  # Đọc từng khung hình từ video
    imgRegion = cv2.bitwise_and(img, mask)  # Áp dụng mặt nạ lên khung hình
    
    # Đọc ảnh đồ họa và chèn lên khung hình
    imgGraphics = cv2.imread("D:/Self - Study/Object Detection/People-counter/assets/images/graphics-1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    
    # Nhận diện các đối tượng trong vùng đã áp dụng mặt nạ
    results = model(imgRegion, stream=True)

    # Khởi tạo mảng rỗng để chứa các bounding boxes và độ tin cậy
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Độ tin cậy
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Tên lớp đối tượng
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Chỉ xét các đối tượng là người và độ tin cậy lớn hơn 0.3
            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Cập nhật các đối tượng theo dõi với các bounding boxes mới
    resultsTracker = tracker.update(detections)

    # Vẽ các đường giới hạn đếm đối tượng
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Kiểm tra xem đối tượng có nằm trên đường vào không
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        # Kiểm tra xem đối tượng có nằm trên đường ra không
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # Hiển thị tổng số đối tượng vào và ra
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # output.write(img)
    cv2.imshow('result', img)  # Hiển thị khung hình với các đối tượng được nhận diện

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn phím 'q' để thoát
        break

cap.release()  # Giải phóng bộ nhớ của video
# output.release()
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ
