from __future__ import print_function  # Import để sử dụng hàm print() trong cả Python 2 và 3

import os  # Import thư viện os để làm việc với hệ thống tệp
import numpy as np  # Import thư viện NumPy để làm việc với các mảng
import matplotlib  # Import thư viện matplotlib để vẽ biểu đồ
matplotlib.use('TkAgg')  # Sử dụng backend 'TkAgg' cho matplotlib
import matplotlib.pyplot as plt  # Import thư viện pyplot từ matplotlib
import matplotlib.patches as patches  # Import patches từ matplotlib để vẽ các hình
from skimage import io  # Import thư viện io từ skimage để đọc và ghi hình ảnh

import glob  # Import thư viện glob để làm việc với các mẫu tên tệp
import time  # Import thư viện time để đo thời gian
import argparse  # Import thư viện argparse để phân tích các tham số dòng lệnh
from filterpy.kalman import KalmanFilter  # Import KalmanFilter từ filterpy.kalman

np.random.seed(0)  # Đặt seed ngẫu nhiên cho NumPy

def linear_assignment(cost_matrix):
    """
    Hàm này dùng để giải bài toán gán tuyến tính dựa trên ma trận chi phí.
    
    Tham số:
    cost_matrix (ndarray): Ma trận chi phí.

    Trả về:
    ndarray: Mảng các cặp chỉ số (i, j) tối ưu dựa trên chi phí.
    """
    try:
        import lap  # Import thư viện lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)  # Sử dụng thuật toán LAPJV để giải bài toán gán
        return np.array([[y[i], i] for i in x if i >= 0])  # Trả về mảng các cặp chỉ số (i, j)
    except ImportError:
        from scipy.optimize import linear_sum_assignment  # Import linear_sum_assignment từ scipy.optimize
        x, y = linear_sum_assignment(cost_matrix)  # Sử dụng thuật toán Hungarian để giải bài toán gán
        return np.array(list(zip(x, y)))  # Trả về mảng các cặp chỉ số (i, j)

def iou_batch(bb_test, bb_gt):
    """
    Tính toán IOU giữa hai bounding boxes dưới dạng [x1, y1, x2, y2]
    
    Tham số:
    bb_test (ndarray): Bounding boxes kiểm tra.
    bb_gt (ndarray): Bounding boxes chuẩn.

    Trả về:
    ndarray: Giá trị IOU giữa các bounding boxes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)  # Thêm một chiều vào bb_gt
    bb_test = np.expand_dims(bb_test, 1)  # Thêm một chiều vào bb_test
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])  # Tọa độ x1 lớn nhất
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])  # Tọa độ y1 lớn nhất
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])  # Tọa độ x2 nhỏ nhất
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])  # Tọa độ y2 nhỏ nhất
    w = np.maximum(0., xx2 - xx1)  # Chiều rộng của vùng giao nhau
    h = np.maximum(0., yy2 - yy1)  # Chiều cao của vùng giao nhau
    wh = w * h  # Diện tích của vùng giao nhau
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + 
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)  # Tính IOU
    return o  # Trả về giá trị IOU

def convert_bbox_to_z(bbox):
    """
    Chuyển đổi bounding box từ dạng [x1, y1, x2, y2] sang dạng [x, y, s, r] 
    với x, y là trung tâm của hộp, s là diện tích, r là tỉ lệ khung hình.
    
    Tham số:
    bbox (ndarray): Bounding box cần chuyển đổi.

    Trả về:
    ndarray: Bounding box dưới dạng [x, y, s, r].
    """
    w = bbox[2] - bbox[0]  # Chiều rộng của bounding box
    h = bbox[3] - bbox[1]  # Chiều cao của bounding box
    x = bbox[0] + w / 2.  # Tọa độ x của trung tâm
    y = bbox[1] + h / 2.  # Tọa độ y của trung tâm
    s = w * h  # Diện tích
    r = w / float(h)  # Tỉ lệ khung hình
    return np.array([x, y, s, r]).reshape((4, 1))  # Trả về bounding box dưới dạng [x, y, s, r]

def convert_x_to_bbox(x, score=None):
    """
    Chuyển đổi bounding box từ dạng [x, y, s, r] sang dạng [x1, y1, x2, y2]
    với x1, y1 là góc trên bên trái và x2, y2 là góc dưới bên phải
    
    Tham số:
    x (ndarray): Bounding box dưới dạng [x, y, s, r].
    score (float, optional): Điểm số của bounding box.

    Trả về:
    ndarray: Bounding box dưới dạng [x1, y1, x2, y2].
    """
    w = np.sqrt(x[2] * x[3])  # Chiều rộng của bounding box
    h = x[2] / w  # Chiều cao của bounding box
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    Lớp này đại diện cho trạng thái nội bộ của các đối tượng được theo dõi, được quan sát dưới dạng bbox.
    """
    count = 0  # Đếm số lượng đối tượng được tạo
    def __init__(self, bbox):
        """
        Khởi tạo một tracker sử dụng bounding box ban đầu.
        
        Tham số:
        bbox (ndarray): Bounding box ban đầu.
        """
        # Định nghĩa mô hình vận tốc không đổi
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # Khởi tạo bộ lọc Kalman với 7 biến trạng thái và 4 biến đo lường
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 0], 
                              [0, 0, 1, 0, 0, 0, 1], 
                              [0, 0, 0, 1, 0, 0, 0],  
                              [0, 0, 0, 0, 1, 0, 0], 
                              [0, 0, 0, 0, 0, 1, 0], 
                              [0, 0, 0, 0, 0, 0, 1]])  # Ma trận chuyển trạng thái
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0, 0, 0], 
                              [0, 0, 1, 0, 0, 0, 0], 
                              [0, 0, 0, 1, 0, 0, 0]])  # Ma trận quan sát

        self.kf.R[2:, 2:] *= 10.  # Ma trận nhiễu quan sát
        self.kf.P[4:, 4:] *= 1000.  # Ma trận hiệp phương sai
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01  # Ma trận nhiễu quá trình
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # Trạng thái ban đầu
        self.time_since_update = 0  # Thời gian kể từ lần cập nhật cuối cùng
        self.id = KalmanBoxTracker.count  # ID của đối tượng
        KalmanBoxTracker.count += 1
        self.history = []  # Lịch sử trạng thái
        self.hits = 0  # Số lần phát hiện
        self.hit_streak = 0  # Số lần phát hiện liên tiếp
        self.age = 0  # Tuổi của đối tượng

    def update(self, bbox):
        """
        Cập nhật vector trạng thái với bbox quan sát được.
        
        Tham số:
        bbox (ndarray): Bounding box quan sát được.
        """
        self.time_since_update = 0  # Đặt lại thời gian kể từ lần cập nhật cuối cùng
        self.history = []  # Xóa lịch sử trạng thái
        self.hits += 1  # Tăng số lần phát hiện
        self.hit_streak += 1  # Tăng số lần phát hiện liên tiếp
        self.kf.update(convert_bbox_to_z(bbox))  # Cập nhật trạng thái của bộ lọc Kalman

    def predict(self):
        """
        Dự đoán vector trạng thái và trả về ước lượng bounding box.
        
        Trả về:
        ndarray: Bounding box dự đoán.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:  # Kiểm tra điều kiện bounding box dự đoán
            self.kf.x[6] *= 0.0
        self.kf.predict()  # Dự đoán trạng thái tiếp theo
        self.age += 1  # Tăng tuổi của đối tượng
        if self.time_since_update > 0:  # Kiểm tra thời gian kể từ lần cập nhật cuối cùng
            self.hit_streak = 0  # Đặt lại số lần phát hiện liên tiếp
        self.time_since_update += 1  # Tăng thời gian kể từ lần cập nhật cuối cùng
        self.history.append(convert_x_to_bbox(self.kf.x))  # Thêm bounding box vào lịch sử trạng thái
        return self.history[-1]  # Trả về bounding box dự đoán

    def get_state(self):
        """
        Trả về ước lượng bounding box hiện tại.
        
        Trả về:
        ndarray: Bounding box hiện tại.
        """
        return convert_x_to_bbox(self.kf.x)  # Trả về bounding box hiện tại

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Gán các phát hiện cho các đối tượng được theo dõi (cả hai đều được biểu diễn dưới dạng bounding boxes).
    
    Tham số:
    detections (ndarray): Các bounding boxes phát hiện được.
    trackers (ndarray): Các bounding boxes của đối tượng theo dõi.
    iou_threshold (float): Ngưỡng IOU tối thiểu để gán đối tượng.

    Trả về:
    tuple: Các chỉ số của đối tượng phù hợp, không phù hợp trong phát hiện và theo dõi.
    """
    if len(trackers) == 0:  # Kiểm tra nếu không có đối tượng theo dõi
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)  # Tính toán ma trận IOU giữa các bounding boxes

    if min(iou_matrix.shape) > 0:  # Kiểm tra nếu có ít nhất một bounding box
        a = (iou_matrix > iou_threshold).astype(np.int32)  # Áp dụng ngưỡng IOU
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:  # Kiểm tra điều kiện gán đối tượng
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)  # Gán đối tượng bằng thuật toán gán tuyến tính
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []  # Danh sách các phát hiện không phù hợp
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []  # Danh sách các đối tượng theo dõi không phù hợp
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []  # Danh sách các cặp chỉ số phù hợp
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:  # Lọc các cặp chỉ số có IOU thấp
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Khởi tạo các tham số chính cho SORT.
        
        Tham số:
        max_age (int): Số khung hình tối đa để duy trì một đối tượng nếu không có phát hiện mới.
        min_hits (int): Số lần phát hiện tối thiểu để xác nhận một đối tượng.
        iou_threshold (float): Ngưỡng IOU tối thiểu để gán đối tượng.
        """
        self.max_age = max_age  # Số khung hình tối đa
        self.min_hits = min_hits  # Số lần phát hiện tối thiểu
        self.iou_threshold = iou_threshold  # Ngưỡng IOU
        self.trackers = []  # Danh sách các đối tượng theo dõi
        self.frame_count = 0  # Đếm số khung hình

    def update(self, dets=np.empty((0, 5))):
        """
        Cập nhật các đối tượng theo dõi với các phát hiện mới.
        
        Tham số:
        dets (ndarray): Mảng các phát hiện dưới dạng [[x1, y1, x2, y2, score], ...].

        Trả về:
        ndarray: Mảng các đối tượng theo dõi với cột cuối là ID của đối tượng.
        """
        self.frame_count += 1  # Tăng số khung hình
        trks = np.zeros((len(self.trackers), 5))  # Khởi tạo mảng trks với kích thước của danh sách đối tượng theo dõi
        to_del = []  # Danh sách các chỉ số cần xóa
        ret = []  # Danh sách kết quả trả về
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # Dự đoán vị trí của đối tượng
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # Cập nhật vị trí vào mảng trks
            if np.any(np.isnan(pos)):  # Kiểm tra nếu vị trí không hợp lệ
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # Nén các hàng không hợp lệ
        for t in reversed(to_del):
            self.trackers.pop(t)  # Xóa các đối tượng không hợp lệ
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)  # Gán các phát hiện mới

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])  # Cập nhật các đối tượng theo dõi với phát hiện mới

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])  # Tạo và khởi tạo các đối tượng theo dõi mới
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # Thêm ID của đối tượng vào kết quả trả về
            i -= 1
            if trk.time_since_update > self.max_age:  # Kiểm tra nếu đối tượng đã quá tuổi
                self.trackers.pop(i)  # Xóa đối tượng
        if len(ret) > 0:
            return np.concatenate(ret)  # Trả về kết quả cuối cùng
        return np.empty((0, 5))

def parse_args():
    """Phân tích các tham số đầu vào."""
    parser = argparse.ArgumentParser(description='SORT demo')  # Tạo đối tượng ArgumentParser
    parser.add_argument('--display', dest='display', help='Hiển thị kết quả theo dõi trực tuyến (chậm) [False]', action='store_true')  # Tham số để hiển thị kết quả
    parser.add_argument("--seq_path", help="Đường dẫn đến các phát hiện.", type=str, default='data')  # Tham số đường dẫn tới các phát hiện
    parser.add_argument("--phase", help="Thư mục con trong seq_path.", type=str, default='train')  # Tham số thư mục con
    parser.add_argument("--max_age", 
                        help="Số khung hình tối đa để duy trì một đối tượng nếu không có phát hiện mới.", 
                        type=int, default=1)  # Tham số số khung hình tối đa
    parser.add_argument("--min_hits", 
                        help="Số lần phát hiện tối thiểu để xác nhận một đối tượng.", 
                        type=int, default=3)  # Tham số số lần phát hiện tối thiểu
    parser.add_argument("--iou_threshold", help="Ngưỡng IOU tối thiểu để gán đối tượng.", type=float, default=0.3)  # Tham số ngưỡng IOU
    args = parser.parse_args()  # Phân tích các tham số
    return args

if __name__ == '__main__':
    # Tất cả các tập huấn luyện
    args = parse_args()  # Phân tích các tham số đầu vào
    display = args.display  # Lấy giá trị tham số display
    phase = args.phase  # Lấy giá trị tham số phase
    total_time = 0.0  # Thời gian tổng cộng
    total_frames = 0  # Tổng số khung hình
    colours = np.random.rand(32, 3)  # Mảng màu ngẫu nhiên dùng để hiển thị
    if display:
        if not os.path.exists('mot_benchmark'):  # Kiểm tra nếu không tồn tại thư mục mot_benchmark
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()  # Thoát chương trình nếu không tìm thấy liên kết
        plt.ion()  # Bật chế độ interactive của matplotlib
        fig = plt.figure()  # Tạo đối tượng Figure
        ax1 = fig.add_subplot(111, aspect='equal')  # Tạo đối tượng Axes

    if not os.path.exists('output'):  # Kiểm tra nếu không tồn tại thư mục output
        os.makedirs('output')  # Tạo thư mục output
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')  # Mẫu tên tệp để tìm kiếm các tệp phát hiện
    for seq_dets_fn in glob.glob(pattern):  # Duyệt qua các tệp phát hiện
        mot_tracker = Sort(max_age=args.max_age, 
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # Tạo instance của SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')  # Đọc các phát hiện từ tệp
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        
        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:  # Mở tệp output để ghi
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):  # Duyệt qua các khung hình
                frame += 1  # Tăng số khung hình
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # Lấy các phát hiện cho khung hình hiện tại
                dets[:, 2:4] += dets[:, 0:2]  # Chuyển đổi từ [x1, y1, w, h] sang [x1, y1, x2, y2]
                total_frames += 1

                if display:
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))  # Tên tệp hình ảnh
                    im = io.imread(fn)  # Đọc hình ảnh
                    ax1.imshow(im)  # Hiển thị hình ảnh
                    plt.title(seq + ' Tracked Targets')  # Tiêu đề của biểu đồ

                start_time = time.time()  # Thời gian bắt đầu
                trackers = mot_tracker.update(dets)  # Cập nhật các đối tượng theo dõi với các phát hiện mới
                cycle_time = time.time() - start_time  # Thời gian xử lý một khung hình
                total_time += cycle_time  # Thời gian tổng cộng

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)  # Ghi kết quả vào tệp
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))  # Vẽ bounding box

                if display:
                    fig.canvas.flush_events()  # Cập nhật biểu đồ
                    plt.draw()  # Vẽ lại biểu đồ
                    ax1.cla()  # Xóa biểu đồ

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))  # Thời gian và số khung hình

    if display:
        print("Note: to get real runtime results run without the option: --display")  # Ghi chú khi không sử dụng tùy chọn hiển thị
