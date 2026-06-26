# CUSTOMER SEGMENTATION USING DBSCAN

## Step 1. Load Dataset

Dữ liệu đã được tiền xử lý trước được đọc từ file:

```text
customer_personality_preprocessed.csv
```

Dataset được chuyển thành ma trận đặc trưng `X` để phục vụ quá trình phân cụm.

Mục tiêu của đề tài là sử dụng thuật toán DBSCAN để phân nhóm khách hàng dựa trên các đặc trưng hành vi và chi tiêu.

---

## Step 2. Distance Function

Khoảng cách giữa hai điểm dữ liệu được tính bằng công thức Euclidean Distance:

[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
]

Khoảng cách này được sử dụng để xác định mức độ gần nhau giữa các khách hàng trong không gian đặc trưng.

---

## Step 3. Neighborhood Search

Hàm `region_query()` được xây dựng để tìm tất cả các điểm nằm trong bán kính `eps` của một điểm dữ liệu.

Kết quả trả về là danh sách các điểm lân cận (neighbors), phục vụ cho việc xác định Core Point trong DBSCAN.

---

## Step 4. K-Distance Analysis

Trước khi xây dựng mô hình, K-Distance Graph được sử dụng để lựa chọn giá trị `eps` phù hợp.

Thực hiện với:

* k = 15
* k = 20

Đối với mỗi điểm dữ liệu:

1. Tính khoảng cách đến tất cả các điểm còn lại.
2. Sắp xếp khoảng cách theo thứ tự tăng dần.
3. Lấy khoảng cách tới láng giềng thứ k.
4. Vẽ đồ thị K-Distance.

Điểm gãy (Elbow Point) trên đồ thị được sử dụng để ước lượng giá trị `eps`.

---

## Step 5. DBSCAN Implementation

Thuật toán DBSCAN được cài đặt thủ công (không sử dụng thư viện có sẵn).

Các tham số chính:

| Parameter   | Ý nghĩa                             |
| ----------- | ----------------------------------- |
| eps         | Bán kính lân cận                    |
| min_samples | Số điểm tối thiểu để tạo Core Point |

Quy trình thực hiện:

1. Duyệt từng điểm dữ liệu.
2. Tìm các điểm lân cận trong phạm vi `eps`.
3. Nếu số lượng lân cận nhỏ hơn `min_samples` → đánh dấu Noise.
4. Nếu đủ điều kiện → tạo Cluster mới.
5. Mở rộng Cluster bằng cách kết nối các Core Point liên thông.

Kết quả:

* Các điểm cùng cụm được gán cùng nhãn.
* Điểm nhiễu được gán nhãn `-1`.

---

## Step 6. Cluster Evaluation

Để đánh giá chất lượng phân cụm, Silhouette Score được xây dựng thủ công.

Silhouette Score đo lường:

* Độ tương đồng của điểm với cụm hiện tại.
* Mức độ tách biệt với các cụm khác.

Giá trị:

* Gần 1: cụm tốt.
* Gần 0: cụm chồng lấn.
* Âm: phân cụm kém.

---

## Step 7. Hyperparameter Tuning

Nhiều tổ hợp tham số được thử nghiệm:

| eps | min_samples |
| --- | ----------- |
| 4.0 | 15          |
| 4.0 | 20          |
| 4.0 | 25          |
| 5.0 | 15          |
| 5.0 | 20          |
| 5.0 | 25          |
| 6.0 | 15          |
| 6.0 | 20          |
| 6.0 | 25          |

Đối với mỗi tổ hợp:

* Huấn luyện DBSCAN.
* Tính số cụm tạo được.
* Tính số điểm nhiễu.
* Tính tỷ lệ nhiễu.
* Tính Silhouette Score.

Các kết quả được sắp xếp theo Silhouette Score để lựa chọn mô hình tối ưu.

---

## Step 8. Final Model Training

Sau quá trình thử nghiệm, mô hình cuối cùng được lựa chọn:

| Parameter   | Value |
| ----------- | ----- |
| eps         | 6.0   |
| min_samples | 15    |

Mô hình được huấn luyện trên toàn bộ tập dữ liệu và sinh ra nhãn phân cụm cho từng khách hàng.

---

## Step 9. Final Evaluation

Sau khi huấn luyện:

* Tính tổng số cụm được tạo.
* Tính Silhouette Score cuối cùng.
* Thống kê số lượng khách hàng trong từng cụm.
* Xác định số lượng điểm nhiễu (Noise).

Các chỉ số này được sử dụng để đánh giá khả năng phân nhóm khách hàng của DBSCAN.

---

## Step 10. Cluster Visualization

Kết quả phân cụm được trực quan hóa bằng biểu đồ Scatter Plot.

Các biến được sử dụng:

* Income
* MntWines

Hai biểu đồ được xây dựng:

### Visualization 1

Hiển thị toàn bộ khách hàng với màu sắc đại diện cho từng cụm.

### Visualization 2

Hiển thị chi tiết:

* Các cụm khách hàng.
* Các điểm nhiễu (Noise) với ký hiệu riêng.

Qua đó có thể quan sát trực quan sự phân tách giữa các nhóm khách hàng.

---

## Workflow Summary

```text
Load Dataset
      ↓
Distance Calculation
      ↓
Neighborhood Search
      ↓
K-Distance Analysis
      ↓
DBSCAN Implementation
      ↓
Silhouette Evaluation
      ↓
Hyperparameter Tuning
      ↓
Final Model Training
      ↓
Cluster Visualization
```
