---

# Phân cụm K-Means (K-Means Clustering) từ con số 0

K-Means là một trong những thuật toán **Học không giám sát (Unsupervised Learning)** phổ biến nhất. Nó được sử dụng để nhóm các điểm dữ liệu tương đồng vào các cụm (clusters) dựa trên khoảng cách giữa chúng.

## 1. Khái niệm cơ bản

Mục tiêu của K-Means là chia $n$ quan sát thành $k$ cụm sao cho mỗi quan sát thuộc về cụm có tâm gần nhất. 

**Các bước thực hiện thuật toán:**
1. **Khởi tạo:** Chọn ngẫu nhiên $k$ điểm làm tâm cụm ban đầu (centroids).
2. **Gán cụm (Assignment):** Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất (thường dùng khoảng cách Euclidean).
3. **Cập nhật (Update):** Tính toán lại vị trí tâm cụm bằng cách lấy trung bình cộng của tất cả các điểm trong cụm đó.
4. **Lặp lại:** Thực hiện lại bước 2 và 3 cho đến khi vị trí các tâm cụm không còn thay đổi đáng kể hoặc đạt số vòng lặp tối đa.

---

## 2. Khoảng cách Euclidean (Euclidean Distance)

Đây là thước đo phổ biến nhất để tính toán độ tương đồng giữa các điểm dữ liệu.

**Công thức:**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

---

## 3. Triển khai thuật toán bằng Python

Dưới đây là cấu trúc cơ bản của lớp `KMeans` sử dụng NumPy:

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # 1. Khởi tạo tâm cụm ngẫu nhiên
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            # 2. Gán điểm vào tâm cụm gần nhất
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            # 3. Cập nhật tâm cụm mới (lấy trung bình cộng)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # Kiểm tra điều kiện dừng
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break
```

---

## 4. Cách chọn số lượng cụm $K$: Phương pháp Khuỷu tay (Elbow Method)

Một trong những thách thức lớn nhất của K-Means là chọn số $k$ phù hợp. Chúng ta sử dụng chỉ số **WCSS** (Within-Cluster Sum of Squares - Tổng bình phương khoảng cách trong cụm).

- Chúng ta chạy thuật toán với các giá trị $k$ khác nhau (ví dụ từ 1 đến 10).
- Vẽ biểu đồ giữa $k$ và WCSS.
- **Điểm khuỷu tay (Elbow point):** Là vị trí mà tại đó WCSS giảm chậm lại rõ rệt. Đây thường là giá trị $k$ tối ưu.

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
- Đơn giản, dễ hiểu và dễ triển khai.
- Hiệu quả về mặt tính toán với tập dữ liệu lớn.

**Nhược điểm:**
- Phải xác định trước số lượng cụm $k$.
- Nhạy cảm với các điểm dữ liệu ngoại lai (outliers).
- Kết quả có thể phụ thuộc vào cách khởi tạo tâm cụm ban đầu (có thể rơi vào cực tiểu địa phương).

---

## 6. Những nội dung cần lưu ý

1. **Chuẩn hóa dữ liệu (Feature Scaling):** Vì K-Means dựa trên khoảng cách, các đặc trưng có quy mô lớn (ví dụ: thu nhập) sẽ lấn át các đặc trưng có quy mô nhỏ (ví dụ: tuổi). Luôn cần chuẩn hóa dữ liệu trước khi chạy.
2. **Khởi tạo thông minh:** Kỹ thuật **K-Means++** thường được dùng để chọn tâm cụm ban đầu tốt hơn là chọn ngẫu nhiên hoàn toàn.
3. **Ứng dụng:** Phân khúc khách hàng, nén ảnh, phát hiện bất thường.

---
