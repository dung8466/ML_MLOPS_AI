
# K-Nearest Neighbors (KNN) từ con số 0

KNN là một trong những thuật toán **Học có giám sát (Supervised Learning)** đơn giản và hiệu quả nhất. Nó có thể dùng cho cả bài toán Phân loại (Classification) và Hồi quy (Regression). KNN được gọi là thuật toán "Học lười" (Lazy Learning) vì nó không có giai đoạn huấn luyện thực sự; thay vào đó, nó ghi nhớ toàn bộ dữ liệu và chỉ tính toán khi cần dự đoán.

## 1. Khái niệm cơ bản

Nguyên lý của KNN rất đơn giản: "Những gì giống nhau thì thường ở gần nhau". 

**Các bước thực hiện thuật toán:**
1. **Lưu trữ:** Lưu toàn bộ tập dữ liệu huấn luyện.
2. **Tính khoảng cách:** Khi có một điểm dữ liệu mới, tính khoảng cách từ điểm đó đến **tất cả** các điểm trong tập huấn luyện.
3. **Tìm hàng xóm:** Chọn ra $K$ điểm có khoảng cách gần nhất.
4. **Đưa ra kết quả:**
    - **Với Phân loại:** Nhãn nào xuất hiện nhiều nhất trong $K$ hàng xóm sẽ là kết quả dự đoán (Majority Voting).
    - **Với Hồi quy:** Lấy giá trị trung bình của $K$ hàng xóm làm kết quả dự đoán.

---

## 2. Khoảng cách Euclidean (Euclidean Distance)

Trong không gian $n$ chiều, khoảng cách Euclidean giữa hai điểm $x$ và $y$ là thước đo phổ biến nhất để xác định "độ gần".

**Công thức:**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

---

## 3. Triển khai thuật toán bằng Python

Dưới đây là cấu trúc cơ bản của lớp `KNN` sử dụng NumPy và thư viện `collections` để đếm nhãn:

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN không học gì cả, chỉ lưu lại dữ liệu
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Dự đoán cho từng điểm dữ liệu trong mảng đầu vào X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # 1. Tính khoảng cách từ x đến tất cả các điểm trong X_train
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # 2. Sắp xếp và lấy chỉ số của K điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Lấy nhãn của K điểm đó
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Bầu chọn nhãn phổ biến nhất
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```

---

## 4. Cách chọn giá trị $K$ phù hợp

Việc chọn siêu tham số $K$ ảnh hưởng trực tiếp đến kết quả của mô hình:

- **Nếu $K$ quá nhỏ (ví dụ $K=1$):** Mô hình rất nhạy cảm với các điểm nhiễu (outliers), dễ dẫn đến **Overfitting** (Quá khớp). Vùng biên phân loại sẽ rất sắc nhọn và lởm chởm.
- **Nếu $K$ quá lớn:** Mô hình trở nên "quá đơn giản", vùng biên phẳng lại, dễ dẫn đến **Underfitting** (Dưới khớp).
- **Quy tắc thực nghiệm:** Người ta thường chọn $K$ là một số lẻ để tránh tình trạng bầu chọn hòa nhau (ví dụ: $K=3, 5, 7$).

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
- Cực kỳ đơn giản, dễ giải thích và triển khai.
- Không cần giả định gì về phân phối của dữ liệu.
- Hiệu quả nếu tập huấn luyện đủ lớn và dữ liệu sạch.

**Nhược điểm:**
- **Chi phí tính toán cao:** Phải tính khoảng cách đến mọi điểm mỗi khi dự đoán (chạy rất chậm với dữ liệu khổng lồ).
- **Tốn bộ nhớ:** Phải lưu trữ toàn bộ tập dữ liệu trong RAM.
- **Nhạy cảm với quy mô:** Nếu một đặc trưng có giá trị lớn hơn các đặc trưng khác, nó sẽ áp đảo kết quả tính khoảng cách.

---

## 6. Những nội dung cần lưu ý

1. **Chuẩn hóa dữ liệu (Feature Scaling):** Đây là bước **sống còn** với KNN. Bạn bắt buộc phải đưa các đặc trưng về cùng quy mô (ví dụ: dùng Z-score hoặc Min-Max Scaling) để khoảng cách được tính toán công bằng.
2. **Lời nguyền đa chiều (Curse of Dimensionality):** Khi số lượng đặc trưng quá lớn, không gian trở nên thưa thớt, khái niệm "gần" không còn ý nghĩa thực tế. Cần giảm chiều dữ liệu (như PCA) trước khi dùng KNN.
3. **Xử lý dữ liệu thiếu:** KNN không thể làm việc với giá trị null, bạn cần điền khuyết (Imputation) trước khi tính khoảng cách.

---
