
# Random Forest (Rừng ngẫu nhiên) từ con số 0

Random Forest là một thuật toán **Học máy kết hợp (Ensemble Learning)** cực kỳ mạnh mẽ. Thay vì dựa vào kết quả của một cây quyết định duy nhất, nó xây dựng một "rừng" gồm rất nhiều cây quyết định và kết hợp kết quả của chúng lại để đưa ra dự đoán chính xác và ổn định hơn.

## 1. Khái niệm cơ bản

Nguyên lý cốt lõi của Random Forest là **"Trí tuệ tập thể"**. Một nhóm các cây quyết định hoạt động độc lập sẽ cho kết quả tốt hơn bất kỳ một cây đơn lẻ nào.

**Hai kỹ thuật then chốt tạo nên sự "Ngẫu nhiên":**
1.  **Bootstrapping (Lấy mẫu lặp lại):** Mỗi cây trong rừng được huấn luyện trên một tập con dữ liệu được lấy mẫu ngẫu nhiên từ tập gốc (có thể trùng lặp).
2.  **Feature Randomness (Ngẫu nhiên hóa đặc trưng):** Tại mỗi nút của cây, thuật toán chỉ chọn một nhóm nhỏ các đặc trưng ngẫu nhiên để tìm điểm chia tốt nhất, thay vì dùng toàn bộ đặc trưng.

**Cách đưa ra dự đoán:**
-   **Phân loại:** Lấy kết quả theo số đông (Majority Voting).
-   **Hồi quy:** Lấy giá trị trung bình (Average) từ tất cả các cây.

---

## 2. Tại sao Random Forest tốt hơn một Cây đơn lẻ?

Một cây quyết định đơn lẻ rất dễ bị **Overfitting** (Quá khớp) vì nó cố gắng mọc sâu để học hết các chi tiết của tập huấn luyện. Random Forest giải quyết vấn đề này bằng cách:
-   Trung bình hóa các kết quả giúp giảm **Variance** (Phương sai).
-   Sự ngẫu nhiên giúp các cây không bị giống hệt nhau, từ đó bao quát được nhiều quy luật khác nhau trong dữ liệu.

---

## 3. Triển khai logic bằng Python

Cấu trúc của Random Forest dựa trên việc lặp lại lớp `DecisionTree` đã xây dựng:

```python
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees              # Số lượng cây trong rừng
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features        # Số lượng đặc trưng ngẫu nhiên mỗi lần chia
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Tạo một cây mới (sử dụng lớp DecisionTree đã có)
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            
            # Lấy mẫu Bootstrap
            X_sample, y_sample = self._bootstrap_samples(X, y)
            
            # Huấn luyện cây trên mẫu đó
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # Chọn ngẫu nhiên chỉ số của các hàng (có lặp lại)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        # Lấy dự đoán từ tất cả các cây
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Chuyển hàng thành cột để dễ dàng bầu chọn
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Bầu chọn số đông cho từng ví dụ
        y_pred = [self._most_common_label(pred) for pred in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
```

---

## 4. Ưu điểm và Nhược điểm

**Ưu điểm:**
-   **Độ chính xác cực cao:** Thường là thuật toán hàng đầu cho các cuộc thi dữ liệu.
-   **Ít bị Overfitting:** Nhờ cơ chế Bagging và sự ngẫu nhiên.
-   **Đo lường được mức độ quan trọng của đặc trưng:** Giúp bạn biết biến nào ảnh hưởng nhất đến kết quả.
-   Làm việc tốt với dữ liệu bị thiếu và dữ liệu có số chiều lớn.

**Nhược điểm:**
-   **Mô hình hộp đen (Black Box):** Rất khó để giải thích chi tiết tại sao hàng trăm cây lại đưa ra kết luận như vậy (khó hơn một cây đơn lẻ).
-   **Tốn tài nguyên:** Cần nhiều bộ nhớ và thời gian tính toán hơn khi số lượng cây tăng lên.
-   Dự đoán chậm hơn các mô hình đơn giản như Naïve Bayes.

---

## 5. Những nội dung cần lưu ý

1.  **n_estimators:** Số lượng cây. Càng nhiều cây mô hình càng ổn định, nhưng đến một ngưỡng nhất định thì độ chính xác sẽ bão hòa và chỉ gây tốn thời gian chạy.
2.  **max_features:** Số lượng đặc trưng được chọn ngẫu nhiên. 
    -   Đối với phân loại, thường chọn: $\sqrt{\text{tổng số đặc trưng}}$.
    -   Đối với hồi quy, thường chọn: $\text{tổng số đặc trưng} / 3$.
3.  **Out-of-Bag (OOB) Error:** Vì chúng ta lấy mẫu Bootstrap, có khoảng 1/3 dữ liệu không được dùng để luyện cho một cây cụ thể. Ta có thể dùng chính phần dữ liệu này để kiểm tra (validation) độ chính xác của cây đó mà không cần chia tập Test riêng.

---
