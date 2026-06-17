
# Extreme Gradient Boosting (XGBoost)

XGBoost là một phiên bản cải tiến "cực đoan" của Gradient Boosting. Nó được thiết kế để tối ưu hóa cả về **tốc độ tính toán** lẫn **hiệu suất dự báo**. Đây là thuật toán "vô địch" trong rất nhiều cuộc thi Kaggle nhờ khả năng kiểm soát Overfitting cực tốt và tốc độ xử lý dữ liệu khổng lồ.

## 1. Khái niệm cơ bản

XGBoost giữ nguyên tư tưởng cốt lõi của Gradient Boosting (cây sau sửa lỗi cây trước) nhưng thêm vào các cải tiến đột phá:

1.  **Regularization (Điều quy hóa):** XGBoost tích hợp sẵn các hình phạt L1 và L2 vào hàm mục tiêu để kiểm soát độ phức tạp của cây, giúp ngăn chặn Overfitting hiệu quả hơn Gradient Boosting truyền thống.
2.  **Đạo hàm bậc hai (Hessian):** Thay vì chỉ dùng đạo hàm bậc nhất (Gradient) để tối ưu hóa, XGBoost sử dụng khai triển Taylor bậc hai của hàm mất mát, giúp việc tìm điểm tối ưu nhanh và chính xác hơn.
3.  **Xử lý dữ liệu thưa (Sparsity Awareness):** Thuật toán có khả năng tự động học cách xử lý các giá trị thiếu (Missing values) hoặc dữ liệu thưa (như One-hot encoding).
4.  **Cắt tỉa cây (Pruning):** XGBoost sử dụng tham số `gamma` để quyết định xem một nút có nên được chia hay không dựa trên mức độ giảm thiểu hàm mất mát.

---

## 2. Công thức toán học then chốt

Hàm mục tiêu (Objective Function) của XGBoost gồm hai phần:
$$Obj(\theta) = \sum L(y_i, \hat{y}_i) + \sum \Omega(f_k)$$

Trong đó $\Omega(f_k)$ là độ phức tạp của cây:
$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w^2$$
*(với $T$ là số lượng lá và $w$ là trọng số tại mỗi lá)*.

**Chỉ số tương đồng (Similarity Score):** Được dùng để đánh giá chất lượng của một nút:
$$Similarity = \frac{(\sum g_i)^2}{\sum h_i + \lambda}$$
*(với $g_i$ là Gradient bậc 1 và $h_i$ là Hessian bậc 2)*.

---

## 3. Triển khai logic bằng Python

Dưới đây là cấu trúc logic cách XGBoost tính toán để xây dựng một cây:

```python
import numpy as np

class XGBoostNode:
    def __init__(self):
        # Tính toán Gradient và Hessian của hàm mất mát
        # Ví dụ với MSE: g = 2 * (y_pred - y), h = 2
        pass

    def calculate_similarity(self, gradients, hessians, lamb):
        # Công thức Similarity Score
        sum_g = np.sum(gradients)
        sum_h = np.sum(hessians)
        return (sum_g ** 2) / (sum_h + lamb)

    def calculate_gain(self, left_g, left_h, right_g, right_h, lamb):
        # Tính toán Gain để chọn điểm chia tốt nhất
        left_sim = self.calculate_similarity(left_g, left_h, lamb)
        right_sim = self.calculate_similarity(right_g, right_h, lamb)
        root_sim = self.calculate_similarity(left_g + right_g, left_h + right_h, lamb)
        
        # Gain = Sim(Trái) + Sim(Phải) - Sim(Gốc)
        return left_sim + right_sim - root_sim

class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.3, lamb=1, gamma=0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.lamb = lamb   # L2 regularization
        self.gamma = gamma # Ngưỡng cắt tỉa
        self.trees = []

    def fit(self, X, y):
        # Dự đoán ban đầu thường là 0.5
        y_pred = np.full(y.shape, 0.5)
        
        for _ in range(self.n_estimators):
            # 1. Tính g và h cho từng điểm dữ liệu
            # 2. Xây dựng cây dựa trên Gain (sử dụng g và h)
            # 3. Tính toán trọng số tối ưu tại mỗi lá: w = -sum(g) / (sum(h) + lamb)
            # 4. Cập nhật dự đoán: y_pred += lr * tree.predict(X)
            pass
```

---

## 4. So sánh: XGBoost vs. Gradient Boosting (GBM)

| Đặc điểm | Gradient Boosting (GBM) | XGBoost |
| :--- | :--- | :--- |
| **Điều quy hóa** | Không có sẵn (dễ Overfit) | Có L1 & L2 (ổn định hơn) |
| **Đạo hàm** | Bậc nhất (Gradient) | Bậc nhất & Bậc hai (Hessian) |
| **Xử lý dữ liệu thiếu** | Cần điền khuyết trước | Tự động xử lý |
| **Tốc độ** | Chậm (tuần tự hoàn toàn) | Rất nhanh (hỗ trợ tính toán song song) |

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
-   **Hiệu suất cực mạnh:** Xử lý tốt các quy luật phi tuyến tính phức tạp.
-   **Tốc độ:** Tận dụng tối đa sức mạnh CPU thông qua xử lý song song các cột dữ liệu.
-   **Cắt tỉa thông minh:** Tránh việc mọc cây quá sâu gây lãng phí tài nguyên bằng cơ chế "Gain" và "Gamma".

**Nhược điểm:**
-   **Khó hiểu bản chất (Black Box):** Cấu trúc toán học đằng sau rất phức tạp so với cây đơn lẻ.
-   **Nhiều siêu tham số:** Cần kinh nghiệm để tinh chỉnh (tuning) các tham số như `eta`, `max_depth`, `lambda`, `gamma`.

---

## 6. Những nội dung cần lưu ý

1.  **Hessian:** Điểm khác biệt lớn nhất là XGBoost quan tâm đến tốc độ thay đổi của Gradient (đạo hàm bậc 2), giúp nó hội tụ về điểm tối ưu nhanh hơn nhiều.
2.  **Tham số `gamma`:** Nếu Gain tính được nhỏ hơn `gamma`, thuật toán sẽ không chia nút đó nữa. Đây là cách cắt tỉa cây ngay trong lúc mọc.
3.  **Hỗ trợ GPU:** Phiên bản thư viện của XGBoost có thể chạy trên GPU, giúp xử lý hàng tỷ dòng dữ liệu trong thời gian ngắn.

---
