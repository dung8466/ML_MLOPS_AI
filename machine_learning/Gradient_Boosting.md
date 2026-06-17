# Gradient Boosting

Gradient Boosting là một kỹ thuật **Học máy kết hợp (Ensemble Learning)** mạnh mẽ, thuộc nhóm **Boosting**. Khác với Random Forest (xây dựng các cây song song), Gradient Boosting xây dựng các cây **tuần tự**, trong đó mỗi cây mới được tạo ra để sửa chữa những sai lầm của các cây trước đó.

## 1. Khái niệm cơ bản

Mục tiêu của Gradient Boosting là tối ưu hóa hàm mất mát (Loss Function) bằng cách sử dụng thuật toán **Gradient Descent** để thêm các mô hình yếu (thường là các cây quyết định nông).

**Cơ chế hoạt động:**
1.  **Khởi tạo:** Bắt đầu bằng một dự đoán đơn giản (thường là giá trị trung bình của nhãn $y$).
2.  **Tính toán phần dư (Residuals):** Tính khoảng cách giữa giá trị thực tế và giá trị dự đoán hiện tại. Đây chính là "sai số" mà các cây sau cần phải xử lý.
3.  **Học từ sai số:** Huấn luyện một cây quyết định mới để dự đoán **phần dư** vừa tính được (thay vì dự đoán nhãn gốc).
4.  **Cập nhật dự đoán:** Cộng dự đoán của cây mới vào kết quả hiện tại để tiến gần hơn đến giá trị thực tế.
5.  **Lặp lại:** Thực hiện liên tục cho đến khi đạt số lượng cây quy định.

---

## 2. Công thức toán học đơn giản

Trong bài toán Hồi quy với hàm mất mát là Bình phương sai số ($MSE$), phần dư chính là đạo âm của hàm mất mát (Negative Gradient):

$$Residual_i = y_i - \hat{y}_i$$

Cập nhật mô hình tại bước $m$:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

*Trong đó:*
- $F_m(x)$: Mô hình hiện tại.
- $\eta$ (Eta): Tốc độ học (**Learning Rate**) giúp kiềm chế mỗi cây để tránh quá khớp.
- $h_m(x)$: Cây quyết định mới được huấn luyện trên phần dư của bước $m-1$.

---

## 3. Triển khai logic bằng Python

Dưới đây là cấu trúc đơn giản của Gradient Boosting cho bài toán Hồi quy:

```python
import numpy as np

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators  # Số lượng cây
        self.learning_rate = learning_rate # Tốc độ học
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None

    def fit(self, X, y):
        # 1. Khởi tạo dự đoán ban đầu (trung bình của y)
        self.init_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.init_prediction)

        for _ in range(self.n_estimators):
            # 2. Tính phần dư (Residuals)
            residuals = y - y_pred

            # 3. Huấn luyện một cây để dự đoán phần dư
            # (Sử dụng lớp DecisionTreeRegressor từ thư viện hoặc tự xây dựng)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # 4. Cập nhật dự đoán hiện tại
            # y_new = y_old + lr * tree_prediction
            predictions = tree.predict(X)
            y_pred += self.learning_rate * predictions
            
            self.trees.append(tree)

    def predict(self, X):
        # Bắt đầu với dự đoán trung bình
        y_pred = np.full((X.shape[0],), self.init_prediction)
        
        # Cộng dồn dự đoán từ tất cả các cây trong rừng
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
```

---

## 4. So sánh: Gradient Boosting vs. Random Forest

| Đặc điểm | Random Forest | Gradient Boosting |
| :--- | :--- | :--- |
| **Cách xây dựng** | Song song (Các cây độc lập) | Tuần tự (Cây sau học từ cây trước) |
| **Mục tiêu** | Giảm **Variance** (Phương sai) | Giảm **Bias** (Độ chệch) |
| **Độ phức tạp cây** | Thường là cây sâu | Thường là cây rất nông (Stumps) |
| **Độ nhạy với nhiễu** | Ít nhạy cảm hơn | Rất nhạy cảm với dữ liệu nhiễu |

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
-   **Độ chính xác vượt trội:** Thường cho kết quả tốt nhất trong các bài toán dữ liệu dạng bảng.
-   **Tính linh hoạt:** Có thể tối ưu hóa nhiều loại hàm mất mát khác nhau (MSE, Log-Loss, MAE).
-   Không cần chuẩn hóa dữ liệu đầu vào.

**Nhược điểm:**
-   **Dễ Overfitting:** Nếu số lượng cây quá lớn hoặc tốc độ học quá cao, mô hình sẽ học thuộc lòng dữ liệu.
-   **Chậm hơn khi huấn luyện:** Do tính chất tuần tự (phải chờ cây trước xong mới luyện cây sau).
-   Khó tinh chỉnh siêu tham số hơn Random Forest.

---

## 6. Những nội dung cần lưu ý

1.  **Shrinkage (Learning Rate):** Việc sử dụng tốc độ học nhỏ (ví dụ 0.01) kết hợp với số lượng cây lớn thường giúp mô hình đạt khả năng tổng quát hóa tốt hơn.
2.  **Early Stopping:** Nên dừng huấn luyện khi sai số trên tập Kiểm chứng (Validation) không còn giảm nữa để tránh Overfitting.
3.  **Hậu duệ mạnh mẽ:** Gradient Boosting là nền tảng cho các thuật toán "vô địch" hiện nay như **XGBoost, LightGBM và CatBoost**.

---
