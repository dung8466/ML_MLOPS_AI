---

# Hồi quy Tuyến tính (Linear Regression) từ con số 0

Hồi quy Tuyến tính là một trong những thuật toán cơ bản và quan trọng nhất trong Machine Learning. Mục tiêu của nó là thiết lập mối quan hệ giữa một biến phụ thuộc (nhãn - $y$) và một hoặc nhiều biến độc lập (đặc trưng - $x$).

## 1. Mô hình toán học (Mathematical Model)

Giả sử chúng ta có một bài toán hồi quy đơn biến (chỉ có một đặc trưng $x$). Mô hình dự đoán có dạng:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Trong đó:
- $h_\theta(x)$: Giá trị dự đoán.
- $\theta_0$: Hệ số chặn (intercept/bias).
- $\theta_1$: Hệ số góc (slope/weight).

Đối với bài toán đa biến:
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T X$$

---

## 2. Hàm mất mát (Cost Function)

Để đo lường xem mô hình dự đoán chính xác đến mức nào, chúng ta sử dụng hàm **Sai số bình phương trung bình (Mean Squared Error - MSE)**.

**Công thức:**
$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Trong đó:
- $m$: Tổng số ví dụ trong tập dữ liệu.
- $1/2$: Được thêm vào để giúp việc tính đạo hàm sau này gọn hơn.

Mục tiêu của chúng ta là tìm bộ tham số $\theta$ sao cho $J(\theta)$ đạt giá trị **nhỏ nhất**.

---

## 3. Thuật toán tối ưu: Gradient Descent

Gradient Descent là một thuật toán lặp để tìm cực tiểu của hàm mất mát. Chúng ta cập nhật các tham số $\theta$ từng bước một cho đến khi hội tụ.

**Quy tắc cập nhật:**
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Sau khi tính đạo hàm, công thức cập nhật cho hồi quy tuyến tính là:
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

Trong đó:
- $\alpha$: Tốc độ học (learning rate). Nếu quá lớn, thuật toán sẽ không hội tụ. Nếu quá nhỏ, thuật toán sẽ chạy rất chậm.

---

## 4. Triển khai bằng Python (NumPy)

Dưới đây là cách xây dựng lớp `LinearRegression` từ đầu:

```python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Khởi tạo tham số bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Vòng lặp Gradient Descent
        for _ in range(self.iterations):
            # Tính mô hình dự đoán (h_theta = X*w + b)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Tính toán Gradient (đạo hàm)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Cập nhật tham số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

---

## 5. Trực quan hóa kết quả

Khi thực hiện hồi quy tuyến tính đơn biến, kết quả thu được là một đường thẳng đi qua các điểm dữ liệu sao cho tổng khoảng cách (bình phương) từ các điểm đến đường thẳng là nhỏ nhất.

- **Đường màu đỏ:** Đường thẳng dự đoán ($y = wx + b$).
- **Các điểm xanh:** Dữ liệu thực tế.

```python
import matplotlib.pyplot as plt

# Giả sử đã huấn luyện xong model
y_pred_line = model.predict(X)
plt.scatter(X, y, color="blue", s=30)
plt.plot(X, y_pred_line, color="red", linewidth=2, label="Prediction")
plt.xlabel("Đặc trưng X")
plt.ylabel("Nhãn Y")
plt.show()
```

---

## 6. Những điểm cần lưu ý khi học

1.  **Chuẩn hóa dữ liệu (Feature Scaling):** Gradient Descent sẽ hội tụ nhanh hơn nhiều nếu các đặc trưng có cùng quy mô (ví dụ: đưa về khoảng [0, 1]).
2.  **Lựa chọn Learning Rate ($\alpha$):** Đây là bước quan trọng nhất. Cần thử các giá trị như 0.001, 0.01, 0.1 để xem giá trị nào làm hàm Loss giảm ổn định nhất.
3.  **Vector hóa (Vectorization):** Sử dụng các phép toán ma trận của NumPy (`np.dot`) thay vì dùng vòng lặp `for` trên từng ví dụ để tăng tốc độ tính toán hàng trăm lần.
4.  **Giải pháp đóng (Normal Equation):** Ngoài Gradient Descent, còn có một phương pháp toán học để tính ngay ra $\theta$ tối ưu mà không cần lặp, nhưng nó chỉ hiệu quả với tập dữ liệu nhỏ.

---
*Tóm tắt này giúp bạn hiểu từ lý thuyết toán học đến cách hiện thực hóa mã nguồn cho mô hình máy học cơ bản nhất.*
