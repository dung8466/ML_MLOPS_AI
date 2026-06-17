---

# Hồi quy Logistic (Logistic Regression) từ con số 0

Hồi quy Logistic là một thuật toán phân loại (classification) được sử dụng để dự đoán xác suất của một biến mục tiêu rời rạc. Mặc dù có tên là "hồi quy", nhưng nó thực chất được dùng cho các bài toán phân loại nhị phân (0 hoặc 1).

## 1. Hàm Sigmoid (Sigmoid Function)

Để chuyển đổi đầu ra của một phương trình tuyến tính thành xác suất (giá trị từ 0 đến 1), chúng ta sử dụng hàm Sigmoid.

**Công thức:**
$$g(z) = \frac{1}{1 + e^{-z}}$$

Trong đó $z = \theta^T X$ (tổng trọng số của các đặc trưng).

**Triển khai bằng Python:**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

## 2. Hàm giả thuyết (Hypothesis Function)

Hàm giả thuyết đại diện cho xác suất mà đầu ra là 1 cho một đầu vào $x$ nhất định.

**Công thức:**
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Ý nghĩa:
- Nếu $h_\theta(x) \geq 0.5$, dự đoán nhãn là 1 ($y=1$).
- Nếu $h_\theta(x) < 0.5$, dự đoán nhãn là 0 ($y=0$).

---

## 3. Hàm mất mát (Cost Function)

Chúng ta không thể sử dụng hàm Mean Squared Error (MSE) cho hồi quy Logistic vì nó sẽ dẫn đến một hàm không lồi (non-convex). Thay vào đó, chúng ta sử dụng **Log Loss**.

**Công thức cho một ví dụ đơn lẻ:**
- Nếu $y = 1$: $Cost(h_\theta(x), y) = -\log(h_\theta(x))$
- Nếu $y = 0$: $Cost(h_\theta(x), y) = -\log(1 - h_\theta(x))$

**Công thức tổng quát cho toàn bộ tập dữ liệu:**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$$

---

## 4. Thuật toán tối ưu: Gradient Descent

Để tìm các tham số $\theta$ tối ưu nhằm giảm thiểu hàm mất mát $J(\theta)$, chúng ta cập nhật trọng số theo hướng ngược lại của gradient.

**Quy tắc cập nhật:**
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

Trong đó $\alpha$ là tốc độ học (learning rate).

---

## 5. Triển khai mô hình Logistic Regression

Dưới đây là lớp `LogisticRegression` được xây dựng bằng NumPy:

```python
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # Khởi tạo trọng số bằng 0
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if i % 10000 == 0:
                print(f'Loss: {self.__loss(h, y)}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
```

---

## 6. Biên quyết định (Decision Boundary)

Biên quyết định là ranh giới phân tách các lớp dữ liệu. Trong hồi quy Logistic với hai đặc trưng, biên quyết định là một đường thẳng nơi mà:
$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$$

Chúng ta có thể vẽ đường này để trực quan hóa cách mô hình phân loại dữ liệu:
- Điểm nằm phía trên đường thẳng dự đoán là lớp 1.
- Điểm nằm phía dưới đường thẳng dự đoán là lớp 0.

---

## 7. Đánh giá mô hình

Sau khi huấn luyện, chúng ta kiểm tra độ chính xác (Accuracy) bằng cách so sánh nhãn dự đoán với nhãn thực tế:

```python
# Giả sử model là thực thể của LogisticRegression
preds = model.predict(X)
accuracy = (preds == y).mean()
print(f'Accuracy: {accuracy}')
```

### Các nội dung cần nhớ để học ML:
1. **Tiền xử lý:** Luôn đảm bảo đặc trưng đã được chuẩn hóa.
2. **Hàm kích hoạt:** Hiểu cách hàm Sigmoid bóp nghẹt giá trị về khoảng xác suất.
3. **Log Loss:** Hiểu tại sao hàm này phạt nặng các dự đoán sai mà mô hình lại quá tự tin.
4. **Tham số:** Learning rate ($\alpha$) và số vòng lặp (iterations) ảnh hưởng trực tiếp đến sự hội tụ.
