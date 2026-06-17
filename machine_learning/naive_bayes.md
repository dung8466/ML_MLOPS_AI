
# Naïve Bayes từ con số 0

Naïve Bayes là một thuật toán **Học có giám sát (Supervised Learning)** dựa trên xác suất, được xây dựng từ định lý Bayes. Đây là thuật toán cực kỳ phổ biến trong các bài toán phân loại văn bản, lọc thư rác (spam) và phân tích cảm xúc.

## 1. Khái niệm cơ bản

Mục tiêu của Naïve Bayes là tính xác suất để một điểm dữ liệu thuộc về một lớp cụ thể, sau đó chọn lớp có xác suất cao nhất.

**Giả định "Ngây thơ" (Naïve):**
Thuật toán giả định rằng tất cả các đặc trưng (features) là **độc lập** với nhau khi biết nhãn của lớp. Dù thực tế điều này hiếm khi đúng, nhưng giả định này giúp việc tính toán trở nên cực kỳ đơn giản và nhanh chóng.

**Các bước thực hiện:**
1. **Huấn luyện:** Tính toán xác suất tiên nghiệm (Prior) của mỗi lớp và xác suất có điều kiện (Likelihood) của từng đặc trưng dựa trên dữ liệu cũ.
2. **Dự đoán:** Với một dữ liệu mới, áp dụng định lý Bayes để tính xác suất của từng lớp.
3. **Kết luận:** Chọn lớp có xác suất hậu nghiệm (Posterior) lớn nhất.

---

## 2. Công thức toán học (Định lý Bayes)

Để dự đoán lớp $y$ cho tập đặc trưng $X = (x_1, x_2, \dots, x_n)$, ta dùng công thức:

$$P(y \mid x_1, \dots, x_n) = \frac{P(x_1, \dots, x_n \mid y) P(y)}{P(x_1, \dots, x_n)}$$

Nhờ giả định độc lập, phần tử Likelihood được tính bằng tích các xác suất thành phần:
$$P(y \mid X) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

---

## 3. Triển khai thuật toán bằng Python (Gaussian NB)

Với dữ liệu số liên tục, chúng ta thường dùng **Gaussian Naïve Bayes**, giả định các đặc trưng tuân theo phân phối chuẩn (phân phối Gauss).

```python
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Khởi tạo trung bình, phương sai và xác suất tiên nghiệm
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self.__predict(x) for x in X]
        return np.array(y_pred)

    def __predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            # Tính log xác suất để tránh hiện tượng số quá nhỏ (underflow)
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # Hàm mật độ xác suất của phân phối Gauss
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
```

---

## 4. Các loại Naïve Bayes phổ biến

- **Gaussian NB:** Dùng khi đặc trưng là dữ liệu số liên tục (ví dụ: chiều cao, cân nặng).
- **Multinomial NB:** Dùng cho dữ liệu đếm tần suất (ví dụ: số lần xuất hiện của từ trong văn bản).
- **Bernoulli NB:** Dùng khi đặc trưng chỉ có 2 giá trị (ví dụ: từ có xuất hiện hay không).

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
- Tốc độ cực nhanh cả khi huấn luyện và dự đoán.
- Hoạt động tốt với tập dữ liệu có ít mẫu.
- Dễ dàng mở rộng với dữ liệu có hàng triệu đặc trưng.

**Nhược điểm:**
- Giả định độc lập giữa các đặc trưng thường không thực tế.
- **Vấn đề tần suất bằng 0:** Nếu một giá trị đặc trưng chưa từng xuất hiện trong tập huấn luyện, xác suất sẽ bằng 0, làm mất hết thông tin của các đặc trưng khác.

---

## 6. Những nội dung cần lưu ý

1. **Làm trơn Laplace (Laplace Smoothing):** Để giải quyết vấn đề tần suất bằng 0, chúng ta thường cộng thêm một hằng số nhỏ ($\alpha=1$) vào các phép đếm xác suất.
2. **Xử lý số quá nhỏ (Log Transformation):** Việc nhân nhiều xác suất nhỏ (0 đến 1) sẽ dẫn đến số cực nhỏ khiến máy tính bị lỗi. Người ta thường chuyển sang tính tổng của các $\log(P)$ để ổn định hơn.
3. **Phân phối dữ liệu:** Trước khi dùng Gaussian NB, hãy kiểm tra xem dữ liệu có dạng hình chuông (chuẩn) hay không. Nếu không, kết quả có thể kém chính xác.

---
