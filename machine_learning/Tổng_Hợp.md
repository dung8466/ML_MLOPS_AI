# Machine Learning: Từ Bản Chất Toán Học Đến Thực Thi Code

## Mục lục
1. [Fundamentals & Preprocessing (Nền tảng & Tiền xử lý)](#1-fundamentals--preprocessing)
2. [Linear Algebra (Đại số tuyến tính)](#2-linear-algebra)
3. [Linear Regression (Hồi quy tuyến tính)](#3-linear-regression)
4. [Logistic Regression (Hồi quy Logistic)](#4-logistic-regression)
5. [Regularization (Điều quy hóa)](#5-regularization)
6. [K-Nearest Neighbors (K hàng xóm gần nhất)](#6-k-nearest-neighbors)
7. [Naïve Bayes (Phân loại Bayes ngây thơ)](#7-naive-bayes)
8. [Decision Tree (Cây quyết định)](#8-decision-tree)
9. [Random Forest (Rừng ngẫu nhiên)](#9-random-forest)
10. [Neural Network (Mạng nơ-ron)](#10-neural-network)
11. [Evaluation Metrics (Chỉ số đánh giá)](#11-evaluation-metrics)

---

## 1. Fundamentals & Preprocessing (Nền tảng & Tiền xử lý)

** Hiểu Toán trước:**
Standardization giúp đưa mọi đặc trưng về cùng một thang đo ($z = \frac{x - \mu}{\sigma}$). Nếu không làm bước này, các đặc trưng có giá trị lớn (như thu nhập) sẽ áp đảo các đặc trưng nhỏ (như số phòng ngủ), làm mô hình học sai lệch.

** Python thuần:**
```python
def standardize(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    std_dev = variance**0.5
    return [(x - mean) / std_dev for x in data]
```

** Thư viện (Scikit-learn):**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

---

## 2. Linear Algebra (Đại số tuyến tính)

** Hiểu Toán trước:**
Tích vô hướng ($W \cdot X$) là phép toán nhân từng cặp giá trị đầu vào với trọng số tương ứng rồi cộng dồn lại. Đây là cách mô hình tổng hợp mọi bằng chứng để đưa ra một con số dự báo cuối cùng.

** Python thuần:**
```python
def matrix_dot_vector(A, v):
    # A là list of lists, v là list
    return [sum(a * b for a, b in zip(row, v)) for row in A]
```

** Thư viện (NumPy):**
```python
import numpy as np
result = np.dot(A, v) # Hoặc A @ v
```

---

## 3. Linear Regression (Hồi quy tuyến tính)

** Hiểu Toán trước:**
MSE ($J = \frac{1}{2m} \sum (y' - y)^2$) phạt nặng các lỗi lớn bằng cách bình phương sai số. Phương trình chuẩn (Normal Equation: $\theta = (X^T X)^{-1} X^T y$) cho phép tìm ngay kết quả tối ưu mà không cần lặp, nhưng sẽ rất chậm nếu dữ liệu có quá nhiều đặc trưng.

** Python thuần (Gradient Descent):**
```python
def update_weights(w, b, X, y, lr):
    dw, db, m = 0, 0, len(y)
    for i in range(m):
        error = (w * X[i] + b) - y[i]
        dw += error * X[i]
        db += error
    w -= lr * (dw / m)
    b -= lr * (db / m)
    return w, b
```

** Thư viện (Scikit-learn):**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
predictions = model.predict(X_new)
```

---

## 4. Logistic Regression (Hồi quy Logistic)

** Hiểu Toán trước:**
Hàm Sigmoid ép kết quả về khoảng $(0, 1)$ để đại diện cho xác suất. Log Loss được dùng thay cho MSE vì nó tạo ra hàm lỗi lồi, đảm bảo Gradient Descent luôn tìm được điểm tối ưu mà không bị kẹt.

** Python thuần:**
```python
import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def log_loss(y_true, y_pred):
    # y_pred là xác suất từ sigmoid
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))
```

** Thư viện (Scikit-learn):**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X, y)
```

---

## 5. Regularization (Điều quy hóa)

** Hiểu Toán trước:**
L1 (Lasso) dùng trị tuyệt đối $|\theta|$ có đạo hàm là hằng số, giúp triệt tiêu các trọng số rác về đúng bằng 0. L2 (Ridge) dùng $\theta^2$ giúp trọng số nhỏ đi nhưng vẫn giữ lại đặc trưng. Weight Decay là cách triển khai L2 trực tiếp vào bước cập nhật trọng số.

** Python thuần (L2 Update):**
```python
def update_with_l2(w, grad, lr, lamb):
    # w = w - lr * (gradient + lambda * w)
    return w * (1 - lr * lamb) - lr * grad
```

** Thư viện (Scikit-learn):**
```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0).fit(X, y) # L2
lasso = Lasso(alpha=0.1).fit(X, y) # L1
```

---

## 6. K-Nearest Neighbors (K hàng xóm gần nhất)

** Hiểu Toán trước:**
KNN là thuật toán dựa trên khoảng cách (thường là Euclidean). Nó giả định rằng các điểm dữ liệu gần nhau trong không gian đa chiều sẽ có nhãn giống nhau. Đây là thuật toán "Học lười" vì nó không có bước huấn luyện, chỉ lục lại trí nhớ khi dự đoán.

** Python thuần:**
```python
def euclidean_dist(p1, p2):
    return sum((a - b)**2 for a, b in zip(p1, p2))**0.5

# Dự đoán: Tính dist đến mọi điểm -> Sort -> Lấy K điểm đầu -> Chọn nhãn đa số
```

** Thư viện (Scikit-learn):**
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
```

---

## 7. Naïve Bayes (Phân loại Bayes ngây thơ)

** Hiểu Toán trước:**
Dựa trên định lý Bayes ($P(A|B) = \frac{P(B|A)P(A)}{P(B)}$). Chữ "Ngây thơ" đến từ việc giả định các đặc trưng hoàn toàn độc lập với nhau (ví dụ: từ "Rẻ" và từ "Khuyến mãi" không liên quan đến nhau), giúp tính xác suất cực nhanh.

** Python thuần:**
```python
# Xác suất hậu nghiệm = Xác suất tiên nghiệm * Khả năng xảy ra (Likelihood)
# P(Lớp|Dữ liệu) ~ P(Lớp) * P(Đặc trưng 1|Lớp) * P(Đặc trưng 2|Lớp)...
```

** Thư viện (Scikit-learn):**
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X, y)
```

---

## 8. Decision Tree (Cây quyết định)

** Hiểu Toán trước:**
Cây quyết định sử dụng Entropy để đo độ hỗn loạn. Mục tiêu là đặt ra các câu hỏi If-Then sao cho sau mỗi lần chia, dữ liệu trở nên "thuần khiết" hơn (Information Gain lớn nhất).

** Python thuần:**
```python
def entropy(labels):
    from collections import Counter
    counts = Counter(labels).values()
    probs = [c / len(labels) for c in counts]
    return -sum(p * math.log2(p) for p in probs if p > 0)
```

** Thư viện (Scikit-learn):**
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=10).fit(X, y)
```

---

## 9. Random Forest (Rừng ngẫu nhiên)

** Hiểu Toán trước:**
Sử dụng kỹ thuật Bagging (Bootstrap Aggregating). Bằng cách kết hợp hàng trăm cây quyết định học trên các mẫu dữ liệu khác nhau, Random Forest làm giảm sai số do nhiễu (Variance) của một cây đơn lẻ, tạo nên một "trí tuệ tập thể" ổn định.

** Python thuần:**
```python
# 1. Lấy mẫu ngẫu nhiên (có lặp lại) từ dữ liệu gốc
# 2. Huấn luyện một cây trên mỗi mẫu đó
# 3. Lấy kết quả dự đoán bằng cách biểu quyết (voting)
```

** Thư viện (Scikit-learn):**
```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100).fit(X, y)
```

---

## 10. Neural Network (Mạng nơ-ron)

** Hiểu Toán trước:**
Backpropagation là trái tim của mạng nơ-ron, thực chất là ứng dụng của Quy tắc chuỗi (Chain Rule). Nó truyền sai số ngược từ đầu ra về từng trọng số ở các lớp ẩn để "truy cứu trách nhiệm" và cập nhật giá trị.

** Python thuần (Forward pass):**
```python
def forward_pass(X, W, b):
    z = sum(xi * wi for xi, wi in zip(X, W)) + b
    return 1 / (1 + math.exp(-z)) # Activation
```

** Thư viện (Keras/TensorFlow):**
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## 11. Evaluation Metrics (Chỉ số đánh giá)

** Hiểu Toán trước:**
Accuracy thường gây hiểu lầm trong dữ liệu lệch. Confusion Matrix giúp ta tính được Precision (tránh báo động nhầm) và Recall (tránh bỏ sót trường hợp đúng).

** Python thuần:**
```python
# Accuracy = (TP + TN) / Tổng số mẫu
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
```

** Thư viện (Scikit-learn):**
```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

---
