
# Mạng nơ-ron (Neural Network)

Mạng nơ-ron (NN) là nền tảng của Deep Learning, được truyền cảm hứng từ cấu trúc các tế bào thần kinh trong não người. Nó là một bộ xấp xỉ hàm vạn năng, có khả năng học được những mối quan hệ cực kỳ phức tạp và phi tuyến tính trong dữ liệu.

## 1. Khái niệm cơ bản và Cấu trúc

Một mạng nơ-ron cơ bản (Multi-Layer Perceptron - MLP) gồm 3 thành phần chính:
1.  **Input Layer (Lớp đầu vào):** Tiếp nhận dữ liệu đặc trưng ($x$).
2.  **Hidden Layers (Lớp ẩn):** Nơi diễn ra các tính toán phức tạp. Càng nhiều lớp ẩn, mạng càng "sâu".
3.  **Output Layer (Lớp đầu ra):** Đưa ra kết quả dự báo ($\hat{y}$).

**Mỗi nơ-ron hoạt động như thế nào?**
Tại mỗi nơ-ron, hai phép toán sau diễn ra liên tiếp:
-   **Tính tổng có trọng số:** $z = \sum (w_i x_i) + b$
-   **Kích hoạt phi tuyến:** $a = \sigma(z)$ (Sử dụng hàm kích hoạt để mô hình học được các đường cong).

---

## 2. Các hàm kích hoạt (Activation Functions) phổ biến

Nếu không có hàm kích hoạt, mạng nơ-ron chỉ là một tổ hợp các phép toán tuyến tính (như Linear Regression), không thể học được dữ liệu phức tạp.

-   **ReLU (Rectified Linear Unit):** Phổ biến nhất cho lớp ẩn. Công thức: $f(x) = \max(0, x)$. Giúp tránh triệt tiêu đạo hàm.
-   **Sigmoid:** Dùng cho bài toán phân loại nhị phân ở đầu ra. Ép giá trị về $[0, 1]$.
-   **Softmax:** Dùng cho bài toán phân loại đa lớp ở đầu ra. Tổng các xác suất bằng $1$.

---

## 3. Quy trình huấn luyện (The Learning Process)

Quá trình học của mạng nơ-ron diễn ra qua 2 bước lặp đi lặp lại:

### A. Lan truyền xuôi (Forward Propagation)
Dữ liệu đi từ lớp đầu vào qua các lớp ẩn để đến đầu ra. Tại mỗi lớp:
$$Z^{[l]} = A^{[l-1]} \cdot W^{[l]} + b^{[l]}$$
$$A^{[l]} = \sigma(Z^{[l]})$$

### B. Lan truyền ngược (Backpropagation) - Trái tim của NN
Sau khi tính được sai số (Loss) ở đầu ra, chúng ta dùng **Quy tắc chuỗi (Chain Rule)** trong đạo hàm để tính xem mỗi trọng số ($w$) đóng góp bao nhiêu vào sai số đó.
-   Tính Gradient: $\frac{\partial Loss}{\partial W}$ và $\frac{\partial Loss}{\partial b}$.
-   Cập nhật trọng số: $W = W - \eta \cdot \frac{\partial Loss}{\partial W}$ (với $\eta$ là Learning Rate).

---

## 4. Triển khai logic bằng Python (NumPy)

Dưới đây là cấu trúc đơn giản của một lớp trong Mạng nơ-ron:

```python
import numpy as np

class NeuralLayer:
    def __init__(self, n_inputs, n_neurons):
        # Khởi tạo trọng số ngẫu nhiên (tránh bằng 0 để phá vỡ tính đối xứng)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, learning_rate):
        # dvalues là đạo hàm từ lớp phía sau truyền về
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        # Cập nhật trọng số
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        return self.dinputs
```

---

## 5. Ưu điểm và Nhược điểm

**Ưu điểm:**
-   Cực kỳ mạnh mẽ với dữ liệu phi cấu trúc (ảnh, âm thanh, văn bản).
-   Không cần trích xuất đặc trưng thủ công (Feature Engineering) nhiều như các thuật toán cũ.
-   Linh hoạt: Có thể tùy biến cấu trúc (số lớp, số nơ-ron) cho mọi loại bài toán.

**Nhược điểm:**
-   **Cần rất nhiều dữ liệu:** NN thường hoạt động kém nếu tập dữ liệu nhỏ.
-   **Chi phí tính toán cao:** Cần GPU để huấn luyện các mạng lớn.
-   **Hộp đen (Black Box):** Rất khó giải thích tại sao mạng lại đưa ra dự đoán đó.
-   **Dễ Overfitting:** Cần các kỹ thuật như Dropout hoặc Weight Decay để kiểm soát.

---

## 6. Những nội dung cần lưu ý

1.  **Chuẩn hóa đầu vào (Input Scaling):** Bắt buộc phải đưa dữ liệu đầu vào về cùng quy mô (ví dụ $[0, 1]$) để tránh hiện tượng nổ/triệt tiêu gradient.
2.  **Khởi tạo trọng số (Weight Initialization):** Không nên khởi tạo bằng 0. Nên dùng các kỹ thuật như **Xavier** hoặc **He initialization** tùy thuộc vào hàm kích hoạt.
3.  **Bộ tối ưu (Optimizers):** Ngoài SGD cơ bản, trong thực tế người ta dùng **Adam** hoặc **RMSprop** để tự động điều chỉnh tốc độ học, giúp hội tụ nhanh hơn.
4.  **Vanishing Gradient (Triệt tiêu đạo hàm):** Khi mạng quá sâu, đạo hàm có thể tiến về 0 khiến các lớp đầu tiên không học được gì. Sử dụng hàm **ReLU** và **Batch Normalization** là cách giải quyết phổ biến.

---
