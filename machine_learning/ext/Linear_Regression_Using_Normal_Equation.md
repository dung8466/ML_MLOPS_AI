# Lý thuyết

---

## Hồi quy Tuyến tính sử dụng Phương trình chuẩn (Normal Equation)

Hồi quy tuyến tính nhằm mục tiêu mô hình hóa mối quan hệ giữa một biến phụ thuộc vô hướng ($y$) và một hoặc nhiều biến giải thích (hay còn gọi là biến độc lập) ($X$). Phương trình chuẩn cung cấp một lời giải giải tích (analytical solution) để tìm ra các hệ số ($\theta$) giúp tối thiểu hóa hàm mất mát (cost function) trong hồi quy tuyến tính.

Cho một ma trận $X$ (với mỗi hàng đại diện cho một ví dụ huấn luyện và mỗi cột đại diện cho một đặc trưng) và một vector $y$ (chứa các giá trị mục tiêu), phương trình chuẩn được xác định như sau:

$$\theta = (X^T X)^{-1} X^T y$$

### Giải thích các thuật ngữ:

*   **$X^T$**: Là ma trận chuyển vị của ma trận $X$.
*   **$(X^T X)^{-1}$**: Là ma trận nghịch đảo của ma trận $(X^T X)$.
*   **$y$**: Là vector chứa các giá trị mục tiêu.

### Các điểm mấu chốt:

*   **Chuẩn hóa đặc trưng (Feature Scaling):** Phương pháp này không yêu cầu phải chuẩn hóa các đặc trưng.
*   **Tốc độ học (Learning Rate):** Không cần phải lựa chọn tốc độ học ($\alpha$).
*   **Chi phí tính toán:** Việc tính toán ma trận nghịch đảo của $(X^T X)$ có thể rất tốn kém tài nguyên máy tính nếu số lượng đặc trưng ($n$) là rất lớn (thông thường khi $n > 10,000$).

---

# Bài toán

Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

Example:

Input:

    X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]

Output:

    [0.0, 1.0]

# Code python

```
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
  	X = np.array(X)
  	y = np.array(y)
	X_T = X.T
	theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
	theta = np.round(theta,4).flatten().tolist()
	return theta
```
