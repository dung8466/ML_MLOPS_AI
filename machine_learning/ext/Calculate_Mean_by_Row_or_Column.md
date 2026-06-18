# Lý thuyết

---

# Tính giá trị trung bình theo Hàng hoặc Cột (Calculate Mean by Row or Column)

Tính toán giá trị trung bình của một ma trận theo hàng hoặc theo cột bao gồm việc lấy trung bình cộng của các phần tử dọc theo một chiều (dimension) cụ thể. Thao tác này cung cấp cái nhìn sâu sắc về sự phân phối của các giá trị trong tập dữ liệu, rất hữu ích cho việc chuẩn hóa (normalization) và thay đổi quy mô dữ liệu (scaling).

### Giá trị trung bình theo hàng (Row Mean)

Giá trị trung bình của một hàng được tính bằng cách tổng tất cả các phần tử trong hàng đó và chia cho số lượng phần tử. Đối với hàng thứ $i$, giá trị trung bình là:

$$\mu_{\text{row } i} = \frac{1}{n} \sum_{j=1}^{n} a_{ij}$$

Trong đó:
*   $a_{ij}$: là phần tử của ma trận tại hàng thứ $i$ và cột thứ $j$.
*   $n$: là tổng số cột.

### Giá trị trung bình theo cột (Column Mean)

Tương tự, giá trị trung bình của một cột được tìm bằng cách tổng tất cả các phần tử trong cột đó và chia cho số lượng phần tử. Đối với cột thứ $j$, giá trị trung bình là:

$$\mu_{\text{column } j} = \frac{1}{m} \sum_{i=1}^{m} a_{ij}$$

Trong đó:
*   $m$: là tổng số hàng.

---

### Tầm quan trọng:
Các công thức toán học này giúp chúng ta hiểu cách dữ liệu được gộp lại (aggregated) theo các chiều khác nhau. Đây là một bước then chốt trong nhiều kỹ thuật tiền xử lý dữ liệu trước khi đưa vào mô hình học máy.

---
**💡 Mẹo cho ML:** Trong thư viện **NumPy**, bạn có thể tính toán các giá trị này cực kỳ nhanh chóng:
*   Tính theo cột: `np.mean(matrix, axis=0)`
*   Tính theo hàng: `np.mean(matrix, axis=1)`

---

# Bài tập

Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.
Example:
Input:

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'

Output:

    [4.0, 5.0, 6.0]

Reasoning:

    Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].

# Code

```python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
  if mode == "column":
		means = [sum(col) / len(col) for col in zip(*matrix)]
	elif mode == "row":
		means = [sum(row) / len(row) for row in matrix]
	else:
		return -1
	return means
```
