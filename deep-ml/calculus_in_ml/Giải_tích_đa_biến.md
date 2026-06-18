# Các nội dung cốt lõi của Multivariate Calculus trong ML

### 1. Hàm nhiều biến (Functions of Multiple Variables)
Trong ML, hàm mất mát không chỉ phụ thuộc vào một biến mà phụ thuộc vào hàng ngàn hoặc hàng triệu tham số (trọng số).
*   **Ký hiệu:** $f(x_1, x_2, \dots, x_n)$ hoặc $f(\mathbf{w})$ với $\mathbf{w}$ là một vector.
*   **Ý nghĩa:** Đại diện cho bề mặt sai số (Loss surface) trong không gian đa chiều.

---

### 2. Đạo hàm riêng (Partial Derivatives)
Đạo hàm riêng cho biết hàm số thay đổi như thế nào khi **chỉ có một biến** thay đổi, trong khi các biến khác được giữ nguyên.
*   **Ký hiệu:** $\frac{\partial f}{\partial x_i}$
*   **Ý nghĩa:** Trong ML, đây là cách ta tính xem nếu chỉ thay đổi một trọng số $w_i$ thì sai số của mô hình sẽ tăng hay giảm bao nhiêu.

---

### 3. Vector Gradient ($\nabla$)
Gradient là một vector chứa tất cả các đạo hàm riêng của hàm số.
*   **Công thức:**
    $$\nabla f(\mathbf{w}) = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_n} \right]^T$$
*   **Ý nghĩa:** Gradient luôn chỉ về hướng mà hàm số **tăng nhanh nhất**. Hướng ngược lại ($-\nabla f$) là hướng **giảm nhanh nhất**, được dùng trong thuật toán Gradient Descent.

---

### 4. Quy tắc chuỗi (Chain Rule)
Đây là kiến thức quan trọng nhất để hiểu **Backpropagation** (Lan truyền ngược) trong Neural Networks. Quy tắc này cho phép tính đạo hàm của các hàm lồng nhau.
*   **Công thức đơn giản:** Nếu $z = f(y)$ và $y = g(x)$, thì:
    $$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$
*   **Trong ML:** Giúp truyền sai số từ lớp đầu ra (Output) quay ngược về các lớp ẩn (Hidden layers) để cập nhật trọng số.

---

### 5. Ma trận Jacobian (Jacobian Matrix)
Khi bạn có một hàm trả về nhiều đầu ra (ví dụ: lớp Softmax trả về xác suất của nhiều lớp), ta dùng ma trận Jacobian.
*   **Định nghĩa:** Là ma trận chứa tất cả các đạo hàm riêng bậc nhất của một hàm vector.
*   **Công thức:** $J_{ij} = \frac{\partial f_i}{\partial x_j}$

---

### 6. Ma trận Hessian (Hessian Matrix)
Hessian chứa các đạo hàm riêng **bậc hai**, cho biết độ cong của hàm mất mát.
*   **Công thức:** $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$
*   **Ý nghĩa:** 
    *   Giúp xác định điểm cực tiểu, cực đại hoặc điểm yên ngựa (Saddle point).
    *   Được dùng trong các thuật toán tối ưu bậc cao (như Newton's Method) để hội tụ nhanh hơn Gradient Descent thông thường.

---

### 7. Khai triển Taylor (Taylor Series)
Dùng để xấp xỉ một hàm số phức tạp bằng một hàm đa thức đơn giản hơn tại một điểm cụ thể.
*   **Ứng dụng:** Hầu hết các thuật toán tối ưu hóa trong ML (như XGBoost) đều dựa trên khai triển Taylor bậc hai để tìm hướng đi tốt nhất cho các cây quyết định.

---

### Ứng dụng trong ML:
1.  **Gradient Descent:** Dùng Gradient để biết hướng đi xuống đáy của hàm Loss.
2.  **Backpropagation:** Dùng Chain Rule để tính toán sự thay đổi cần thiết cho từng nơ-ron.
3.  **Optimization:** Dùng Hessian để hiểu về độ dốc và độ cong, giúp tránh các vùng phẳng hoặc điểm yên ngựa.

