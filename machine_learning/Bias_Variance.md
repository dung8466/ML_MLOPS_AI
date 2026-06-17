---

# Sự đánh đổi Bias - Variance và Hồi quy Đa thức (Polynomial Regression)

Mục tiêu của bài viết này là khám phá mối quan hệ giữa hai thuộc tính cạnh tranh nhau trong một mô hình học thống kê: **Bias (Độ chệch)** và **Variance (Phương sai)**.

### 1. Khái niệm cơ bản
Một khái niệm quan trọng trong Machine Learning là sự đánh đổi **Bias-Variance**. 
*   Các mô hình có **Bias cao** thường không đủ phức tạp để nắm bắt quy luật dữ liệu, dẫn đến hiện tượng **Underfitting** (Dưới khớp). 
*   Ngược lại, các mô hình có **Variance cao** thường quá linh hoạt, dẫn đến hiện tượng **Overfitting** (Quá khớp) vào dữ liệu huấn luyện.

**Variance (Phương sai):** Là mức độ thay đổi của mô hình khi chúng ta huấn luyện nó trên các tập dữ liệu khác nhau. Một mô hình lý tưởng không nên thay đổi quá đáng kể khi dữ liệu đầu vào thay đổi nhẹ. Nếu thay đổi quá lớn, mô hình được gọi là có *High Variance*.

**Bias (Độ chệch):** Là sai số phát sinh khi chúng ta cố gắng mô hình hóa một vấn đề thực tế phức tạp bằng một giả định quá đơn giản.

**Mục tiêu:** Tạo ra một mô hình tìm được điểm cân bằng (**Tradeoff**) giữa Bias và Variance. Theo quy tắc chung, khi tăng tính linh hoạt của mô hình, Variance sẽ tăng và Bias sẽ giảm.

---

### 2. Ví dụ thực tế: Dự đoán lượng nước xả khỏi đập
Chúng ta sẽ xây dựng một mô hình để dự báo lượng nước chảy ra khỏi một con đập dựa trên sự thay đổi mực nước của hồ chứa. Chúng ta sẽ bắt đầu với **Hồi quy tuyến tính (Linear Regression)** bằng một đường thẳng, sau đó mở rộng không gian đặc trưng bằng các **Đa thức bậc p**.

#### Công thức toán học (Hàm mất mát Regularized):
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$

---

### 3. Mô hình High Bias (Độ chệch cao)
Ở giai đoạn đầu, chúng ta chạy Hồi quy tuyến tính thuần túy với $\lambda = 0$ (không điều quy hóa) trên một tập dữ liệu có dạng đường cong.

*   **Kết quả:** Đường thẳng thu được không thể biểu diễn được mối quan hệ thực tế giữa mực nước và lưu lượng. 
*   **Chẩn đoán:** Mô hình không đủ linh hoạt $\rightarrow$ **High Bias** và **Low Variance**.

#### Đồ thị đường cong học tập (Learning Curve):
Khi quan sát biểu đồ Learning Curve (Sai số tập Huấn luyện vs. Sai số tập Kiểm chứng theo số lượng mẫu):
*   Cả sai số huấn luyện và sai số kiểm chứng đều dừng lại ở mức **rất cao**.
*   Điều này khẳng định mô hình quá đơn giản (Underfitting).

---

### 4. Hồi quy Đa thức (Polynomial Regression)
Để giải quyết vấn đề High Bias, chúng ta thêm các đặc trưng mới là lũy thừa của mực nước:
$$h_\theta(x) = \theta_0 + \theta_1(x) + \theta_2(x)^2 + \dots + \theta_p(x)^p$$

**Lưu ý quan trọng:** 
1.  **Feature Mapping:** Biến một đặc trưng $x$ thành vector $[x, x^2, \dots, x^p]$.
2.  **Chuẩn hóa đặc trưng (Feature Normalization):** Vì $x^p$ có thể cực lớn so với $x$, chúng ta bắt buộc phải đưa dữ liệu về cùng một thang đo (trừ cột bias) bằng cách tính Trung bình (Mean) và Độ lệch chuẩn (Sigma).

---

### 5. Mô hình High Variance (Phương sai cao)
Nếu chúng ta dùng bậc đa thức quá cao (ví dụ $p=10$) và vẫn giữ $\lambda = 0$:

*   **Kết quả:** Đường dự đoán đi qua chính xác mọi điểm dữ liệu huấn luyện, uốn lượn cực đoan ở các điểm đầu mút.
*   **Chẩn đoán:** Sai số huấn luyện cực thấp nhưng mô hình không thể khái quát hóa cho dữ liệu mới $\rightarrow$ **High Variance (Overfitting)**.
*   **Learning Curve:** Có một khoảng cách lớn (Gap) giữa đường sai số huấn luyện (rất thấp) và sai số kiểm chứng (rất cao).

---

### 6. Điều quy hóa (Regularization) và Chọn Lambda ($\lambda$) tối ưu
Để khắc phục Variance cao, chúng ta cần dùng **Regularization**. Siêu tham số $\lambda$ điều khiển mức độ phạt các trọng số lớn:
*   $\lambda = 0$: Overfitting (High Variance).
*   $\lambda = 100$: Underfitting (High Bias).

#### Quy trình chọn $\lambda$ (Validation Set Approach):
Chúng ta thử nghiệm một danh sách các giá trị $\lambda = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]$:
1.  Huấn luyện mô hình với từng $\lambda$ trên tập **Huấn luyện (Train)**.
2.  Đánh giá sai số trên tập **Kiểm chứng (Validation)** (không tính phần phạt $\lambda$ khi tính sai số này).
3.  Chọn $\lambda$ có sai số Validation thấp nhất.

**Kết quả:** Giá trị $\lambda = 0.003$ được xác định là tối ưu nhất.

---

### 7. Kết quả cuối cùng và Đánh giá trên tập Kiểm tra (Test Set)
Sau khi tìm được bộ tham số $\theta$ với $\lambda = 0.003$:
*   **Sai số tập Kiểm tra (Test Error):** Đạt khoảng **3.14**.
*   **Trực quan hóa:** Đường cong đa thức giờ đây bám sát xu hướng của dữ liệu một cách mềm mại, không bị uốn lượn quá mức.
*   **Learning Curve cuối cùng:** Cả sai số huấn luyện và kiểm chứng hội tụ về một mức thấp và sát nhau. 

**Kết luận:** Chúng ta đã tạo ra một mô hình thành công, cân bằng được Bias và Variance, đạt được khả năng dự báo tốt nhất cho dữ liệu chưa từng thấy.
