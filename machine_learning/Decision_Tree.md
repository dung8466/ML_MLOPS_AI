
# Cây quyết định (Decision Tree) từ con số 0

Decision Tree là một thuật toán **Học có giám sát (Supervised Learning)** hoạt động theo cơ chế chia để trị. Nó tạo ra một cấu trúc dạng sơ đồ cây, trong đó mỗi nút đại diện cho một câu hỏi về một đặc trưng, mỗi nhánh là một câu trả lời và mỗi lá là một kết quả dự đoán cuối cùng.

## 1. Khái niệm cơ bản

Mục tiêu của cây quyết định là chia tập dữ liệu thành các nhóm con ngày càng thuần khiết (chứa các phần tử cùng loại).

**Các thành phần của cây:**
- **Nút gốc (Root Node):** Điểm bắt đầu, chứa toàn bộ dữ liệu.
- **Nút nhánh (Decision Node):** Nơi dữ liệu được chia dựa trên một quy tắc cụ thể.
- **Lá (Leaf Node):** Điểm cuối cùng, đưa ra nhãn (phân loại) hoặc giá trị (hồi quy).

**Cơ chế hoạt động:** Tại mỗi bước, thuật toán duyệt qua tất cả các đặc trưng và tất cả các điểm cắt có thể, chọn ra phép chia giúp giảm thiểu sự "hỗn loạn" của dữ liệu nhất.

---

## 2. Các chỉ số đo lường độ thuần khiết

Để chọn được điểm cắt tốt nhất, chúng ta cần các công thức toán học đo lường mức độ "sạch" của dữ liệu sau khi chia:

**A. Entropy (Độ hỗn loạn):**
Đo lường mức độ không chắc chắn của thông tin. Giá trị càng cao, dữ liệu càng hỗn loạn.
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

**B. Information Gain (Lợi ích thông tin):**
Mức độ giảm Entropy sau khi chia. Thuật toán sẽ chọn phép chia có Information Gain **lớn nhất**.
$$IG = Entropy(parent) - [Weighted\ Average] \cdot Entropy(children)$$

**C. Gini Impurity (Độ vẩn đục Gini):**
Thường được dùng trong thuật toán CART (Scikit-Learn dùng mặc định), tính toán nhanh hơn Entropy vì không phải tính logarit.
$$G = 1 - \sum_{i=1}^{c} p_i^2$$

---

## 3. Triển khai logic bằng Python

Cây quyết định được xây dựng thông qua hàm đệ quy:

```python
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # Chỉ số đặc trưng dùng để chia
        self.threshold = threshold   # Ngưỡng chia
        self.left = left             # Nhánh trái (con)
        self.right = right           # Nhánh phải (con)
        self.value = value           # Giá trị nếu là nút lá

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng: đạt độ sâu tối đa, hoặc chỉ còn 1 nhãn, hoặc quá ít mẫu
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Tìm phép chia tốt nhất (Best Split)
        best_feat, best_thresh = self._best_split(X, y)

        # Tạo các nhánh con bằng đệ quy
        left_idx, right_idx = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        # Duyệt qua tất cả feature và threshold để tìm IG lớn nhất
        # (Logic tính IG được đặt ở đây)
        pass
```

---

## 4. Ưu điểm và Nhược điểm

**Ưu điểm:**
- **Dễ hiểu (White Box):** Con người có thể nhìn vào cây và hiểu tại sao mô hình ra quyết định đó.
- **Không cần chuẩn hóa dữ liệu:** Khác với KNN hay SVM, Decision Tree không quan tâm đến quy mô dữ liệu (Scaling).
- Xử lý được cả dữ liệu số và dữ liệu phân loại.

**Nhược điểm:**
- **Dễ Overfitting:** Cây có xu hướng mọc rất sâu để khớp hoàn toàn dữ liệu huấn luyện.
- **Kém ổn định (High Variance):** Chỉ cần một thay đổi nhỏ trong dữ liệu cũng có thể làm cấu trúc cây thay đổi hoàn toàn.

---

## 5. Các kỹ thuật chống Overfitting

1. **Giới hạn độ sâu (Pruning):**
    - **Pre-pruning:** Dừng cây sớm bằng cách giới hạn `max_depth` hoặc yêu cầu `min_samples_leaf` (số mẫu tối thiểu ở mỗi lá).
    - **Post-pruning:** Cho cây mọc hết cỡ rồi cắt bỏ các nhánh không đóng góp nhiều vào độ chính xác.
2. **Yêu cầu số lượng mẫu để chia:** Không chia nếu một nút có quá ít dữ liệu (`min_samples_split`).

---

## 6. Những nội dung cần lưu ý

1. **Tính trực quan:** Bạn luôn có thể vẽ cây ra để kiểm tra các biến nào là quan trọng nhất (những biến nằm ở gần Nút gốc thường quan trọng hơn).
2. **Tham số quan trọng:** Khi dùng thư viện, hãy chú ý điều chỉnh `max_depth`. Một cây quá sâu là một cây "học vẹt".
3. **Tiền đề cho Ensemble:** Một cây đơn lẻ thường yếu, nhưng khi kết hợp hàng trăm cây lại, chúng ta có những thuật toán mạnh mẽ nhất hiện nay như **Random Forest** và **XGBoost**.

---
