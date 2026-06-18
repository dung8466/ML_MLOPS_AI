Để giải quyết bài toán tính đạo hàm của hàm hợp bằng **Quy tắc chuỗi (Chain Rule)**, chúng ta cần thực hiện hai bước chính:
1.  **Lan truyền xuôi (Forward Pass):** Tính giá trị đầu vào cho từng hàm trong chuỗi để phục vụ việc tính đạo hàm tại điểm đó.
2.  **Tính đạo hàm (Chain Rule):** Tính đạo hàm riêng lẻ của từng hàm và nhân chúng lại với nhau theo công thức: 
    $$(f_n \circ \dots \circ f_1)'(x) = f_n'(f_{n-1}(\dots f_1(x))) \cdot f_{n-1}'(\dots f_1(x)) \cdot \dots \cdot f_1'(x)$$

### Hiểu Toán trước: Quy tắc chuỗi (Chain Rule)

Giả sử ta có hàm hợp $y = f(g(h(x)))$. Đạo hàm của $y$ theo $x$ được tính bằng cách "truy cứu trách nhiệm" từ ngoài vào trong:

$$\frac{dy}{dx} = \underbrace{f'(g(h(x)))}_{\text{Lớp ngoài cùng}} \cdot \underbrace{g'(h(x))}_{\text{Lớp giữa}} \cdot \underbrace{h'(x)}_{\text{Lớp trong cùng}}$$

### Ứng dụng trong ML:
Đây chính là nền tảng của thuật toán **Backpropagation** (Lan truyền ngược) trong Neural Networks. Khi bạn tính Gradient để cập nhật trọng số, máy tính sẽ thực hiện chính xác quy trình này nhưng theo chiều ngược lại để tiết kiệm chi phí tính toán.


### Triển khai Code

```python
import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Tính đạo hàm của hàm hợp bằng quy tắc chuỗi.
    Các hàm được áp dụng từ phải sang trái (phần tử cuối danh sách áp dụng trước).
    """
    
    # Định nghĩa các hàm và đạo hàm tương ứng của chúng
    func_map = {
        'square': lambda val: val**2,
        'sin': lambda val: np.sin(val),
        'exp': lambda val: np.exp(val),
        'log': lambda val: np.log(val)
    }
    
    deriv_map = {
        'square': lambda val: 2 * val,
        'sin': lambda val: np.cos(val),
        'exp': lambda val: np.exp(val),
        'log': lambda val: 1 / val
    }

    # 1. Lan truyền xuôi (Forward Pass) để lưu lại các giá trị đầu vào (inputs)
    # Nếu functions là ['sin', 'square'], ta cần x cho 'square' và x² cho 'sin'
    inputs = []
    current_val = x
    
    # Duyệt từ phải sang trái
    for func_name in reversed(functions):
        inputs.append(current_val)
        current_val = func_map[func_name](current_val)

    # 2. Áp dụng Quy tắc chuỗi (Chain Rule)
    # Đạo hàm tổng = Tích các đạo hàm thành phần tại các điểm input tương ứng
    gradient = 1.0
    for i, func_name in enumerate(reversed(functions)):
        local_input = inputs[i]
        local_derivative = deriv_map[func_name](local_input)
        gradient *= local_derivative

    return float(gradient)
```

---

**Tại sao logic code lại như vậy?**
1.  **`reversed(functions)`**: Vì đề bài nói hàm được áp dụng từ phải sang trái, nên phần tử cuối cùng của list là hàm tiếp xúc với $x$ đầu tiên ($h(x)$).
2.  **`inputs` list**: Để tính $f'(u)$, bạn cần biết $u$ là bao nhiêu. $u$ chính là kết quả của các hàm phía trước nó. Đó là lý do ta cần bước Forward Pass.
3.  **Nhân dồn (`gradient *= ...`)**: Đây chính là hiện thực hóa của dấu nhân $(\cdot)$ trong công thức toán học.
