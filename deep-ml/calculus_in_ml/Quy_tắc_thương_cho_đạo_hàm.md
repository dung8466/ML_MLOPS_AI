Để tính đạo hàm của hàm phân thức $f(x) = \frac{g(x)}{h(x)}$ tại một điểm $x$ cho trước, chúng ta sẽ áp dụng **Quy tắc đạo hàm của một thương (Quotient Rule)**.

---

### Hiểu Toán trước: Quy tắc thương

Công thức đạo hàm của một thương là:
$$f'(x) = \left( \frac{g(x)}{h(x)} \right)' = \frac{g'(x)h(x) - g(x)h'(x)}{[h(x)]^2}$$

**Các bước logic để lập trình:**
1.  **Tính $g(x)$ và $h(x)$**: Thay giá trị $x$ vào đa thức gốc.
2.  **Tìm $g'(x)$ và $h'(x)$**: Tính đạo hàm của từng đa thức, sau đó thay giá trị $x$ vào.
3.  **Áp dụng công thức**: Ráp các giá trị vừa tính được vào công thức quy tắc thương.
4.  **Lưu ý**: Nếu $h(x) = 0$, đạo hàm không xác định (lỗi chia cho 0).

=> f'(x) = (g'h - gh') / h^2

---

```python
def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # 1. Tính giá trị g(x) và g'(x)
    # Với hệ số giảm dần, bậc của phần tử i là p = len - 1 - i
    gx = 0
    g_prime_x = 0
    n_g = len(g_coeffs)
    for i in range(n_g):
        p = n_g - 1 - i  # Số mũ hiện tại
        gx += g_coeffs[i] * (x ** p)
        if p > 0:
            g_prime_x += (g_coeffs[i] * p) * (x ** (p - 1))

    # 2. Tính giá trị h(x) và h'(x)
    hx = 0
    h_prime_x = 0
    n_h = len(h_coeffs)
    for i in range(n_h):
        p = n_h - 1 - i  # Số mũ hiện tại
        hx += h_coeffs[i] * (x ** p)
        if p > 0:
            h_prime_x += (h_coeffs[i] * p) * (x ** (p - 1))

    # 3. Kiểm tra mẫu số h(x) khác 0
    if hx == 0:
        return 0.0 # Hoặc xử lý lỗi tùy yêu cầu (ví dụ raise ValueError)

    # 4. Áp dụng quy tắc thương: f'(x) = (g'h - gh') / h^2
    result = (g_prime_x * hx - gx * h_prime_x) / (hx ** 2)
    
    return float(result)
```

---

