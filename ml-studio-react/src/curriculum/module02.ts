import type { GradModule } from './types';

const COMMON_SETUP = String.raw`import json, math
import numpy as np

def _check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, 'Chưa cài đặt (NotImplementedError)'
    except Exception as error:
        ok, detail = False, f'{type(error).__name__}: {error}'
    return {'name': name, 'ok': bool(ok), 'detail': str(detail)}

def _report(ex_id, *checks):
    print('__GRAD__' + json.dumps({'id': ex_id, 'checks': list(checks)}, ensure_ascii=False))
`;

export const MODULE_02: GradModule = {
  id: 2,
  title: 'Mô hình Tuyến tính & GLMs',
  englishTitle: 'Linear Models & Generalized Linear Models',
  color: '#84e0a8',
  status: 'ready',
  overview:
    'Phân tích hình học và xác suất của Hồi quy tuyến tính OLS (phép chiếu trực giao, ma trận Hat, SVD giả nghịch đảo, định lý Gauss–Markov), điều chuẩn Ridge/Lasso/ElasticNet từ góc nhìn MAP và Coordinate Descent, phân loại Logistic/Softmax cùng tính ổn định số Log-Sum-Exp, và khung thống nhất Generalized Linear Models từ họ phân phối mũ.',
  outcomes: [
    'Chứng minh nghiệm OLS dưới góc nhìn phép chiếu trực giao $H = X(X^\\top X)^{-1}X^\\top$, phân tích leverage và xử lý ma trận suy biến hạng bằng SVD.',
    'Cài đặt nghiệm đóng Ridge có hệ số chặn không bị phạt và thuật toán Coordinate Descent cho Lasso từ điều kiện dưới vi phân (subdifferential).',
    'Chứng minh tính lồi của Cross-Entropy nhị phân/đa lớp và cài đặt hàm mất mát kèm gradient ổn định số không tràn số mũ.',
    'Thống nhất hồi quy Gaussian, Logistic, Softmax và Poisson qua dạng chính tắc của họ phân phối mũ (Exponential Family) và hàm liên kết chính tắc.',
  ],
  sources: [
    'UC Berkeley CS189/289A — Lecture 06–11 (Linear & Logistic Regression)',
    'Stanford CS229 — Machine Learning Complete Notes (Part I: Supervised Learning & GLMs)',
    'ISLR 7th Printing — Ch. 3, Ch. 4, Ch. 6 (Linear Models & Regularization)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 10–12)',
  ],
  units: [
    {
      id: 'm2-u1',
      moduleId: 2,
      order: 1,
      title: 'OLS dưới góc nhìn Hình học (Phép chiếu) & Xác suất (Gaussian MLE)',
      summary:
        'Bản chất hình học của Bình phương tối thiểu (Ordinary Least Squares): phép chiếu trực giao vector mục tiêu $y$ lên không gian cột $\\text{col}(X)$, ma trận Hat $H$, tính chất định lý Gauss–Markov và ổn định hóa bằng SVD khi $X^\\top X$ suy biến.',
      outcomes: [
        'Dẫn xuất Phương trình chuẩn $X^\\top X w = X^\\top y$ bằng cả giải tích ma trận và điều kiện trực giao $(y - Xw) \\perp \\text{col}(X)$.',
        'Phân tích các tính chất đại số của ma trận chiếu trực giao $H$: đối xứng, lũy đẳng ($H^2 = H$) và $\\text{tr}(H) = \\text{rank}(X)$.',
        'Hiểu định lý Gauss–Markov (OLS là BLUE) và dùng phân rã SVD tính giả nghịch đảo Moore–Penrose khi xảy ra đa cộng tuyến hoàn hảo.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 4/Lecture 07 -- Linear Regression (2).pdf',
          section: 'Geometric View & Normal Equations',
          note: 'Biểu diễn phép chiếu trực giao lên không gian con sinh bởi các cột của X.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part I — Section 1 (Linear Regression)',
          note: 'Normal Equations và tương đương với Gaussian MLE.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§11.2 (Linear Regression — Least Squares & SVD)',
          note: 'Giải hệ trực chuẩn bằng QR/SVD thay vì nghịch đảo trực tiếp ma trận Gram XᵀX.',
        },
      ],
      sections: [
        {
          heading: '1. Phương trình chuẩn & Phép chiếu trực giao lên không gian cột col(X)',
          body: String.raw`Xét ma trận thiết kế $X \in \mathbb{R}^{n \times d}$ và vector mục tiêu $y \in \mathbb{R}^n$. Bài toán OLS tìm $\hat{y} = Xw \in \text{col}(X)$ gần $y$ nhất theo chuẩn Euclid $\|y - Xw\|_2$.

Theo hình học không gian, vector nối $\hat{y}$ với $y$ — tức vector phần dư $r = y - \hat{y}$ — đạt độ dài ngắn nhất khi và chỉ khi nó trực giao với mọi vector cơ sở (mọi cột) trong không gian cột $\text{col}(X)$:`,
          math: String.raw`X^\top (y - X\hat{w}) = 0 \iff X^\top X \hat{w} = X^\top y \implies \hat{y} = \underbrace{X(X^\top X)^{-1}X^\top}_{H}\, y`,
          intuition:
            'Ma trận $H \\in \\mathbb{R}^{n \\times n}$ (gọi là **Hat matrix** vì nó "đội mũ" lên $y$ thành $\\hat{y}$) là toán tử chiếu trực giao thỏa mãn hai tính chất đại số quan trọng: đối xứng ($H^\\top = H$) và lũy đẳng ($H^2 = H$, chiếu hai lần vẫn giữ nguyên vị trí). Phần tử đường chéo $h_{ii} = H_{ii}$ gọi là **leverage**, đo ảnh hưởng riêng của mẫu thứ $i$ lên dự đoán của chính nó, với tổng $\\sum_{i=1}^n h_{ii} = \\text{tr}(H) = \\text{rank}(X)$.',
        },
        {
          heading: '2. Định lý Gauss–Markov & Tính không chệch phương sai cực tiểu',
          body: String.raw`Dưới mô hình tuyến tính chuẩn $y = Xw^* + \varepsilon$ với các giả định tối thiểu: $\mathbb{E}[\varepsilon \mid X] = 0$, đồng phương sai $\text{Cov}(\varepsilon \mid X) = \sigma^2 I_n$ (không cần giả định $\varepsilon$ theo phân phối chuẩn Gaussian), **Định lý Gauss–Markov** khẳng định: trong mọi ước lượng tuyến tính không chệch của $w^*$ (dạng $\tilde{w} = Cy$ sao cho $\mathbb{E}[\tilde{w}] = w^*$), nghiệm OLS $\hat{w}_{\text{OLS}}$ có ma trận hiệp phương sai nhỏ nhất theo thứ tự Loewner (**BLUE — Best Linear Unbiased Estimator**):`,
          math: String.raw`\text{Cov}(\hat{w}_{\text{OLS}}) = \sigma^2 (X^\top X)^{-1} \preceq \text{Cov}(\tilde{w})`,
          pitfall:
            'Khi các cột đặc trưng có tương quan tuyến tính rất mạnh (đa cộng tuyến — multicollinearity), một trị riêng của $X^\\top X$ tiệm cận $0$ làm phần tử trên $(X^\\top X)^{-1}$ bùng nổ, khiến phương sai của hệ số $\\hat{w}$ tăng vọt dù sai số trên tập train vẫn nhỏ.',
        },
        {
          heading: '3. Phân rã SVD & Nghiệm chuẩn cực tiểu (Pseudoinverse)',
          body: String.raw`Tính trực tiếp $(X^\top X)^{-1}$ bình phương số điều kiện ($\kappa(X^\top X) = \kappa(X)^2$) và thất bại hoàn toàn nếu $X$ suy biến hạng ($\text{rank}(X) < d$, hoặc $d > n$).

Bằng phân rã giá trị kỳ dị rút gọn $X = U \Sigma V^\top$, giả nghịch đảo Moore–Penrose $X^+ = V \Sigma^+ U^\top$ cho nghiệm chuẩn $\ell_2$ nhỏ nhất trong tập vô số nghiệm tối ưu của $\|Xw - y\|_2^2$:`,
          math: String.raw`\hat{w}_{\text{SVD}} = X^+ y = \sum_{k : \sigma_k > 0} \frac{u_k^\top y}{\sigma_k} v_k, \qquad H = X X^+ = U_r U_r^\top`,
          intuition:
            'Hàm `np.linalg.lstsq(X, y, rcond=None)` không bao giờ nhân `X.T @ X` mà giải trực tiếp qua SVD/LAPACK, vừa ổn định số học gấp bình phương lần về sai số làm tròn, vừa tự động chọn nghiệm chuẩn tối thiểu khi thiếu hạng.',
        },
      ],
      simulationId: 'm2-u1',
      exercises: [
        {
          id: 'm2-u1-ols-svd',
          title: 'Khớp OLS bằng SVD (An toàn khi đa cộng tuyến / thiếu hạng)',
          prompt:
            'Cài đặt `fit_ols_svd(X, y)` học mô hình tuyến tính $\\hat{y} = Xw + b$ bằng cách mở rộng ma trận $X$ thêm cột hệ số $1$ và giải qua SVD (`np.linalg.lstsq(..., rcond=None)`). Trả về `(w, b)` với `w` là mảng 1D `(d,)` và `b` là số thực `float`. Hàm phải hoạt động ổn định cả khi các cột của $X$ phụ thuộc tuyến tính.',
          hint: 'Tạo `X_aug = np.column_stack([X, np.ones(len(X))])`, gọi `theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]`, sau đó trả về `theta[:-1], float(theta[-1])`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X_full = np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.], [5., 6.]])
_y_full = np.array([5.1, 4.9, 9.2, 8.8, 13.0])
_X_rank_def = np.column_stack([_X_full[:, 0], 2.0 * _X_full[:, 0], _X_full[:, 1]])
`,
          starter: String.raw`def fit_ols_svd(X, y):
    raise NotImplementedError
`,
          solution: String.raw`def fit_ols_svd(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = np.column_stack([X, np.ones(len(X), dtype=float)])
    theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    return theta[:-1], float(theta[-1])
`,
          test: String.raw`_report('m2-u1-ols-svd',
    _check('Phần dư trực giao với mọi cột của X và cột hằng 1 (Xᵀr ≈ 0, sum(r) ≈ 0)', lambda: (
        (lambda w, b: (
            (lambda r: np.allclose(_X_full.T @ r, 0.0, atol=1e-9) and abs(np.sum(r)) < 1e-9)(_X_full @ w + b - _y_full)
        ))(*fit_ols_svd(_X_full, _y_full)),
        'Kiểm tra điều kiện phương trình chuẩn Normal Equations.'
    )),
    _check('Không sụp đổ với ma trận suy biến hạng (cột 2 = 2 × cột 1) và cho nghiệm chuẩn cực tiểu', lambda: (
        (lambda w_rd, b_rd: (
            np.all(np.isfinite(w_rd)) and np.isfinite(b_rd) and np.allclose(_X_rank_def @ w_rd + b_rd, _X_full @ fit_ols_svd(_X_full, _y_full)[0] + fit_ols_svd(_X_full, _y_full)[1])
        ))(*fit_ols_svd(_X_rank_def, _y_full)),
        'Dự đoán của không gian cột trùng nhau phải giống hệt bản đủ hạng.'
    )),
)
`,
        },
        {
          id: 'm2-u1-hat-matrix',
          title: 'Ma trận chiếu Hat Matrix, Leverage & Bậc tự do',
          prompt:
            'Cài đặt `hat_matrix_leverage(X)` tính ma trận chiếu trực giao $H = X X^+$ lên không gian cột của $X$ (dùng `np.linalg.pinv(X)` để an toàn cho cả ma trận thiếu hạng), rồi trả về tuple `(H, leverages, trace_H)` với `leverages = np.diag(H)` và `trace_H = float(np.trace(H))`.',
          hint: 'Dùng `H = X @ np.linalg.pinv(X)`, `leverages = np.diag(H)`, `trace_H = float(np.trace(H))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X_hat = np.array([[1., 0.], [1., 1.], [1., 2.], [1., 5.], [1., 10.]])
`,
          starter: String.raw`def hat_matrix_leverage(X):
    raise NotImplementedError
`,
          solution: String.raw`def hat_matrix_leverage(X):
    X = np.asarray(X, dtype=float)
    H = X @ np.linalg.pinv(X)
    leverages = np.diag(H)
    trace_H = float(np.trace(H))
    return H, leverages, trace_H
`,
          test: String.raw`_report('m2-u1-hat-matrix',
    _check('H đối xứng (H = Hᵀ), lũy đẳng (H² = H) và tr(H) = rank(X)', lambda: (
        (lambda H, lev, tr: (
            np.allclose(H, H.T) and np.allclose(H @ H, H) and np.isclose(tr, np.linalg.matrix_rank(_X_hat))
        ))(*hat_matrix_leverage(_X_hat)),
        'Kiểm chứng tính chất toán tử chiếu trực giao.'
    )),
    _check('Leverage nằm trong [0, 1] và mẫu ngoại biên (x=10) có leverage cao nhất', lambda: (
        (lambda H, lev, tr: (
            np.all((lev >= -1e-9) & (lev <= 1.0 + 1e-9)) and int(np.argmax(lev)) == 4
        ))(*hat_matrix_leverage(_X_hat)),
        'Điểm xa tâm đặc trưng nhất mang trọng số tự kéo lớn nhất.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Chứng minh công thức Leave-One-Out Cross-Validation (LOOCV) dạng đóng cho OLS không cần fit lại $n$ lần: $e_i^{(-i)} = \\frac{y_i - \\hat{y}_i}{1 - h_{ii}}$. Ý nghĩa của $h_{ii} \\to 1$ là gì?',
          guidance:
            'Gọi $\\hat{y}^{(-i)}$ là dự đoán khi bỏ mẫu $i$. Theo định lý Sherman–Morrison, bỏ hàng $i$ làm dịch dự đoán tại $i$ đúng một lượng tỉ lệ với phần dư $y_i - \\hat{y}_i$ chia $1 - h_{ii}$. Khi $h_{ii} \\to 1$, mẫu $i$ một mình gánh phương sai mô hình theo hướng ngoại biên; nếu bỏ đi, sai số dự đoán bùng nổ.',
        },
        {
          question:
            'Tại sao trong tính toán khoa học chuyên nghiệp không bao giờ dùng `np.linalg.inv(X.T @ X) @ X.T @ y` dù hai công thức tương đương về đại số với SVD?',
          guidance:
            'Số điều kiện (condition number) của $X^\\top X$ bằng $\\kappa(X)^2$. Với $\\kappa(X) = 10^8$, chuẩn 64-bit IEEE float (16 chữ số có nghĩa) bị mất sạch chữ số khi bình phương lên $\\kappa(X)^2 = 10^{16}$, dẫn đến nghiệm sai hoàn toàn hoặc ma trận suy biến số học.',
        },
      ],
    },
    {
      id: 'm2-u2',
      moduleId: 2,
      order: 2,
      title: 'Regularization Ridge (L2), Lasso (L1) & ElasticNet từ MAP',
      summary:
        'Kiểm soát phương sai mô hình tuyến tính trong chiều cao: nghiệm đóng Ridge (chỉ phạt trọng số $w$, không phạt bias $b$), khái niệm Bậc tự do hiệu dụng $\\text{df}(\\lambda)$, hình học thưa của chuẩn $L_1$, toán tử Soft-Thresholding và thuật toán Coordinate Descent cho Lasso.',
      outcomes: [
        'Cài đặt đúng nghiệm đóng Ridge có tâm hóa dữ liệu (centering) để không phạt hệ số chặn $b$.',
        'Giải thích sự co cụm theo từng phương chính SVD: $\\frac{\\sigma_k^2}{\\sigma_k^2 + \\lambda}$ và bậc tự do hiệu dụng $\\text{df}(\\lambda)$.',
        'Dẫn xuất phép cập nhật Coordinate Descent cho Lasso qua dưới vi phân và hàm Soft-Thresholding $S_\\lambda(z)$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§6.2 (Shrinkage Methods: Ridge & Lasso)',
          note: 'Đường đi regularization path, chọn lọc biến của L1 và so sánh đánh đổi với L2.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§11.3–11.4 (Ridge Regression, MAP & Sparse Linear Models)',
          note: 'Liên hệ giữa Gaussian/Laplace prior và L2/L1 regularization.',
        },
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 5/Lecture 09 -- Regression (Cont.) and Bias-Variance Trade-off.pdf',
          section: 'Ridge Regression & SVD Shrinkage',
          note: 'Phân tích trị kỳ dị và bậc tự do hiệu dụng df(λ).',
        },
      ],
      sections: [
        {
          heading: '1. Hồi quy Ridge ($L_2$), Tâm hóa dữ liệu & Bậc tự do hiệu dụng',
          body: String.raw`Hồi quy Ridge thêm thành phần phạt bình phương chuẩn Euclidean vào hàm mất mát OLS: $\min_{w, b} \|Xw + b\mathbf{1} - y\|_2^2 + \lambda \|w\|_2^2$.
Một nguyên tắc thiết yếu là **không bao giờ phạt hệ số chặn $b$**, vì phạt $b$ làm kết quả phụ thuộc vào việc dịch gốc đo của biến mục tiêu $y$. Bằng cách trừ trung bình theo cột ($\tilde{X} = X - \bar{x}^\top$, $\tilde{y} = y - \bar{y}$), ta tách riêng bài toán tối ưu chỉ theo $w$, sau đó khôi phục $b = \bar{y} - \bar{x}^\top \hat{w}$:`,
          math: String.raw`\hat{w}_{\text{Ridge}} = (\tilde{X}^\top \tilde{X} + \lambda I_d)^{-1} \tilde{X}^\top \tilde{y}, \qquad \hat{y}_{\text{Ridge}} - \bar{y} = \sum_{k=1}^r \frac{\sigma_k^2}{\sigma_k^2 + \lambda} u_k (u_k^\top \tilde{y})`,
          intuition:
            'Khai triển qua SVD cho thấy Ridge tác động mạnh nhất lên các phương có phương sai nhỏ ($\\sigma_k^2 \\ll \\lambda$), nơi nhiễu thống kê dễ bị khuếch đại nhất. Tổng hệ số co cụm $\\text{df}(\\lambda) = \\sum_{k=1}^r \\frac{\\sigma_k^2}{\\sigma_k^2 + \\lambda}$ gọi là **Effective Degrees of Freedom** — giảm liên tục từ $r$ về $0$ khi $\\lambda$ tăng từ $0$ tới $\\infty$.',
        },
        {
          heading: '2. Hồi quy Lasso ($L_1$) & Hình học thưa (Sparsity)',
          body: String.raw`Lasso thay phạt $L_2$ bằng chuẩn $L_1$: $\min_{w} \frac{1}{2n}\|Xw - y\|_2^2 + \lambda \|w\|_1$. Tuy $L_1$ không khả vi tại các điểm có $w_j = 0$, hàm mục tiêu vẫn lồi chặt (hoặc lồi) nên nghiệm toàn cục tồn tại duy nhất trong điều kiện tổng quát.

Dưới góc nhìn hình học, đường đồng mức elip của hàm mất mát bình phương tiếp xúc với khối đa diện cầu chuẩn $L_1$ ($\sum |w_j| \le t$) rất dễ chạm đúng tại các **đỉnh góc nhọn** nằm trên trục tọa độ — tương ứng chính xác với một số hệ số $w_j = 0$ tuyệt đối.`,
          pitfall:
            'Nếu hai đặc trưng có tương quan rất cao ($x_1 \\approx x_2$), Lasso thường chọn tùy ý một biến và triệt tiêu biến kia về 0, gây mất ổn định khi chọn đặc trưng. **ElasticNet** kết hợp cả $L_1$ và $L_2$ ($\\lambda_1\\|w\\|_1 + \\frac{\\lambda_2}{2}\\|w\\|_2^2$) giải quyết nhược điểm này (grouping effect).',
        },
        {
          heading: '3. Toán tử Soft-Thresholding & Coordinate Descent cho Lasso',
          body: String.raw`Vì $\lambda\|w\|_1 = \sum_{j=1}^d \lambda|w_j|$ tách rời được theo từng tọa độ, **Coordinate Descent** là một trong những thuật toán nhanh nhất giải bài toán Lasso: giữ cố định mọi $w_{k \neq j}$ và cực tiểu hóa chính xác theo từng biến $w_j$ đơn lẻ bằng giải tích dưới vi phân (subgradient).

Đặt phần dư riêng $r_{-j} = y - \sum_{k \neq j} X_{:,k} w_k$, $\rho_j = \frac{1}{n} X_{:,j}^\top r_{-j}$, và $z_j = \frac{1}{n}\|X_{:,j}\|_2^2$. Đạo hàm dưới vi phân bằng $0$ cho nghiệm giải tích qua hàm **Soft-Thresholding** $S_\lambda(\rho)$:`,
          math: String.raw`S_\lambda(\rho) = \text{sign}(\rho)\max(|\rho| - \lambda,\, 0), \qquad w_j^{\text{new}} = \frac{S_\lambda(\rho_j)}{z_j}`,
          intuition:
            'Bất kỳ đặc trưng nào có tương quan riêng $|\\rho_j| \\le \\lambda$ đều bị thuật toán đóng thẳng về đúng $0.0$ trong một bước, biến Lasso thành bộ chọn lọc đặc trưng tự động.',
        },
      ],
      simulationId: 'm2-u2',
      exercises: [
        {
          id: 'm2-u2-ridge',
          title: 'Nghiệm đóng Ridge có tâm hóa (Không phạt hệ số chặn)',
          prompt:
            'Cài đặt `fit_ridge_closed_form(X, y, lam)` tìm nghiệm Ridge cho $\\hat{y} = Xw + b$ với hệ số phạt `lam` ($\\lambda \\ge 0$) chỉ áp lên vector $w$ (không phạt $b$). Trả về `(w, b)`. Yêu cầu thực hiện tâm hóa $X, y$ và giải hệ $(\\tilde{X}^\\top \\tilde{X} + \\lambda I) w = \\tilde{X}^\\top \\tilde{y}$.',
          hint: 'Tính `x_mu = X.mean(axis=0)`, `y_mu = float(y.mean())`, `Xc = X - x_mu`, `yc = y - y_mu`. Giải `w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)` và trả về `w, float(y_mu - x_mu @ w)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xr = np.array([[1., 0.9], [2., 2.1], [3., 2.9], [4., 4.1], [5., 4.9], [6., 6.1]])
_yr = np.array([3.2, 6.0, 9.1, 11.9, 15.2, 18.0])
`,
          starter: String.raw`def fit_ridge_closed_form(X, y, lam):
    raise NotImplementedError
`,
          solution: String.raw`def fit_ridge_closed_form(X, y, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mu = X.mean(axis=0)
    y_mu = float(y.mean())
    Xc = X - x_mu
    yc = y - y_mu
    d = X.shape[1]
    A = Xc.T @ Xc + float(lam) * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y_mu - x_mu @ w)
    return w, b
`,
          test: String.raw`_report('m2-u2-ridge',
    _check('Chuẩn ||w|| giảm đơn điệu khi tăng λ (0.1 → 10 → 100)', lambda: (
        (lambda n1, n2, n3: n1 > n2 > n3 > 0)(
            np.linalg.norm(fit_ridge_closed_form(_Xr, _yr, 0.1)[0]),
            np.linalg.norm(fit_ridge_closed_form(_Xr, _yr, 10.0)[0]),
            np.linalg.norm(fit_ridge_closed_form(_Xr, _yr, 100.0)[0]),
        ),
        'Kiểm tra hiệu ứng co cụm tham số của phạt L2.'
    )),
    _check('Hệ số chặn b không bị phạt: dịch y thêm +1000 chỉ dịch b thêm +1000, w giữ nguyên', lambda: (
        (lambda res1, res2: np.allclose(res1[0], res2[0]) and np.isclose(res2[1] - res1[1], 1000.0))(
            fit_ridge_closed_form(_Xr, _yr, 5.0),
            fit_ridge_closed_form(_Xr, _yr + 1000.0, 5.0),
        ),
        'Tâm hóa dữ liệu bảo toàn bất biến dịch chuyển của bias.'
    )),
)
`,
        },
        {
          id: 'm2-u2-soft-thresh',
          title: 'Toán tử Soft-Thresholding & Coordinate Descent cho Lasso',
          prompt:
            'Cài đặt hai hàm: `soft_threshold(z, lam)` trả về $\\text{sign}(z)\\max(|z| - \\lambda, 0)$, và `lasso_coord_step(X, y, w, lam)` thực hiện **một vòng quét đầy đủ** ($j = 0, \\dots, d-1$ tuần tự theo kiểu Gauss–Seidel trên bản sao của $w$) tối ưu mục tiêu $\\frac{1}{2n}\\|Xw - y\\|_2^2 + \\lambda\\|w\\|_1$. Trả về vector `w_new`.',
          hint: 'Ở mỗi tọa độ $j$, tính `col = w_new[:, ...]` — cụ thể `xj = X[:, j]`, phần dư riêng `r_noj = y - (X @ w_new - xj * w_new[j])`, `rho = (xj @ r_noj) / n`, `zj = (xj @ xj) / n`, rồi cập nhật `w_new[j] = soft_threshold(rho, lam) / zj` (nếu `zj > 0`).',
          setup:
            COMMON_SETUP +
            String.raw`
_Xl = np.array([[1., 0.1, -0.2], [2., 0.2, 0.4], [-1., 0.1, 1.5], [0.5, -0.3, -1.0], [-2., 0.2, 0.3]])
_yl = 3.0 * _Xl[:, 0] + 0.05 * _Xl[:, 1]
def _lasso_obj(X, y, w, lam):
    return 0.5 * float(np.mean((X @ w - y) ** 2)) + float(lam * np.sum(np.abs(w)))
def _run_lasso_cd(X, y, lam, sweeps=150):
    w = np.zeros(X.shape[1])
    for _ in range(sweeps):
        w = lasso_coord_step(X, y, w, lam)
    return w
`,
          starter: String.raw`def soft_threshold(z, lam):
    raise NotImplementedError

def lasso_coord_step(X, y, w, lam):
    raise NotImplementedError
`,
          solution: String.raw`def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

def lasso_coord_step(X, y, w, lam):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w_new = np.asarray(w, dtype=float).copy()
    n, d = X.shape
    for j in range(d):
        xj = X[:, j]
        r_noj = y - (X @ w_new - xj * w_new[j])
        rho = float(xj @ r_noj) / n
        zj = float(xj @ xj) / n
        w_new[j] = float(soft_threshold(rho, lam) / zj) if zj > 0 else 0.0
    return w_new
`,
          test: String.raw`_report('m2-u2-soft-thresh',
    _check('soft_threshold hoạt động đúng trong và ngoài vùng ngưỡng [-λ, λ]', lambda: (
        np.allclose([soft_threshold(3.5, 1.0), soft_threshold(-2.5, 1.0), soft_threshold(0.8, 1.0)], [2.5, -1.5, 0.0]),
        'Kiểm tra giá trị dương, âm và bị kẹp về 0.'
    )),
    _check('Mỗi vòng Coordinate Descent đều làm giảm (không tăng) hàm mục tiêu Lasso', lambda: (
        (lambda w0, w1: _lasso_obj(_Xl, _yl, w1, 0.5) < _lasso_obj(_Xl, _yl, w0, 0.5))(
            np.zeros(3),
            lasso_coord_step(_Xl, _yl, np.zeros(3), 0.5),
        ),
        'Khẳng định tính đơn điệu giảm của Coordinate Descent.'
    )),
    _check('Chạy 150 vòng hội tụ và triệt tiêu biến nhiễu về đúng 0.0 (Sparsity)', lambda: (
        (lambda w_fin: abs(w_fin[1]) == 0.0 and w_fin[0] > 1.5)(_run_lasso_cd(_Xl, _yl, 0.4, sweeps=150)),
        'Đặc trưng yếu thứ 2 bị đóng về 0 tuyệt đối.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question: String.raw`Chứng minh mối quan hệ giữa Ridge Regression và tiên nghiệm Bayes: nếu đặt $y \mid X, w \sim \mathcal{N}(Xw, \sigma^2 I)$ và tiên nghiệm đẳng hướng $w \sim \mathcal{N}(0, \tau^2 I)$, hậu nghiệm $p(w \mid X, y)$ cũng là Gaussian với trung bình $\mu_{\text{post}}$ chính bằng $\hat{w}_{\text{Ridge}}$ tại $\lambda = \sigma^2 / \tau^2$.`,
          guidance:
            'Lấy log của $p(w \\mid X, y) \\propto p(y \\mid X, w)p(w)$, ta được dạng toàn phương theo $w$: $-\\frac{1}{2}\\big[ w^\\top(\\frac{1}{\\sigma^2}X^\\top X + \\frac{1}{\\tau^2}I)w - \\frac{2}{\\sigma^2}y^\\top Xw \\big] + \\text{const}$. Hoàn thành bình phương cho Gaussian cho hiệp phương sai hậu nghiệm $\\Sigma_{\\text{post}} = (\\frac{1}{\\sigma^2}X^\\top X + \\frac{1}{\\tau^2}I)^{-1}$ và trung bình $\\mu_{\\text{post}} = (X^\\top X + \\frac{\\sigma^2}{\\tau^2}I)^{-1}X^\\top y$.',
        },
        {
          question:
            'Vì sao bắt buộc phải chuẩn hóa thang đo các cột của $X$ về cùng độ lệch chuẩn trước khi áp dụng Ridge hoặc Lasso?',
          guidance:
            'Nếu nhân đặc trưng $x_j$ với $1000$ (ví dụ đổi đơn vị mét sang milimet), hệ số $w_j$ tương ứng phải chia $1000$ để giữ nguyên dự đoán. Tuy nhiên thành phần phạt $\\lambda w_j^2$ hoặc $\\lambda |w_j|$ xem mọi hệ số bình đẳng, dẫn tới phạt quá nặng đặc trưng có đơn vị lớn hoặc quá nhẹ đặc trưng có đơn vị nhỏ.',
        },
      ],
    },
    {
      id: 'm2-u3',
      moduleId: 2,
      order: 3,
      title: 'Logistic Regression, Softmax & Cross-Entropy',
      summary:
        'Phân loại xác suất tuyến tính: hàm Logistic Sigmoid, chứng minh tính lồi toàn cục của Binary Cross-Entropy qua ma trận Hessian $X^\\top S X$, mở rộng đa lớp Softmax với kỹ thuật ổn định số Log-Sum-Exp và hiệu chỉnh ngưỡng quyết định.',
      outcomes: [
        'Chứng minh hàm mất mát Binary Cross-Entropy (NLL Bernoulli) là hàm lồi theo $(w, b)$ bằng giải tích Hessian.',
        'Cài đặt Sigmoid, Binary Cross-Entropy và Softmax Cross-Entropy bất biến trước logit cực đại/cực tiểu ($|z| = 1000$) không gặp lỗi `overflow` hay `log(0)`.',
        'Hiểu sự tách biệt giữa mô hình xác suất có hiệu chỉnh $P(y=1 \\mid x)$ và lựa chọn ngưỡng quyết định theo chi phí sai số.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 6/Lecture 10 -- Logistic Regression (1).pdf',
          section: 'Sigmoid, Cross-Entropy & Convexity',
          note: 'Dẫn xuất NLL của phân phối Bernoulli và cực tiểu hóa lồi.',
        },
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 6/Lecture 11 -- Logistic Regression (2).pdf',
          section: 'Multiclass Softmax Regression',
          note: 'Tổng quát hóa sang K lớp qua Softmax và đối xứng trọng số.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§10.2–10.3 (Binary & Multinomial Logistic Regression)',
          note: 'Bất đẳng thức Log-Sum-Exp và giải tích ma trận Hessian.',
        },
      ],
      sections: [
        {
          heading: '1. Mô hình Bernoulli NLL (Binary Cross-Entropy) & Tính lồi qua Hessian',
          body: String.raw`Với nhãn nhị phân $y_i \in \{0, 1\}$, Logistic Regression mô hình hóa xác suất hậu nghiệm thông qua logit $z_i = w^\top x_i + b$ và hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, có đạo hàm đẹp $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

Tối đa hóa hàm Log-Likelihood Bernoulli tương đương tối thiểu hóa **Binary Cross-Entropy (BCE)**:`,
          math: String.raw`\mathcal{L}_{\text{BCE}}(w, b) = -\frac{1}{n}\sum_{i=1}^n \Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big], \qquad \nabla_w \mathcal{L}_{\text{BCE}} = \frac{1}{n} X^\top (p - y)`,
          intuition:
            'Tính ma trận đạo hàm bậc hai (Hessian) theo $w$ cho ta $H(w) = \\frac{1}{n} X^\\top S X$, trong đó $S = \\text{diag}(p_1(1-p_1), \\dots, p_n(1-p_n))$ có mọi phần tử đường chéo $> 0$. Do đó $v^\\top H(w) v = \\frac{1}{n} \\sum p_i(1-p_i)(x_i^\\top v)^2 \\ge 0$ với mọi $v$ — chứng tỏ BCE **lồi toàn cục**, hoàn toàn không bị kẹt ở cực tiểu địa phương.',
          pitfall:
            'Dùng công thức ngây thơ `-y * np.log(sigmoid(z)) - (1 - y) * np.log(1 - sigmoid(z))` sẽ bị tràn mũ `exp(1000)` hoặc `log(0) = -inf` gây `NaN` khi `0 * (-inf)`. Dạng biến đổi đại số ổn định là `np.logaddexp(0, z) - y * z`.',
        },
        {
          heading: '2. Softmax đa lớp & Thủ thuật Log-Sum-Exp',
          body: String.raw`Với bài toán $K$ lớp, logit $z \in \mathbb{R}^K$ ánh xạ thành phân phối xác suất qua hàm Softmax. Để chống tràn số khi các logit mang giá trị rất lớn hoặc rất âm, ta khai thác tính bất biến tịnh tiến của Softmax khi trừ đi hằng số $m = \max_k z_k$:`,
          math: String.raw`\text{Softmax}(z)_c = \frac{e^{z_c - m}}{\sum_{k=1}^K e^{z_k - m}}, \qquad \log \sum_{k=1}^K e^{z_k} = m + \log \sum_{k=1}^K e^{z_k - m}`,
          intuition:
            'Với cách viết trừ $m$, mọi số hạng trong `exp(...)` luôn $\\le 0$, nên `exp` luôn nằm trong $[0, 1]$ và tổng mẫu luôn $\\ge 1$ — loại bỏ 100% khả năng tràn số trên (`OverflowError`) lẫn chia cho $0$.',
        },
      ],
      simulationId: 'm2-u3',
      exercises: [
        {
          id: 'm2-u3-stable-bce',
          title: 'Binary Cross-Entropy & Gradient ổn định số tuyệt đối',
          prompt:
            'Cài đặt `stable_bce_and_grad(X, y, w, b)` tính `(loss, dw, db)` cho Binary Cross-Entropy không bị tràn số hay `NaN` ngay cả khi logit `z = X @ w + b` đạt tới $\\pm 500$. Kiểm tra tự động trên cả độ chính xác gradient và giá trị biên cực hạn.',
          hint: 'Dùng `z = X @ w + b`, mất mát ổn định bằng `np.mean(np.logaddexp(0.0, z) - y * z)`, và sigmoid ổn định bằng nhánh `z >= 0` (`1/(1+exp(-z))`) vs `z < 0` (`exp(z)/(1+exp(z))`). Gradient là `dw = X.T @ ((p - y) / n)` và `db = float(np.mean(p - y))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xb = np.array([[0.5, -1.2], [-1.0, 0.8], [1.4, 1.1], [-0.3, -0.7]])
_yb = np.array([1., 0., 1., 0.])
_wb = np.array([0.6, -0.4])
_bb = 0.2

def _num_bce_grad(X, y, w, b, eps=1e-6):
    def f(tw, tb):
        z = X @ tw + tb
        return float(np.mean(np.logaddexp(0.0, z) - y * z))
    gw = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = eps
        gw[j] = (f(w + e, b) - f(w - e, b)) / (2 * eps)
    gb = (f(w, b + eps) - f(w, b - eps)) / (2 * eps)
    return gw, gb
`,
          starter: String.raw`def stable_bce_and_grad(X, y, w, b):
    raise NotImplementedError
`,
          solution: String.raw`def stable_bce_and_grad(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    z = X @ w + float(b)
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, 0.0, None))), np.exp(np.clip(z, None, 0.0)) / (1.0 + np.exp(np.clip(z, None, 0.0))))
    err = (p - y) / len(y)
    dw = X.T @ err
    db = float(np.sum(err))
    return loss, dw, db
`,
          test: String.raw`_report('m2-u3-stable-bce',
    _check('Gradient dw, db khớp sai phân hữu hạn (< 1e-7)', lambda: (
        (lambda loss, dw, db: (
            (lambda ngw, ngb: np.allclose(dw, ngw, atol=1e-7) and abs(db - ngb) < 1e-7)(*_num_bce_grad(_Xb, _yb, _wb, _bb))
        ))(*stable_bce_and_grad(_Xb, _yb, _wb, _bb)),
        'Xác nhận đạo hàm đúng chính xác theo công thức Xᵀ(p - y)/n.'
    )),
    _check('An toàn số học khi logit đạt ngưỡng cực lớn z = ±500 (không NaN/Inf)', lambda: (
        (lambda res: np.isfinite(res[0]) and np.all(np.isfinite(res[1])) and np.isfinite(res[2]) and res[0] > 400.0)(
            stable_bce_and_grad(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([0.0, 1.0]), np.array([500.0, 0.0]), 0.0)
        ),
        'Phạt tuyến tính theo |z| khi dự đoán sai hoàn toàn với logit cực đại.'
    )),
)
`,
        },
        {
          id: 'm2-u3-softmax-ce',
          title: 'Softmax & Cross-Entropy đa lớp bằng Log-Sum-Exp',
          prompt:
            'Cài đặt `softmax_cross_entropy(logits, y_int)` nhận ma trận `logits` `(n, K)` và vector nhãn nguyên `y_int` `(n,)` thuộc `0..K-1`. Trả về `(probs, loss)` trong đó `probs` là ma trận xác suất `(n, K)` có tổng mỗi hàng bằng $1$, và `loss` là trung bình Cross-Entropy ổn định số ngay cả khi `logits` chứa các số hàng nghìn.',
          hint: 'Tính `shifted = logits - np.max(logits, axis=1, keepdims=True)`, `log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))`, `log_probs = shifted - log_sum_exp`, `probs = np.exp(log_probs)`, và `loss = float(-np.mean(log_probs[np.arange(len(y_int)), y_int]))`.',
          setup: COMMON_SETUP,
          starter: String.raw`def softmax_cross_entropy(logits, y_int):
    raise NotImplementedError
`,
          solution: String.raw`def softmax_cross_entropy(logits, y_int):
    logits = np.asarray(logits, dtype=float)
    y_int = np.asarray(y_int, dtype=int)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(exp_s, axis=1, keepdims=True))
    loss = float(-np.mean(log_probs[np.arange(len(y_int)), y_int]))
    return probs, loss
`,
          test: String.raw`_report('m2-u3-softmax-ce',
    _check('Tổng xác suất mỗi hàng bằng 1 và khớp giá trị Softmax chuẩn', lambda: (
        (lambda probs, loss: np.allclose(probs.sum(axis=1), 1.0) and np.allclose(probs[0], np.exp([1., 2., 0.]) / np.sum(np.exp([1., 2., 0.]))))(
            *softmax_cross_entropy([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], [1, 2])
        ),
        'Kiểm tra tính chuẩn hóa phân phối xác suất đa lớp.'
    )),
    _check('Chống tràn số tuyệt đối với logit ±1000 (không sinh NaN/Inf)', lambda: (
        (lambda probs, loss: np.all(np.isfinite(probs)) and np.isfinite(loss) and np.allclose(probs.sum(axis=1), 1.0))(
            *softmax_cross_entropy([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]], [1, 0])
        ),
        'Phép trừ max(logits) giữ exp nằm trong [0, 1].'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Điều gì xảy ra nếu tập dữ liệu hoàn toàn tách biệt tuyến tính (linearly separable) và ta tối ưu Binary Cross-Entropy không có regularization $L_2$?',
          guidance: String.raw`Nếu tồn tại $w^*$ sao cho $y_i = 1 \implies x_i^\top w^* > 0$ và $y_i = 0 \implies x_i^\top w^* < 0$, thì khi nhân quy mô $\|w\|_2 \to \infty$ dọc theo phương $w^*$, mọi xác suất dự đoán tiến dần tới đúng $0$ hoặc $1$, kéo hàm mất mát về $0$. Vì vậy không tồn tại điểm cực tiểu hữu hạn nếu thiếu $L_2$ regularization — trong thực tế Gradient Descent hội tụ về hướng bộ phân tách **Max-Margin**!`,
        },
      ],
    },
    {
      id: 'm2-u4',
      moduleId: 2,
      order: 4,
      title: 'Generalized Linear Models (GLMs) & Exponential Family',
      summary:
        'Khung thống nhất toán học đằng sau hầu hết các mô hình tuyến tính kinh điển: Họ phân phối mũ chính tắc, hai bổ đề đạo hàm của hàm log-partition $a(\\eta)$ tạo kỳ vọng và phương sai, cùng nguyên lý xây dựng Poisson GLM cho dữ liệu đếm.',
      outcomes: [
        'Viết lại các phân phối Gaussian, Bernoulli, Categorical và Poisson dưới dạng chính tắc $p(y; \\eta) = b(y)\\exp(\\eta^\\top T(y) - a(\\eta))$.',
        'Chứng minh hai tính chất vàng của hàm log-partition: $\\nabla a(\\eta) = \\mathbb{E}[T(y)]$ và $\\nabla^2 a(\\eta) = \\text{Cov}(T(y))$.',
        'Thiết kế hàm mất mát và gradient cho Poisson Regression ($\hat{\\lambda} = \\exp(w^\\top x + b)$) trong bài toán dự đoán dữ liệu đếm.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part I — Section 4 (Generalized Linear Models)',
          note: 'Dẫn xuất Exponential Family cho Gaussian, Bernoulli, Multinomial và thiết kế GLM.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§3.4 & §12 (Exponential Family & Generalized Linear Models)',
          note: 'Tính chất Cumulant Generating Function của log-partition a(η) và Poisson Regression.',
        },
      ],
      sections: [
        {
          heading: '1. Họ phân phối mũ (Exponential Family) & Các bổ đề Cumulant',
          body: String.raw`Một họ phân phối xác suất thuộc **Exponential Family** dạng chính tắc (canonical form) nếu hàm mật độ hoặc hàm khối xác suất viết được dưới dạng:

$$p(y; \eta) = b(y)\exp\!\big(\eta^\top T(y) - a(\eta)\big)$$

trong đó $\eta$ là **tham số tự nhiên (natural parameter)**, $T(y)$ là **thống kê đủ (sufficient statistic)**, và $a(\eta) = \log \int b(y)\exp(\eta^\top T(y))\,dy$ là **hàm phân hoạch log (log-partition function)** bảo đảm tổng xác suất bằng $1$.

Lấy đạo hàm theo $\eta$ hai bên đẳng thức $\int p(y; \eta)\,dy = 1$, ta nhận được hai hệ quả cực đẹp:`,
          math: String.raw`\nabla_\eta a(\eta) = \mathbb{E}[T(y)], \qquad \nabla_\eta^2 a(\eta) = \text{Cov}(T(y)) \succeq 0`,
          intuition:
            'Vì hiệp phương sai luôn nửa xác định dương, $a(\\eta)$ luôn là **hàm lồi** theo $\\eta$ — đây chính là lý do sâu xa giải thích vì sao Negative Log-Likelihood của mọi GLM với hàm liên kết chính tắc đều là bài toán tối ưu lồi!',
        },
        {
          heading: '2. Bức tranh thống nhất GLM & Poisson Regression cho dữ liệu đếm',
          body: String.raw`Để xây dựng một Generalized Linear Model, ta giả định $y \mid x \sim \text{ExpFamily}(\eta)$ với $\eta = w^\top x + b$ (liên kết chính tắc $\eta$ tuyến tính theo $x$). Khi đó mọi mô hình đều chia sẻ chung một cấu trúc gradient giống nhau đến kinh ngạc:

- **Gaussian ($\sigma^2$ cố định)**: $\mu = \eta$ (Identity link) $\implies$ OLS / MSE Loss.
- **Bernoulli**: $\mu = \frac{1}{1 + e^{-\eta}}$ (Sigmoid) $\implies$ Logistic Regression / BCE Loss.
- **Poisson** (dữ liệu đếm $y \in \{0, 1, 2, \dots\}$): Phân phối $p(y; \lambda) = \frac{\lambda^y e^{-\lambda}}{y!} = \frac{1}{y!}\exp(y \ln\lambda - \lambda)$, suy ra $\eta = \ln\lambda \implies \mu = \mathbb{E}[y \mid x] = e^{\eta} = e^{w^\top x + b}$.`,
          math: String.raw`\mathcal{L}_{\text{Poisson}}(w, b) = \frac{1}{n}\sum_{i=1}^n \Big(e^{\eta_i} - y_i \eta_i\Big), \qquad \nabla_w \mathcal{L} = \frac{1}{n} X^\top (\mu - y)`,
          pitfall:
            'Dùng Hồi quy tuyến tính OLS cho biến đếm (số lượt truy cập, số tai nạn) có thể dự đoán ra giá trị **âm** phi lý và bỏ qua tính phương sai tăng theo kỳ vọng ($\\text{Var}(y) = \\mathbb{E}[y] = \\lambda$). Poisson GLM bảo đảm $\\hat{y} > 0$ tại mọi $x$.',
        },
      ],
      simulationId: 'm2-u4',
      exercises: [
        {
          id: 'm2-u4-poisson-glm',
          title: 'Poisson Regression GLM: Negative Log-Likelihood & Gradient',
          prompt:
            'Cài đặt `poisson_nll_and_grad(X, y, w, b)` cho Poisson GLM với liên kết chính tắc $\\eta = Xw + b$, $\\mu = \\exp(\\eta)$. Trả về tuple `(nll, dw, db)` trong đó `nll` là trung bình $(\\mu_i - y_i \\eta_i)$ (bỏ hằng số $\\log(y_i!)$ không phụ thuộc tham số) và `dw, db` là gradient tương ứng.',
          hint: 'Tính `eta = X @ w + float(b)`, `mu = np.exp(eta)`, `nll = float(np.mean(mu - y * eta))`. Sau đó `err = (mu - y) / len(y)`, `dw = X.T @ err`, `db = float(np.sum(err))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xp = np.array([[0.2, -0.5], [1.1, 0.3], [-0.4, 1.2], [0.8, 0.7]])
_yp = np.array([1., 5., 2., 4.])
_wp = np.array([0.4, 0.25])
_bp = 0.5

def _num_poisson_grad(X, y, w, b, eps=1e-6):
    def f(tw, tb):
        eta = X @ tw + tb
        return float(np.mean(np.exp(eta) - y * eta))
    gw = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = eps
        gw[j] = (f(w + e, b) - f(w - e, b)) / (2 * eps)
    gb = (f(w, b + eps) - f(w, b - eps)) / (2 * eps)
    return gw, gb
`,
          starter: String.raw`def poisson_nll_and_grad(X, y, w, b):
    raise NotImplementedError
`,
          solution: String.raw`def poisson_nll_and_grad(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    eta = X @ w + float(b)
    mu = np.exp(eta)
    nll = float(np.mean(mu - y * eta))
    err = (mu - y) / len(y)
    dw = X.T @ err
    db = float(np.sum(err))
    return nll, dw, db
`,
          test: String.raw`_report('m2-u4-poisson-glm',
    _check('Gradient dw, db khớp chính xác với sai phân trung tâm (< 1e-6)', lambda: (
        (lambda nll, dw, db: (
            (lambda ngw, ngb: np.allclose(dw, ngw, atol=1e-6) and abs(db - ngb) < 1e-6)(*_num_poisson_grad(_Xp, _yp, _wp, _bp))
        ))(*poisson_nll_and_grad(_Xp, _yp, _wp, _bp)),
        'Kiểm chứng dạng gradient phổ quát Xᵀ(μ - y)/n của mọi GLM chính tắc.'
    )),
    _check('Giá trị NLL khớp định nghĩa (e^η - yη)/n trên toàn tập mẫu', lambda: (
        (lambda res: np.isclose(res[0], float(np.mean(np.exp(_Xp @ _wp + _bp) - _yp * (_Xp @ _wp + _bp)))))(
            poisson_nll_and_grad(_Xp, _yp, _wp, _bp)
        ),
        'Đối chiếu công thức log-likelihood Poisson.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question: String.raw`Trong thực tế dữ liệu đếm thường gặp hiện tượng **overdispersion** ($\text{Var}(y \mid x) > \mathbb{E}[y \mid x]$), vi phạm giả định của phân phối Poisson. Làm sao phát hiện hiện tượng này và mô hình thay thế nào khắc phục nó trong họ GLM?`,
          guidance:
            'Ta kiểm tra tỉ số Pearson chi-squared chia bậc tự do trên tập validation; nếu tỉ số vượt xa $1.0$ nghĩa là có overdispersion. Mô hình **Negative Binomial Regression** đưa thêm tham số phân tán $\\alpha$ sao cho $\\text{Var}(y) = \\mu + \\alpha \\mu^2$, khắc phục việc ước lượng sai lệch chuẩn của hệ số hồi quy.',
        },
      ],
    },
  ],
};
