import type { GradModule } from './types';

const COMMON_SETUP = String.raw`import ast, inspect, json, math
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

export const MODULE_01: GradModule = {
  id: 1,
  title: 'Nền tảng Toán học, Xác suất & Lý thuyết Học thống kê',
  englishTitle: 'Mathematical Foundations, Statistical Inference & Learning Theory',
  color: '#42d3e5',
  status: 'ready',
  overview:
    'Thiết lập ngôn ngữ hình thức cho học máy ở bậc cao học: biểu diễn dữ liệu trong không gian vector, giải tích ma trận cho tối ưu hóa, suy luận xác suất MLE/MAP, phân rã Bias–Variance và các chặn tổng quát hóa PAC/VC-Dimension cùng quy trình thực nghiệm chống rò rỉ dữ liệu.',
  outcomes: [
    'Thực hiện giải tích ma trận (Matrix Calculus) và cài đặt các phép biến đổi vector hóa bằng NumPy không dùng vòng lặp.',
    'Chứng minh mối liên hệ tương đương giữa MLE Gaussian với MSE, và MAP Gaussian Prior với Regularization $L_2$ (Ridge).',
    'Phân rã sai số tổng quát hóa thành $\\text{Bias}^2 + \\text{Variance} + \\sigma^2$ và kiểm chứng bằng thực nghiệm Monte Carlo.',
    'Phát biểu chặn tổng quát hóa Hoeffding/VC-Dimension và thiết kế quy trình tách tập train/validation/test không rò rỉ dữ liệu.',
  ],
  sources: [
    'Kevin P. Murphy — Probabilistic Machine Learning: An Introduction (Ch. 2–5, Ch. 7)',
    'Stanford CS229 — Machine Learning Complete Notes (Part I, Learning Theory)',
    'UC Berkeley CS189/289A — Lecture 01–04, Lecture 09',
    'ISLR 7th Printing (Ch. 2, Ch. 5)',
    'Vũ Hữu Tiệp — Machine Learning cơ bản (Ch. 1–3)',
  ],
  units: [
    {
      id: 'm1-u1',
      moduleId: 1,
      order: 1,
      title: 'Không gian đặc trưng, Đại số tuyến tính & Matrix Calculus',
      summary:
        'Biểu diễn tập dữ liệu dạng ma trận thiết kế $X \\in \\mathbb{R}^{n \\times d}$, hình học của chuẩn và phép chiếu trực giao, kỹ thuật vector hóa NumPy và tính gradient dạng ma trận.',
      outcomes: [
        'Phân biệt hình học giữa các chuẩn $L_1, L_2, L_\\infty$ và chuẩn Frobenius trên không gian đặc trưng.',
        'Sử dụng broadcasting NumPy để tính khoảng cách cặp và chuẩn hóa Z-score ở độ phức tạp tối ưu không cần vòng lặp.',
        'Tính chính xác gradient theo vector tham số $w$ cho hàm tuyến tính và dạng toàn phương $\\|Xw - y\\|_2^2$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§7.1–7.5 (Linear Algebra & Matrix Calculus)',
          note: 'Định nghĩa chuẩn vector, ma trận xác định dương và quy tắc lấy đạo hàm theo vector.',
        },
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 2/Lecture 02 -- Data Tools.pdf',
          section: 'Toàn bộ slide Lecture 02',
          note: 'Biểu diễn ma trận dữ liệu, broadcasting và chuẩn hóa đặc trưng.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/book_ML.pdf',
          section: 'Chương 1: Ôn tập Đại số tuyến tính',
          note: 'Nhắc lại tích vô hướng, chuẩn và tích ma trận–vector.',
        },
      ],
      sections: [
        {
          heading: '1. Ma trận thiết kế, Chuẩn vector & Hình học không gian đặc trưng',
          body: String.raw`Trong học có giám sát, mỗi quan sát được biểu diễn dưới dạng vector cột $x_i \in \mathbb{R}^d$, ghép theo từng hàng thành ma trận thiết kế $X \in \mathbb{R}^{n \times d}$, cùng vector nhãn mục tiêu $y \in \mathbb{R}^n$.

Để đo độ lớn sai số hoặc ràng buộc độ phức tạp của tham số $w \in \mathbb{R}^d$, ta sử dụng chuẩn $\ell_p$:
- **Chuẩn $\ell_2$ (Euclidean)**: hình cầu đẳng hướng, trơn khả vi tại mọi điểm khác $0$.
- **Chuẩn $\ell_1$ (Manhattan)**: mặt cầu là khối bát diện có góc nhọn trên các trục tọa độ, thúc đẩy tính thưa (sparsity).
- **Chuẩn $\ell_\infty$**: đo sai lệch lớn nhất theo một tọa độ đơn lẻ.`,
          math: String.raw`\|x\|_p = \left(\sum_{j=1}^d |x_j|^p\right)^{1/p}, \quad \|x\|_1 = \sum_{j=1}^d |x_j|, \quad \|x\|_2 = \sqrt{x^\top x}, \quad \|x\|_\infty = \max_{j} |x_j|`,
          intuition:
            'Khoảng cách Euclidean bình phương giữa hai tập điểm $A \\in \\mathbb{R}^{n \\times d}$ và $B \\in \\mathbb{R}^{m \\times d}$ có thể khai triển qua tích vô hướng hoặc tính trực tiếp bằng broadcasting 3 chiều $(n, 1, d) - (1, m, d)$ mà không cần vòng lặp Python.',
          pitfall:
            'Chuẩn hóa Z-score với cột hằng (độ lệch chuẩn $\\sigma_j = 0$) sẽ tạo ra `NaN` hoặc `Inf` nếu không thay mẫu số bằng $1$ trước khi chia.',
        },
        {
          heading: '2. Tích vô hướng & Phép chiếu trực giao',
          body: String.raw`Mô hình tuyến tính cơ bản thực chất tính tích vô hướng $x^\top w + b$:
- Về đại số, đây là tổ hợp tuyến tính của các đặc trưng.
- Về hình học, $(w^\top x) / \|w\|_2$ là độ dài hình chiếu có dấu của vector dữ liệu $x$ lên phương của vector pháp tuyến $w$. Siêu phẳng quyết định $\{x : w^\top x + b = 0\}$ luôn vuông góc với vector $w$.`,
          math: String.raw`\text{proj}_w(x) = \frac{w^\top x}{\|w\|_2^2}\, w, \qquad Xw + b\mathbf{1}_n = \hat{y} \in \mathbb{R}^n`,
          intuition:
            'Trong lập trình ma trận, quy ước một mẫu nằm trên một hàng của $X$ cho phép biểu thức `X @ w + b` tự động broadcast bias vô hướng $b$ ra toàn bộ $n$ mẫu cùng lúc.',
        },
        {
          heading: '3. Giải tích Ma trận (Matrix Calculus) cho Tối ưu hóa',
          body: String.raw`Thay vì lấy đạo hàm riêng từng phần tử $w_1, \dots, w_d$ rồi ghép lại, ta đạo hàm trực tiếp trên không gian vector $\mathbb{R}^d$ (theo quy ước mẫu số — cùng kích thước với tham số $w$).

Ba đồng nhất thức cốt lõi dùng xuyên suốt các mô hình học máy:
- Gradient của dạng tuyến tính: $\nabla_w (w^\top x) = x$.
- Gradient của dạng toàn phương tổng quát: $\nabla_w (w^\top A w) = (A + A^\top)w$ (rút gọn thành $2Aw$ nếu $A$ đối xứng).
- Gradient của hàm mất mát bình phương trung bình (MSE) với $L(w) = \frac{1}{n}\|Xw - y\|_2^2$.`,
          math: String.raw`L(w) = \frac{1}{n}(Xw - y)^\top(Xw - y) \implies \nabla_w L(w) = \frac{2}{n} X^\top (Xw - y) \in \mathbb{R}^d`,
          intuition:
            'Chú ý sự khớp chiều (dimensionality sanity check): $X$ có kích thước $(n, d)$, phần dư $(Xw - y)$ có kích thước $(n,)$, do đó $X^\top (Xw - y)$ có kích thước $(d, n) \\times (n,) = (d,)$ — trùng khớp chính xác với shape của $w$.',
          pitfall:
            'Lấy sai chiều chuyển vị $X(Xw - y)$ hoặc bỏ sót hệ số $\\frac{2}{n}$ sẽ làm sai bước cập nhật Gradient Descent theo kích thước tập mẫu $n$.',
        },
      ],
      simulationId: 'm1-u1',
      exercises: [
        {
          id: 'm1-u1-standardize',
          title: 'Chuẩn hóa Z-score theo từng cột (An toàn với cột hằng)',
          prompt:
            'Cài đặt `standardize(X)` biến đổi ma trận đầu vào $X \\in \\mathbb{R}^{n \\times d}$ về trung bình $0$ và độ lệch chuẩn $1$ trên mỗi cột (`ddof=0`). Nếu một cột có độ lệch chuẩn bằng $0$, giữ nguyên giá trị sau trừ trung bình (bằng $0$). Không được làm thay đổi mảng `X` gốc.',
          hint: 'Chuyển X sang float với `np.asarray(X, dtype=float)`, tính `mu = X.mean(axis=0)`, `sd = X.std(axis=0)`, sau đó thay `sd == 0` bằng `1.0` qua `np.where`.',
          setup:
            COMMON_SETUP +
            String.raw`
_RNG = np.random.default_rng(42)
_R = _RNG.normal([3., -2., 10.], [2., 0.5, 7.], size=(60, 3))
`,
          starter: String.raw`def standardize(X):
    raise NotImplementedError
`,
          solution: String.raw`def standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    safe_sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / safe_sd
`,
          test: String.raw`_report('m1-u1-standardize',
    _check('Mỗi cột có mean ≈ 0 và std(ddof=0) ≈ 1', lambda: (
        np.allclose(standardize(_R).mean(axis=0), 0.0) and np.allclose(standardize(_R).std(axis=0), 1.0),
        'Kiểm tra trên ma trận 60x3.'
    )),
    _check('Cột hằng cho kết quả 0 thay vì NaN', lambda: (
        np.allclose(standardize(np.array([[2., 5.], [4., 5.]])), [[-1., 0.], [1., 0.]]),
        'Cột thứ hai không đổi giá trị.'
    )),
    _check('Không sửa đổi mảng X đầu vào (immutability)', lambda: (
        (lambda orig: (standardize(orig), np.array_equal(orig, np.array([[2., 5.], [4., 5.]])))[1])(np.array([[2., 5.], [4., 5.]])),
        'Mảng gốc phải được giữ nguyên.'
    )),
)
`,
        },
        {
          id: 'm1-u1-pairwise',
          title: 'Khoảng cách Euclidean bình phương (Vectorized Broadcasting)',
          prompt:
            'Cài đặt `pairwise_sq_dists(A, B)` nhận `A` kích thước `(n, d)` và `B` kích thước `(m, d)`, trả về ma trận `(n, m)` chứa $\|a_i - b_j\|_2^2$. Bắt buộc vector hóa 100%, không dùng `for`, `while` hay comprehension.',
          hint: 'Dùng broadcasting 3D: `A[:, None, :] - B[None, :, :]` tạo tensor `(n, m, d)`, bình phương rồi `.sum(axis=2)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_A = np.array([[1., 2.], [-3., 4.]])
_B = np.array([[2., 2.], [7., 0.], [0., 0.]])
_loop_dist = np.array([[sum((a - b)**2) for b in _B] for a in _A])

def _no_loops(fn):
    forbidden = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    return not any(isinstance(node, forbidden) for node in ast.walk(ast.parse(inspect.getsource(fn))))
`,
          starter: String.raw`def pairwise_sq_dists(A, B):
    raise NotImplementedError
`,
          solution: String.raw`def pairwise_sq_dists(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff ** 2, axis=2)
`,
          test: String.raw`_report('m1-u1-pairwise',
    _check('Khớp kết quả vòng lặp chuẩn trên hai tập khác kích thước', lambda: (
        np.allclose(pairwise_sq_dists(_A, _B), _loop_dist) and pairwise_sq_dists(_A, _B).shape == (2, 3),
        'Shape kỳ vọng (2, 3).'
    )),
    _check('Thân hàm không chứa vòng lặp hay comprehension (AST check)', lambda: (
        (pairwise_sq_dists(_A, _B), _no_loops(pairwise_sq_dists))[1],
        'Phải dùng phép toán ma trận / broadcasting NumPy.'
    )),
)
`,
        },
        {
          id: 'm1-u1-quad-grad',
          title: 'Gradient giải tích cho hàm mất mát MSE tuyến tính',
          prompt:
            'Cho $L(w) = \\frac{1}{n}\\|Xw - y\\|_2^2$. Cài đặt `mse_gradient(X, y, w)` trả về vector gradient $\\nabla_w L(w) = \\frac{2}{n} X^\\top (Xw - y)$. Kết quả sẽ được đối chiếu với sai phân trung tâm (central finite differences).',
          hint: 'Chuyển `X`, `y`, `w` sang mảng float NumPy và trả về `(2.0 / len(y)) * (X.T @ (X @ w - y))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xg = np.array([[1., 2., -1.], [0.5, -1., 3.], [2., 0., 1.], [-1., 1., 2.]])
_yg = np.array([1.2, -0.5, 3.1, 0.8])
_wg = np.array([0.4, -0.3, 0.7])

def _numeric_grad(X, y, w, eps=1e-6):
    grad = np.zeros_like(w, dtype=float)
    for j in range(len(w)):
        e = np.zeros_like(w, dtype=float)
        e[j] = eps
        lp = np.mean((X @ (w + e) - y) ** 2)
        lm = np.mean((X @ (w - e) - y) ** 2)
        grad[j] = (lp - lm) / (2 * eps)
    return grad
`,
          starter: String.raw`def mse_gradient(X, y, w):
    raise NotImplementedError
`,
          solution: String.raw`def mse_gradient(X, y, w):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    return (2.0 / len(y)) * (X.T @ (X @ w - y))
`,
          test: String.raw`_report('m1-u1-quad-grad',
    _check('Gradient giải tích khớp sai phân hữu hạn trung tâm (< 1e-7)', lambda: (
        np.allclose(mse_gradient(_Xg, _yg, _wg), _numeric_grad(_Xg, _yg, _wg), atol=1e-7),
        'Đối chiếu từng tọa độ với central difference.'
    )),
    _check('Kích thước vector gradient khớp d = X.shape[1]', lambda: (
        np.asarray(mse_gradient(_Xg, _yg, _wg)).shape == (3,),
        'Vector gradient phải thuộc R^d.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Phân tích đánh đổi bộ nhớ (Memory–Compute Trade-off) giữa biểu thức broadcasting `(n, 1, d) - (1, m, d)` và khai triển hằng đẳng thức $\\|a_i\\|_2^2 + \\|b_j\\|_2^2 - 2 a_i^\\top b_j$. Khi $n = m = 50\\,000$ và $d = 512$, cách nào gây tràn RAM?',
          guidance:
            'Broadcasting 3D cấp phát tensor trung gian kích thước $O(n \\cdot m \\cdot d)$, tốn $\\approx 50\\,000^2 \\times 512 \\times 8$ bytes (> 10 TB). Trong khi khai triển tích vô hướng chỉ tốn bộ nhớ $O(n \\cdot m)$ thông qua phép nhân ma trận BLAS cấp 3 (`A @ B.T`), dù cần chú ý sai số dấu phẩy động nếu khoảng cách rất nhỏ.',
        },
        {
          question:
            'Chứng minh ma trận Gram $X^\\top X \\in \\mathbb{R}^{d \\times d}$ luôn là ma trận nửa xác định dương (Positive Semi-Definite). Điều kiện cần và đủ để $X^\\top X$ khả nghịch là gì?',
          guidance:
            'Với mọi $v \\in \\mathbb{R}^d$, $v^\\top (X^\\top X) v = (Xv)^\\top (Xv) = \\|Xv\\|_2^2 \\ge 0$. Do đó $X^\\top X \\succeq 0$. Để khả nghịch (xác định dương), không được tồn tại $v \\neq 0$ sao cho $Xv = 0$, tức $\\text{rank}(X) = d$ (các cột của $X$ độc lập tuyến tính).',
        },
      ],
    },
    {
      id: 'm1-u2',
      moduleId: 1,
      order: 2,
      title: 'Xác suất, Ước lượng MLE, MAP & Suy luận Bayes',
      summary:
        'Khung xác suất thống kê đằng sau các hàm mất mát của học máy: phân biệt Likelihood và Xác suất, ước lượng hợp lý cực đại (MLE), ước lượng hậu nghiệm cực đại (MAP) với tiên nghiệm liên hợp Beta–Binomial và Gaussian–Ridge.',
      outcomes: [
        'Phân biệt đúng hàm hợp lý $L(\\theta; \\mathcal{D}) = p(\\mathcal{D} \\mid \\theta)$ với phân phối hậu nghiệm $p(\\theta \\mid \\mathcal{D})$.',
        'Giải bài toán tìm $\\hat{\\theta}_{\\text{MLE}}$ bằng cực đại hóa Log-Likelihood và chứng minh Gaussian MLE tương đương tối thiểu hóa MSE.',
        'Áp dụng định lý Bayes cho cặp liên hợp Beta–Binomial và chứng minh Gaussian Prior tương đương thêm phạt Regularization $L_2$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: 'Ch. 2–4 (Probability, Statistics, Decision Theory)',
          note: 'Định lý Bayes, MLE, MAP, phân phối liên hợp Beta–Binomial và suy luận phân phối dự đoán hậu nghiệm.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part I — Probabilistic Interpretation of Linear Regression',
          note: 'Chứng minh giả định nhiễu Gaussian i.i.d. dẫn trực tiếp tới hàm mất mát bình phương tối thiểu.',
        },
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 3/Lecture 04 -- K-means and Probability.pdf',
          section: 'Likelihood & Maximum Likelihood Estimation',
          note: 'Ví dụ MLE và trực giác suy luận tham số từ mẫu.',
        },
      ],
      sections: [
        {
          heading: '1. Hàm hợp lý (Likelihood) & Ước lượng Hợp lý Cực đại (MLE)',
          body: String.raw`Cho tập dữ liệu $\mathcal{D} = \{x_1, \dots, x_n\}$ được sinh độc lập và đồng phân phối (i.i.d.) từ phân phối tham số $p(x \mid \theta)$. Nguyên lý MLE chọn bộ tham số $\theta$ làm cho dữ liệu thực tế quan sát được có khả năng xảy ra cao nhất. Vì hàm $\log$ đơn điệu tăng, việc chuyển tích xác suất sang tổng Log-Likelihood giúp tránh tràn số dưới (underflow) và thuận tiện lấy đạo hàm:`,
          math: String.raw`\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^n p(x_i \mid \theta) = \arg\max_{\theta} \sum_{i=1}^n \log p(x_i \mid \theta)`,
          intuition:
            'Với phân phối chuẩn 1 chiều $\\mathcal{N}(\\mu, \\sigma^2)$, đặt đạo hàm Log-Likelihood bằng $0$ cho kết quả giải tích: $\\hat{\\mu}_{\\text{MLE}} = \\frac{1}{n}\\sum x_i$ và $\\hat{\\sigma}^2_{\\text{MLE}} = \\frac{1}{n}\\sum (x_i - \\hat{\\mu})^2$.',
          pitfall:
            'Hàm hợp lý $L(\\theta; \\mathcal{D})$ **không phải là phân phối xác suất theo $\\theta$** (tích phân theo $\\theta$ không bằng $1$). Ngoài ra, khi cỡ mẫu $n$ nhỏ (ví dụ tung đồng xu $3$ lần đều ra ngửa), $\\hat{\\theta}_{\\text{MLE}} = 1.0$ dễ bị cực đoan hóa (overconfident).',
        },
        {
          heading: '2. Ước lượng Hậu nghiệm Cực đại (MAP) & Tiên nghiệm liên hợp Beta–Binomial',
          body: String.raw`Thay vì xem $\theta$ là hằng số chưa biết (như trường phái tần suất), suy luận Bayes gán cho $\theta$ một phân phối tiên nghiệm (Prior) $p(\theta)$ phản ánh tri thức trước khi thấy dữ liệu, rồi cập nhật thành phân phối hậu nghiệm (Posterior) theo Định lý Bayes: $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$.

Với dữ liệu Bernoulli có $k$ lần thành công trong $n$ phép thử và tiên nghiệm liên hợp $\theta \sim \text{Beta}(\alpha, \beta)$, phân phối hậu nghiệm vẫn là $\text{Beta}(\alpha + k, \beta + n - k)$:`,
          math: String.raw`\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right] = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} \quad (\alpha + k > 1,\; \beta + n - k > 1)`,
          intuition:
            'Các siêu tham số $\\alpha - 1$ và $\\beta - 1$ đóng vai trò như số "quan sát giả" (pseudo-counts) giúp làm mượt ước lượng khi dữ liệu thưa. Khi $n \\to \\infty$, đóng góp của prior mờ dần và $\\hat{\\theta}_{\\text{MAP}} \\to \\hat{\\theta}_{\\text{MLE}}$.',
        },
        {
          heading: '3. Cầu nối Xác suất – Tối ưu: Từ MLE/MAP tới MSE và Ridge ($L_2$)',
          body: String.raw`Xác suất cung cấp cơ sở lý thuyết chặt chẽ lý giải vì sao các hàm mất mát tiêu chuẩn có dạng như hiện nay:
- **Gaussian Likelihood $\equiv$ MSE Loss**: Nếu $y_i = w^\top x_i + \varepsilon_i$ với $\varepsilon_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$, tối đa hóa $\log p(y \mid X, w)$ tương đương tối thiểu hóa $\sum_{i=1}^n (y_i - w^\top x_i)^2$.
- **Gaussian Prior $\equiv$ Ridge ($L_2$)**: Nếu đặt thêm tiên nghiệm $w \sim \mathcal{N}(0, \tau^2 I)$, số hạng $-\log p(w) = \frac{1}{2\tau^2}\|w\|_2^2 + \text{const}$ chính xác là thành phần phạt Ridge với $\lambda = \frac{\sigma^2}{\tau^2}$!`,
          math: String.raw`\hat{w}_{\text{MAP}} = \arg\min_{w} \left[ \frac{1}{2\sigma^2}\|Xw - y\|_2^2 + \frac{1}{2\tau^2}\|w\|_2^2 \right]`,
          intuition:
            'Tiên nghiệm phương sai $\\tau^2$ càng nhỏ (niềm tin mạnh rằng các trọng số gần $0$), hệ số điều chuẩn $\\lambda = \\sigma^2 / \\tau^2$ càng lớn. Tương tự, tiên nghiệm Laplace cho ra phạt Lasso $L_1$.',
        },
      ],
      simulationId: 'm1-u2',
      exercises: [
        {
          id: 'm1-u2-gaussian-mle',
          title: 'Gaussian 1D MLE & Tính Log-Likelihood',
          prompt:
            'Cài đặt `gaussian_mle_loglik(x)` nhận vector quan sát `x` gồm $n$ số thực, trả về tuple `(mu_hat, var_hat, log_lik)` trong đó `mu_hat` là $\\hat{\\mu}_{\\text{MLE}}$, `var_hat` là phương sai MLE (chia cho $n$, tức `ddof=0`), và `log_lik` là giá trị tổng $\\sum_{i=1}^n \\log \\mathcal{N}(x_i \\mid \\hat{\\mu}_{\\text{MLE}}, \\hat{\\sigma}^2_{\\text{MLE}})$.',
          hint: 'Dùng công thức rút gọn hoặc tính trực tiếp `-0.5 * n * np.log(2 * np.pi * var_hat) - 0.5 * n` (khi thay MLE vào phương trình log-likelihood). Kiểm tra lại bằng công thức từng điểm `-0.5 * np.sum(np.log(2 * np.pi * var_hat) + ((x - mu_hat)**2) / var_hat)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X1d = np.array([2.1, 2.5, 3.3, 1.9, 2.7, 3.0])
`,
          starter: String.raw`def gaussian_mle_loglik(x):
    raise NotImplementedError
`,
          solution: String.raw`def gaussian_mle_loglik(x):
    x = np.asarray(x, dtype=float)
    mu_hat = float(np.mean(x))
    var_hat = float(np.mean((x - mu_hat) ** 2))
    ll = float(np.sum(-0.5 * np.log(2.0 * np.pi * var_hat) - ((x - mu_hat) ** 2) / (2.0 * var_hat)))
    return mu_hat, var_hat, ll
`,
          test: String.raw`_report('m1-u2-gaussian-mle',
    _check('Ước lượng mu_hat và var_hat (chia n, ddof=0) chính xác', lambda: (
        (lambda res: np.isclose(res[0], np.mean(_X1d)) and np.isclose(res[1], np.var(_X1d, ddof=0)))(gaussian_mle_loglik(_X1d)),
        'Lưu ý dùng chia n (ddof=0), không phải n-1.'
    )),
    _check('Giá trị Log-Likelihood khớp định nghĩa tại MLE', lambda: (
        (lambda res: np.isclose(res[2], -0.5 * len(_X1d) * (np.log(2 * np.pi * res[1]) + 1.0)))(gaussian_mle_loglik(_X1d)),
        'Tại điểm MLE, tổng (x_i - mu)^2 / var = n.'
    )),
)
`,
        },
        {
          id: 'm1-u2-beta-map',
          title: 'So sánh MLE, MAP & Posterior Mean trên mô hình Beta–Binomial',
          prompt:
            'Cho $k$ lần thành công trên $n$ phép thử Bernoulli với tiên nghiệm $\\text{Beta}(\\alpha, \\beta)$ (với $\\alpha, \\beta \\ge 1$). Cài đặt `beta_binomial_estimates(k, n, alpha, beta)` trả về tuple `(mle, map_est, post_mean)` tương ứng với ước lượng MLE, mode của hậu nghiệm (MAP), và kỳ vọng hậu nghiệm $\\mathbb{E}[\\theta \\mid k, n]$.',
          hint: '`mle = k / n`; hậu nghiệm là `Beta(alpha + k, beta + n - k)`, nên `post_mean = (alpha + k) / (alpha + beta + n)` và khi $a = \\alpha + k > 1, b = \\beta + n - k > 1$ thì `map_est = (a - 1) / (a + b - 2)`.',
          setup: COMMON_SETUP,
          starter: String.raw`def beta_binomial_estimates(k, n, alpha, beta):
    raise NotImplementedError
`,
          solution: String.raw`def beta_binomial_estimates(k, n, alpha, beta):
    mle = k / n
    a_post = alpha + k
    b_post = beta + (n - k)
    denom_map = a_post + b_post - 2.0
    map_est = (a_post - 1.0) / denom_map if denom_map > 0 else mle
    post_mean = a_post / (a_post + b_post)
    return float(mle), float(map_est), float(post_mean)
`,
          test: String.raw`_report('m1-u2-beta-map',
    _check('Tiên nghiệm đều Beta(1, 1) cho MAP trùng hoàn toàn với MLE', lambda: (
        (lambda res: np.allclose(res, [0.7, 0.7, (1 + 7) / (1 + 1 + 10)]))(beta_binomial_estimates(7, 10, 1, 1)),
        'Kỳ vọng hậu nghiệm (1+7)/(12) khác MAP/MLE.'
    )),
    _check('Tiên nghiệm thông tin Beta(4, 4) kéo ước lượng khi n nhỏ', lambda: (
        np.allclose(beta_binomial_estimates(3, 3, 4, 4), [1.0, (3 + 3) / (3 + 4 + 4 - 2), (4 + 3) / (4 + 4 + 3)]),
        'MLE=1.0 bị quá tự tin, MAP và Posterior Mean được làm mượt về 0.5.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao ước lượng phương sai $\\hat{\\sigma}^2_{\\text{MLE}} = \\frac{1}{n}\\sum_{i=1}^n (x_i - \\bar{x})^2$ lại bị chệch hệ thống (biased) khi cỡ mẫu nhỏ? Chứng minh $\\mathbb{E}[\\hat{\\sigma}^2_{\\text{MLE}}] = \\frac{n-1}{n}\\sigma^2$.',
          guidance:
            'Vì $\\bar{x}$ được ước lượng từ chính $n$ điểm dữ liệu đó nên tổng bình phương quanh $\\bar{x}$ luôn nhỏ hơn tổng bình phương quanh $\\mu$ thật một lượng bằng $n(\\bar{x} - \\mu)^2$. Lấy kỳ vọng: $\\mathbb{E}[\\sum (x_i - \\mu)^2 - n(\\bar{x} - \\mu)^2] = n\\sigma^2 - n(\\sigma^2/n) = (n-1)\\sigma^2$.',
        },
        {
          question:
            'Cho biến ví dụ $\\theta > 0$ và thực hiện đổi biến $\\phi = \\log(\\theta)$. So sánh tính bất biến khi đổi biến (reparameterization invariance) giữa ước lượng MLE và ước lượng MAP.',
          guidance:
            'MLE bất biến: $\\hat{\\phi}_{\\text{MLE}} = \\log \\hat{\\theta}_{\\text{MLE}}$ vì hàm hợp lý chỉ biến đổi giá trị theo tọa độ mới. Ngược lại, MAP **không bất biến** vì biến đổi mật độ xác suất hậu nghiệm kéo theo nhân tử Jacobian $|d\\theta / d\\phi| = e^\\phi$, làm dịch chuyển vị trí điểm cực đại.',
        },
      ],
    },
    {
      id: 'm1-u3',
      moduleId: 1,
      order: 3,
      title: 'Empirical Risk Minimization & Phân rã Bias–Variance',
      summary:
        'Khung hình thức chuẩn của học thống kê từ Risk thật tới Empirical Risk, chứng minh đầy đủ đẳng thức phân rã Bias–Variance trên sai số bình phương và khảo sát cân bằng giữa Underfitting và Overfitting.',
      outcomes: [
        'Phân biệt True Risk $R(h)$, Empirical Risk $\\hat{R}_n(h)$ và tác động của việc chọn lớp giả thuyết $\\mathcal{H}$.',
        'Chứng minh hoàn chỉnh định lý phân rã $\\mathbb{E}[(y - \\hat{f}(x))^2] = \\text{Bias}^2 + \\text{Var} + \\sigma^2$.',
        'Chẩn đoán hiện tượng Underfitting / Overfitting dựa trên đường cong học và hiệu ứng của điều chuẩn.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 5/Lecture 09 -- Regression (Cont.) and Bias-Variance Trade-off.pdf',
          section: 'Bias–Variance Decomposition',
          note: 'Chứng minh đại số phân rã sai số kỳ vọng theo phân phối sinh tập huấn luyện.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§2.2.2 (The Bias-Variance Trade-Off)',
          note: 'Hình minh họa các đường cong độ phức tạp và sự đánh đổi Bias–Variance.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Learning Theory — Bias/Variance & ERM',
          note: 'Phân tích Approximation Error vs Estimation Error.',
        },
      ],
      sections: [
        {
          heading: '1. True Risk vs. Empirical Risk Minimization (ERM)',
          body: String.raw`Mục tiêu tối thượng của học có giám sát là tìm giả thuyết $h \in \mathcal{H}$ cực tiểu hóa **Rủi ro kỳ vọng thật (True Risk)** trên phân phối dữ liệu chưa biết $P(X, Y)$. Vì $P(X, Y)$ không quan sát được, thuật toán học thay thế bằng cực tiểu hóa **Rủi ro thực nghiệm (Empirical Risk)** trên $n$ mẫu huấn luyện:`,
          math: String.raw`R(h) = \mathbb{E}_{(x, y)\sim P}[\ell(h(x), y)], \qquad \hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)`,
          intuition:
            'Sai lệch giữa mô hình tối ưu trong tập giả thuyết $\\mathcal{H}$ và quy luật thật gọi là **Approximation Error**; sai lệch do chỉ có $n$ mẫu hữu hạn thay vì toàn bộ phân phối gọi là **Estimation Error**.',
        },
        {
          heading: '2. Định lý Phân rã Bias–Variance (Chứng minh đầy đủ)',
          body: String.raw`Giả sử dữ liệu sinh bởi $y = f(x) + \varepsilon$ với $\mathbb{E}[\varepsilon] = 0$, $\text{Var}[\varepsilon] = \sigma^2$. Ký hiệu $\hat{f}(x) = \hat{f}_{\mathcal{D}}(x)$ là mô hình học được từ tập mẫu ngẫu nhiên $\mathcal{D}$ cỡ $n$, và $\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]$ là dự đoán trung bình qua vô số lần lấy mẫu $\mathcal{D}$ mới. Thêm bớt $\bar{f}(x)$ rồi khai triển kỳ vọng, các số hạng chéo triệt tiêu nhờ $\mathbb{E}[\varepsilon] = 0$ và $\mathbb{E}[\hat{f}(x) - \bar{f}(x)] = 0$:`,
          math: String.raw`\mathbb{E}_{\mathcal{D}, \varepsilon}\!\left[(y - \hat{f}(x))^2\right] = \underbrace{\big(f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\big)^2}_{\text{Bias}^2[\hat{f}(x)]} + \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\big(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\big)^2\right]}_{\text{Var}[\hat{f}(x)]} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}`,
          intuition:
            '**Bias²** đo độ lệch hệ thống do lớp giả thuyết quá đơn giản không diễn tả nổi $f(x)$. **Variance** đo độ nhạy của mô hình khi tập train bị nhiễu hoặc thay đổi đôi chút. **Nhiễu $\\sigma^2$** là cận dưới bất khả kháng của mọi mô hình.',
          pitfall:
            'Nhầm lẫn giữa độ nhạy của tập kiểm thử cố định với phương sai qua các tập train $\\mathcal{D}$. Trong thực nghiệm, muốn đo trực tiếp Bias và Variance cần mô phỏng lấy mẫu lại (bootstrap hoặc Monte Carlo nhiều tập $\\mathcal{D}_1, \\dots, \\mathcal{D}_M$).',
        },
        {
          heading: '3. Điều chuẩn (Regularization) & Ghi chú về Double Descent',
          body: String.raw`Trong miền cổ điển (classical regime), khi độ phức tạp mô hình (bậc đa thức $d$, chiều sâu cây, số tham số) tăng: Bias giảm dần còn Variance tăng vọt, tạo ra đường cong tổng lỗi Test hình chữ U. Kỹ thuật Regularization (như Ridge $\lambda \|w\|_2^2$) chấp nhận hy sinh một chút Bias tăng nhẹ để kéo giảm mạnh Variance.

Ở chế độ hiện đại (over-parameterized regime với mạng sâu có số tham số $p \gg n$), đường cong rủi ro Test có thể giảm trở lại sau đỉnh nội suy (*Double Descent*) nhờ thiên vị ngầm (implicit bias) hướng tới nghiệm chuẩn cực tiểu.`,
          intuition:
            'Không phải mọi mô hình nhiều tham số đều tất yếu overfit nếu có điều chuẩn tường minh hoặc điều chuẩn ngầm từ thuật toán tối ưu như SGD.',
        },
      ],
      simulationId: 'm1-u3',
      exercises: [
        {
          id: 'm1-u3-bv-decomp',
          title: 'Ước lượng thực nghiệm Bias², Variance & Tổng MSE kỳ vọng',
          prompt:
            'Cho ma trận dự đoán `preds` kích thước `(M, N)` của `M` mô hình huấn luyện độc lập trên `M` tập mẫu, đánh giá tại `N` điểm kiểm thử có giá trị hàm thật `f_true` kích thước `(N,)` và phương sai nhiễu `noise_var` ($\\sigma^2$). Cài đặt `bias_variance_decomp(preds, f_true, noise_var)` trả về tuple `(bias_sq, variance, expected_mse)` lấy trung bình trên `N` điểm.',
          hint: 'Tính dự đoán trung bình theo trục mô hình `mean_pred = preds.mean(axis=0)` (shape `(N,)`). Khi đó `bias_sq = np.mean((mean_pred - f_true)**2)`, `variance = np.mean(np.var(preds, axis=0))`, và `expected_mse = bias_sq + variance + noise_var`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_bv = np.random.default_rng(7)
_f_true = np.sin(np.linspace(0, np.pi, 20))
_preds = _f_true[None, :] + 0.3 + _rng_bv.normal(0.0, 0.25, size=(50, 20))
_noise_var = 0.04
`,
          starter: String.raw`def bias_variance_decomp(preds, f_true, noise_var):
    raise NotImplementedError
`,
          solution: String.raw`def bias_variance_decomp(preds, f_true, noise_var):
    preds = np.asarray(preds, dtype=float)
    f_true = np.asarray(f_true, dtype=float)
    mean_pred = np.mean(preds, axis=0)
    bias_sq = float(np.mean((mean_pred - f_true) ** 2))
    variance = float(np.mean(np.var(preds, axis=0)))
    total = float(bias_sq + variance + noise_var)
    return bias_sq, variance, total
`,
          test: String.raw`_report('m1-u3-bv-decomp',
    _check('Tổng Bias² + Variance + noise_var khớp đẳng thức kỳ vọng trực tiếp E[(y - f_hat)^2]', lambda: (
        (lambda b, v, tot: np.isclose(tot, b + v + _noise_var) and np.isclose(tot, np.mean((_preds - _f_true[None, :]) ** 2) + _noise_var))(
            *bias_variance_decomp(_preds, _f_true, _noise_var)
        ),
        'Kiểm tra bảo toàn tổng sai số bình phương.'
    )),
    _check('Tách đúng thành phần dịch chuyển hệ thống (Bias²)', lambda: (
        bias_variance_decomp(_preds, _f_true, _noise_var)[0] > 0.05 and np.isclose(bias_variance_decomp(np.tile(_f_true + 0.5, (10, 1)), _f_true, 0.0)[0], 0.25),
        'Mô hình dịch hằng +0.5 có Bias²=0.25 và Variance=0.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong phòng lab Bias–Variance, nếu tăng kích thước tập huấn luyện $n$ lên gấp 10 lần nhưng giữ nguyên mô hình đa thức bậc 1 cho dữ liệu hàm $\\sin(\\pi x)$, đại lượng nào trong 3 thành phần ($\\text{Bias}^2$, $\\text{Variance}$, $\\sigma^2$) sẽ thay đổi rõ rệt nhất và đại lượng nào hầu như bất biến?',
          guidance:
            'Khi tăng $n$, Variance của đường thẳng giảm dần về $0$ vì các tập mẫu khác nhau cho ước lượng tham số ổn định hơn. Tuy nhiên, $\\text{Bias}^2$ (sai số xấp xỉ giữa đường thẳng và đường cong $\\sin$) và $\\sigma^2$ hầu như không giảm.',
        },
        {
          question:
            'Chứng minh trực tiếp từ công thức nghiệm đóng Ridge $\\hat{w}_\\lambda = (X^\\top X + \\lambda I)^{-1} X^\\top y$ rằng khi tăng $\\lambda > 0$, phương sai của $\\hat{w}_\\lambda$ giảm xuống so với OLS ($\\lambda = 0$).',
          guidance:
            'Ta có $\\text{Cov}(\\hat{w}_\\lambda) = \\sigma^2 (X^\\top X + \\lambda I)^{-1} X^\\top X (X^\\top X + \\lambda I)^{-1}$. Theo phân rã giá trị riêng $X^\\top X = Q \\Lambda Q^\\top$, các trị riêng của ma trận hiệp phương sai tỉ lệ với $\\frac{s_i}{(s_i + \\lambda)^2}$, giảm đơn điệu khi $\\lambda$ tăng.',
        },
      ],
    },
    {
      id: 'm1-u4',
      moduleId: 1,
      order: 4,
      title: 'Lý thuyết Tổng quát hóa (PAC, VC-Dimension) & Quy trình Thực nghiệm Chuẩn',
      summary:
        'Bảo chứng toán học cho việc học từ mẫu hữu hạn: bất đẳng thức Hoeffding, độ phức tạp mẫu trong khung PAC Learning, chiều VC (VC-Dimension) và quy trình tách dữ liệu chuẩn chống rò rỉ dữ liệu (Data Leakage).',
      outcomes: [
        'Sử dụng bất đẳng thức Hoeffding và Union Bound để suy ra chặn sai số tổng quát hóa và độ phức tạp mẫu $n$ theo $(\\epsilon, \\delta, |\\mathcal{H}|)$.',
        'Hiểu khái niệm Shattering và chiều VC ($\\text{VC}(\\mathcal{H}) = d+1$ cho bộ phân loại tuyến tính trong $\\mathbb{R}^d$).',
        'Triển khai chính xác phép chia Train / Validation / Test tái lập được và ngăn chặn hoàn toàn rò rỉ dữ liệu tiền xử lý.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part IV — Learning Theory (Hoeffding, Uniform Convergence, VC Dimension)',
          note: 'Chứng minh đầy đủ định lý hội tụ đều cho tập giả thuyết hữu hạn và vô hạn.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§5.1 (Cross-Validation & Validation Set Approach)',
          note: 'Cơ sở thống kê của đánh giá chéo và ước lượng lỗi kiểm thử.',
        },
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 2/Lecture 03 -- ML Mechanics - Techniques.pdf',
          section: 'Train / Val / Test & Avoiding Data Leakage',
          note: 'Kỷ luật thực nghiệm bảo vệ tập test và fit pipeline bên trong tập train.',
        },
      ],
      sections: [
        {
          heading: '1. Bất đẳng thức Hoeffding & Độ phức tạp mẫu PAC (|H| hữu hạn)',
          body: String.raw`Với hàm mất mát $0\text{--}1$, cho bất kỳ giả thuyết cố định $h \in \mathcal{H}$, bất đẳng thức Hoeffding giới hạn xác suất sai lệch giữa rủi ro thật $R(h)$ và rủi ro thực nghiệm $\hat{R}_n(h)$ trên $n$ mẫu i.i.d. Áp dụng Union Bound qua toàn bộ $k = |\mathcal{H}|$ giả thuyết, với xác suất ít nhất $1 - \delta$ (khung Probably Approximately Correct — PAC), mọi $h \in \mathcal{H}$ đều thỏa mãn hội tụ đều:`,
          math: String.raw`|R(h) - \hat{R}_n(h)| \le \sqrt{\frac{\ln(2|\mathcal{H}| / \delta)}{2n}} \quad \Longleftrightarrow \quad n \ge \frac{1}{2\epsilon^2} \ln\frac{2|\mathcal{H}|}{\delta}`,
          intuition:
            'Đáng chú ý, số lượng mẫu $n$ chỉ tăng theo $\\ln |\\mathcal{H}|$ (tốc độ logarit), nên việc xét hàng tỉ bộ tham số có thể được kiểm soát bằng lượng dữ liệu khả thi. Tuy nhiên nếu tập giả thuyết vô hạn ($\\mathcal{H}$ liên tục), chặn chứa $\\ln |\\mathcal{H}|$ trở nên vô nghĩa.',
        },
        {
          heading: '2. Shattering & Chiều VC (VC-Dimension) cho lớp giả thuyết vô hạn',
          body: String.raw`Để đo sức biểu diễn của một lớp giả thuyết vô hạn $\mathcal{H}$, Vapnik và Chervonenkis định nghĩa **Chiều VC ($\text{VC}(\mathcal{H})$)** là số lượng điểm tối đa mà $\mathcal{H}$ có thể **phá vỡ (shatter)** — tức chia cắt đúng cho mọi tổ hợp $2^m$ phép gán nhãn nhị phân trên $m$ điểm đó (tồn tại ít nhất một cách sắp xếp $m$ điểm để thực hiện điều này).

Kết quả kinh điển: Lớp siêu phẳng phân tách tuyến tính trên $\mathbb{R}^d$ (với bias) có $\text{VC}(\mathcal{H}) = d + 1$. Theo Định lý Vapnik–Chervonenkis, với xác suất $\ge 1 - \delta$, độ chênh tổng quát hóa được chặn bởi:`,
          math: String.raw`R(\hat{h}_{\text{ERM}}) \le R(h^*) + O\!\left(\sqrt{\frac{d_{\text{VC}}\ln(n/d_{\text{VC}}) + \ln(1/\delta)}{n}}\right)`,
          intuition:
            'Muốn giảm khoảng cách tổng quát hóa khi bài toán giữ nguyên kích thước mẫu $n$, ta cần kiểm soát chiều VC hiệu dụng của mô hình (ví dụ qua lựa chọn đặc trưng hoặc ràng buộc chuẩn tham số $\\|w\\|_2$).',
        },
        {
          heading: '3. Kỷ luật thực nghiệm & Chống rò rỉ dữ liệu (Data Leakage)',
          body: String.raw`Để kết quả đánh giá trên phòng lab phản ánh đúng lỗi tổng quát hóa $R(h)$, dữ liệu phải được chia tách nghiêm ngặt ngay từ bước đầu tiên trước mọi thao tác thăm dò (EDA), chọn đặc trưng hay biến đổi dữ liệu:
- **Train**: Dùng để fit tham số mô hình $w$ **và** ước lượng toàn bộ thống kê tiền xử lý ($\mu_{\text{train}}, \sigma_{\text{train}}$, ngưỡng điền khuyết, thành phần PCA).
- **Validation** (hoặc K-Fold CV trong phần Train): Dùng để chọn siêu tham số ($k$, bậc đa thức, $\lambda$) và kiến trúc mô hình.
- **Test**: Khóa kín hoàn toàn; chỉ mở **một lần duy nhất** trên mô hình chốt cuối cùng để báo cáo.`,
          math: String.raw`Z_{\text{val}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad Z_{\text{test}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}`,
          pitfall:
            'Lỗi phổ biến nhất trong các báo cáo thực nghiệm là fit `StandardScaler` hoặc chọn top-K đặc trưng tương quan cao trên **toàn bộ** tập dữ liệu rồi mới chia K-Fold Cross-Validation. Việc này làm thông tin từ tập đánh giá rò rỉ vào huấn luyện, thổi phồng độ chính xác ảo.',
        },
      ],
      simulationId: 'm1-u4',
      exercises: [
        {
          id: 'm1-u4-split',
          title: 'Chia tách Train / Validation / Test tái lập theo Seed',
          prompt:
            'Cài đặt `train_val_test_split(n, val_frac, test_frac, seed=2026)` xáo trộn các chỉ số `0..n-1` một cách tái lập bằng `np.random.default_rng(seed).permutation(n)`, rồi cắt chính xác thành ba mảng NumPy rời nhau `(train_idx, val_idx, test_idx)` với `n_val = int(round(n * val_frac))` và `n_test = int(round(n * test_frac))`.',
          hint: 'Tính `order = np.random.default_rng(seed).permutation(n)`, `n_test = int(round(n * test_frac))`, `n_val = int(round(n * val_frac))`, `n_train = n - n_val - n_test`, sau đó cắt `order[:n_train]`, `order[n_train:n_train+n_val]`, `order[n_train+n_val:]`.',
          setup: COMMON_SETUP,
          starter: String.raw`def train_val_test_split(n, val_frac=0.2, test_frac=0.2, seed=2026):
    raise NotImplementedError
`,
          solution: String.raw`def train_val_test_split(n, val_frac=0.2, test_frac=0.2, seed=2026):
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
`,
          test: String.raw`_report('m1-u4-split',
    _check('Ba tập hoàn toàn rời nhau và hợp lại đủ đúng n chỉ số', lambda: (
        (lambda tr, va, te: (
            len(tr) == 60 and len(va) == 20 and len(te) == 20
            and len(set(tr) | set(va) | set(te)) == 100
            and len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
        ))(*train_val_test_split(100, 0.2, 0.2, seed=2026)),
        'Tách 100 mẫu với tỷ lệ 60/20/20.'
    )),
    _check('Tái lập chính xác khi dùng cùng seed và thay đổi khi đổi seed', lambda: (
        np.array_equal(train_val_test_split(50, 0.2, 0.2, seed=7)[0], train_val_test_split(50, 0.2, 0.2, seed=7)[0])
        and not np.array_equal(train_val_test_split(50, 0.2, 0.2, seed=7)[0], train_val_test_split(50, 0.2, 0.2, seed=8)[0]),
        'Đảm bảo khả năng tái lập thực nghiệm (reproducibility).'
    )),
)
`,
        },
        {
          id: 'm1-u4-safe-scale',
          title: 'Chuẩn hóa Pipeline không rò rỉ dữ liệu',
          prompt:
            'Cài đặt `scale_without_leakage(X_train, X_val)` ước lượng trung bình $\\mu$ và độ lệch chuẩn $\\sigma$ (`ddof=0`) **chỉ trên `X_train`**, rồi trả về `(Z_train, Z_val)`. Nếu một cột có $\\sigma = 0$ trong `X_train`, thay bằng $1.0$ để tránh chia cho 0.',
          hint: 'Tính `mu = X_train.mean(axis=0)`, `sd = X_train.std(axis=0)`, `sd = np.where(sd == 0, 1.0, sd)` và biến đổi cả `(X_train - mu) / sd` lẫn `(X_val - mu) / sd`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xtr = np.array([[1., 10.], [2., 10.], [3., 10.], [4., 10.]])
_Xva = np.array([[2.5, 10.], [100., 10.]])
_Xva_alt = np.array([[2.5, 10.], [5.0, 10.]])
`,
          starter: String.raw`def scale_without_leakage(X_train, X_val):
    raise NotImplementedError
`,
          solution: String.raw`def scale_without_leakage(X_train, X_val):
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_val - mu) / sd
`,
          test: String.raw`_report('m1-u4-safe-scale',
    _check('Z_train hoàn toàn bất biến dù X_val chứa ngoại lệ cực lớn (100.0)', lambda: (
        np.allclose(scale_without_leakage(_Xtr, _Xva)[0], scale_without_leakage(_Xtr, _Xva_alt)[0]),
        'Không có thông tin từ X_val đi ngược vào thống kê của X_train.'
    )),
    _check('X_val được biến đổi bằng đúng (mu_train, sd_train) và xử lý cột hằng', lambda: (
        (lambda Ztr, Zva: (
            np.allclose(Ztr.mean(axis=0), 0.0)
            and np.allclose(Zva[0], [0.0, 0.0])
            and np.isclose(Zva[1, 0], (100.0 - 2.5) / np.std(_Xtr[:, 0]))
        ))(*scale_without_leakage(_Xtr, _Xva)),
        'Kiểm tra giá trị chuẩn hóa chính xác.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao bước tiền xử lý chọn K đặc trưng có tương quan cao nhất với nhãn mục tiêu $y$ (`SelectKBest`) bắt buộc phải đặt bên trong từng fold của Cross-Validation chứ không được làm trước khi chia fold?',
          guidance:
            'Nếu tính tương quan với nhãn $y$ trên toàn bộ tập dữ liệu trước khi chia fold, thuật toán đã sử dụng nhãn của tập validation để chọn đặc trưng tốt giả tạo trên chính tập đó. Với dữ liệu thuần nhiễu (không có tín hiệu thật) gồm 10.000 đặc trưng và 50 mẫu, rò rỉ này có thể tạo ra độ chính xác CV ảo trên 85% dù hiệu năng thật chỉ tương đương đoán ngẫu nhiên 50%.',
        },
        {
          question:
            'Giải thích vì sao lớp hàm $\\mathcal{H} = \\{ x \\mapsto \\text{sign}(\\sin(\\omega x)) : \\omega \\in \\mathbb{R} \\}$ chỉ có 1 tham số $\\omega$ nhưng lại có $\\text{VC}(\\mathcal{H}) = \\infty$. Điều này hàm ý gì về mối quan hệ giữa số lượng tham số và sức chứa tổng quát hóa?',
          guidance:
            'Với tần số $\\omega$ đủ lớn, hàm $\\sin(\\omega x)$ dao động nhanh tùy ý và có thể cắt qua tập hợp bất kỳ gồm $m$ điểm gán nhãn bất kỳ (shatter mọi $m$). Do đó số lượng tham số vô hướng không đồng nhất với chiều VC; điều quyết định khả năng tổng quát hóa là độ phức tạp hàm và mức độ nhẵn (smoothness / capacity) chứ không chỉ là số tham số.',
        },
      ],
    },
  ],
};
