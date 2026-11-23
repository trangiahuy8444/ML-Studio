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

export const MODULE_03: GradModule = {
  id: 3,
  title: 'Tối ưu hóa cho Học máy',
  englishTitle: 'Numerical Optimization for Machine Learning',
  color: '#ffd173',
  status: 'ready',
  overview:
    'Nền tảng tối ưu lồi và tối ưu số học cho các thuật toán học máy: tập lồi, hàm lồi, tính trơn $L$-smooth & lồi chặt $\\mu$-strongly convex, số điều kiện $\\kappa(H)$ và điều kiện tối ưu KKT; phân tích hội tụ Gradient Descent và Armijo Backtracking; động học của SGD, Momentum, Nesterov, RMSProp và Adam; cùng phương pháp bậc hai Newton–Raphson, IRLS và L-BFGS.',
  outcomes: [
    'Phân tích hình thái mặt mất mát qua phổ trị riêng Hessian: tính $L$-smoothness, $\\mu$-strong convexity và số điều kiện $\\kappa = L/\\mu$.',
    'Chứng minh định lý hội tụ tuyến tính của Gradient Descent với bước học tối ưu $\\eta = \\frac{2}{L + \\mu}$ và cài đặt tìm kiếm bước Armijo Backtracking.',
    'Giải thích động lực học và cài đặt chuẩn xác SGD Momentum và Adam (kèm hiệu chỉnh thiên lệch bias correction).',
    'Cài đặt thuật toán bậc hai Newton–Raphson / IRLS cho Logistic Regression đạt tốc độ hội tụ bậc hai chỉ sau vài vòng lặp.',
  ],
  sources: [
    'UC Berkeley CS189/289A — Lecture 12–13 (Gradient Descent & Stochastic Gradient Descent)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 8: Optimization)',
    'Vũ Hữu Tiệp — Machine Learning cơ bản (Chương 7–8: Gradient Descent & Bài toán Tối ưu Lồi)',
    'Stanford CS229 — Machine Learning Complete Notes (Part I: Newton Method)',
  ],
  units: [
    {
      id: 'm3-u1',
      moduleId: 3,
      order: 1,
      title: 'Tập lồi, Hàm lồi, Điều kiện KKT & Condition Number',
      summary:
        'Hình học và giải tích của bài toán tối ưu lồi: định nghĩa bậc 1 & bậc 2, hằng số Lipschitz gradient ($L$-smoothness), độ cong cực tiểu ($\\mu$-strong convexity), số điều kiện $\\kappa = L/\\mu$ gây hiện tượng zic-zắc và hệ điều kiện KKT cho bài toán có ràng buộc.',
      outcomes: [
        'Nhận diện tính lồi qua bất đẳng thức Jensen, siêu phẳng tiếp tuyến (bậc 1) và Hessian nửa xác định dương (bậc 2).',
        'Tính chính xác $L = \\lambda_{\\max}(H)$, $\\mu = \\lambda_{\\min}(H)$ và số điều kiện $\\kappa = L/\\mu$ cho hàm toàn phương và Ridge.',
        'Phát biểu bốn nhóm điều kiện Karush–Kuhn–Tucker (KKT) và mối liên hệ với đối ngẫu Lagrange.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§8.1–8.2 (Convexity, Smoothness & KKT Optimality Conditions)',
          note: 'Tổng hợp chuẩn xác về hàm lồi, tính trơn L-smooth, lồi chặt μ-strongly convex và đối ngẫu Lagrange.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/book_ML.pdf',
          section: 'Chương 8: Giới thiệu về Tối ưu Lồi (Convex Optimization)',
          note: 'Giải thích trực quan tập lồi, hàm lồi, đối ngẫu và điều kiện KKT.',
        },
      ],
      sections: [
        {
          heading: '1. Đặc trưng bậc 1 & bậc 2 của Hàm lồi, L-Smoothness & Strong Convexity',
          body: String.raw`Một tập $C \subseteq \mathbb{R}^d$ là **tập lồi** nếu đoạn thẳng nối hai điểm bất kỳ thuộc $C$ đều nằm trọn trong $C$. Hàm khả vi $f: \mathbb{R}^d \to \mathbb{R}$ là **hàm lồi** khi và chỉ khi xấp xỉ tuyến tính bậc 1 luôn nằm dưới đồ thị của $f$, hay tương đương ma trận Hessian $\nabla^2 f(w) \succeq 0$ tại mọi điểm.

Để định lượng tốc độ hội tụ của các thuật toán lặp, ta chặn hai phía độ cong của $\nabla^2 f(w)$ bằng hai hằng số $0 \le \mu \le L$:`,
          math: String.raw`\mu I \preceq \nabla^2 f(w) \preceq L I \quad \Longleftrightarrow \quad \frac{\mu}{2}\|v-w\|_2^2 \le f(v) - f(w) - \nabla f(w)^\top(v-w) \le \frac{L}{2}\|v-w\|_2^2`,
          intuition:
            'Hằng số $L$ ($L$-smoothness) kẹp độ cong tối đa không cho gradient biến thiên quá gắt — giúp giới hạn trên cỡ bước học $\\eta$ an toàn. Hằng số $\\mu > 0$ (lồi chặt — $\\mu$-strongly convex) bảo đảm đáy thung lũng không bị bẹt, tồn tại duy nhất điểm cực tiểu toàn cục $w^*$.',
        },
        {
          heading: '2. Số điều kiện Hessian κ = L / μ & Hiệu ứng thung lũng hẹp',
          body: String.raw`Cho hàm mục tiêu MSE có điều chuẩn Ridge: $L(w) = \frac{1}{n}\|Xw - y\|_2^2 + \lambda \|w\|_2^2$. Ma trận Hessian hằng là $H = \frac{2}{n}X^\top X + 2\lambda I_d$, có các trị riêng $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$. Khi đó:

$$L = \lambda_{\max}(H), \qquad \mu = \lambda_{\min}(H), \qquad \kappa(H) = \frac{L}{\mu}$$`,
          math: String.raw`\kappa\big(H_{\text{Ridge}}\big) = \frac{\lambda_{\max}(\frac{2}{n}X^\top X) + 2\lambda}{\lambda_{\min}(\frac{2}{n}X^\top X) + 2\lambda}`,
          intuition:
            'Số điều kiện $\\kappa$ đo tỉ lệ độ dốc giữa trục dốc nhất và trục phẳng nhất trên đường đồng mức elip. Khi $\\kappa \\gg 1$ (do thang đo các cột chênh lệch lớn hoặc đa cộng tuyến), đường đồng mức kéo dài thành hẻm núi hẹp: vector $-\\nabla f$ gần như vuông góc với thung lũng thay vì hướng về đáy $w^*$, làm Gradient Descent dao động zic-zắc chậm chạp. Chuẩn hóa đặc trưng (Standardization) và thêm Ridge $\\lambda > 0$ giúp hạ mạnh $\\kappa$.',
        },
        {
          heading: '3. Tối ưu có ràng buộc & Hệ điều kiện KKT',
          body: String.raw`Xét bài toán tối ưu gốc với ràng buộc bất đẳng thức $g_i(w) \le 0$ và đẳng thức $h_j(w) = 0$. Hàm Lagrange $\mathcal{L}(w, \alpha, \nu) = f(w) + \sum_i \alpha_i g_i(w) + \sum_j \nu_j h_j(w)$ dẫn đến 4 điều kiện cần **Karush–Kuhn–Tucker (KKT)** (đồng thời là điều kiện đủ cho bài toán lồi thỏa mãn điều kiện Slater):
- **Stationarity**: $\nabla_w \mathcal{L}(w^*, \alpha^*, \nu^*) = 0$.
- **Primal feasibility**: $g_i(w^*) \le 0,\; h_j(w^*) = 0$.
- **Dual feasibility**: $\alpha_i^* \ge 0$.
- **Complementary slackness (Độ bù)**: $\alpha_i^* g_i(w^*) = 0$ (nếu ràng buộc $g_i(w^*) < 0$ không căng thì $\alpha_i^* = 0$ — đây chính là cơ chế sinh ra Support Vectors trong SVM ở Học phần 4).`,
          pitfall:
            'Dấu nhân tử Lagrange cho ràng buộc bất đẳng thức luôn đòi hỏi $\\alpha_i \\ge 0$ cùng chiều tối thiểu hóa $f(w) + \\sum \\alpha_i g_i(w)$; đảo dấu bất đẳng thức ràng buộc phải đảo dấu $\\alpha_i$ tương ứng.',
        },
      ],
      simulationId: 'm3-u1',
      exercises: [
        {
          id: 'm3-u1-cond-number',
          title: 'Phân tích phổ Hessian & Số điều kiện κ(H)',
          prompt:
            'Cho hàm mất mát MSE có Ridge $L(w) = \\frac{1}{n}\\|Xw - y\\|_2^2 + \\lambda\\|w\\|_2^2$ với ma trận Hessian $H = \\frac{2}{n} X^\\top X + 2\\lambda I_d$. Cài đặt `hessian_condition_number(X, lam=0.0)` sử dụng `np.linalg.eigvalsh` để trả về tuple `(mu_strong, L_smooth, kappa)`. Nếu `mu_strong <= 1e-15`, trả về `kappa = float("inf")`.',
          hint: 'Tính `H = (2.0 / X.shape[0]) * (X.T @ X) + 2.0 * float(lam) * np.eye(X.shape[1])`, lấy `eigs = np.linalg.eigvalsh(H)` (mảng tăng dần), suy ra `mu = float(eigs[0])`, `L = float(eigs[-1])`, và `kappa = float(L / mu) if mu > 1e-15 else float("inf")`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X_ill = np.array([[100., 0.1], [200., -0.2], [-150., 0.3], [50., 0.4]])
_X_sing = np.array([[1., 2.], [2., 4.], [-3., -6.]])
`,
          starter: String.raw`def hessian_condition_number(X, lam=0.0):
    raise NotImplementedError
`,
          solution: String.raw`def hessian_condition_number(X, lam=0.0):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    H = (2.0 / n) * (X.T @ X) + 2.0 * float(lam) * np.eye(d)
    eigs = np.linalg.eigvalsh(H)
    mu = float(eigs[0])
    L = float(eigs[-1])
    kappa = float(L / mu) if mu > 1e-15 else float('inf')
    return mu, L, kappa
`,
          test: String.raw`_report('m3-u1-cond-number',
    _check('Tính chính xác μ, L và chứng minh Ridge λ > 0 làm giảm mạnh κ', lambda: (
        (lambda res0, res1: res0[2] > 1e5 and res1[2] < res0[2] * 0.01 and np.isclose(res1[0], res0[0] + 20.0))(
            hessian_condition_number(_X_ill, 0.0),
            hessian_condition_number(_X_ill, 10.0),
        ),
        'Thêm 2λI dịch toàn bộ phổ trị riêng lên thêm 2λ.'
    )),
    _check('Ma trận suy biến hạng (rank < d) cho μ ≈ 0 và κ = inf khi λ = 0', lambda: (
        (lambda mu, L, kap: mu < 1e-10 and kap == float('inf'))(*hessian_condition_number(_X_sing, 0.0)),
        'Phân biệt hàm lồi thường (μ=0) với lồi chặt (μ>0).'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Chứng minh định lý: nếu $f$ vừa $L$-smooth vừa $\\mu$-strongly convex, với bất kỳ ma trận thiết kế chuẩn hóa khác nhau nào, điều kiện $\\kappa \\ge 1$ luôn đúng và $\\kappa = 1$ xảy ra khi nào?',
          guidance:
            'Vì $\\mu \\le \\lambda_{\\min}(\\nabla^2 f) \\le \\lambda_{\\max}(\\nabla^2 f) \\le L$, luôn có $\\kappa = L/\\mu \\ge 1$. Dấu bằng $\\kappa = 1$ xảy ra khi toàn bộ trị riêng của Hessian bằng nhau ($H = L I_d$), tức đường đồng mức là hình cầu tròn hoàn hảo; khi đó Gradient Descent với $\\eta = 1/L$ hội tụ chỉ sau đúng **1 bước**!',
        },
      ],
    },
    {
      id: 'm3-u2',
      moduleId: 3,
      order: 2,
      title: 'Gradient Descent & Phân tích Tốc độ Hội tụ',
      summary:
        'Chứng minh bổ đề giảm giá trị hàm mục tiêu (Descent Lemma) cho hàm $L$-smooth, chặn hội tụ tuyến tính $\\left(\\frac{\\kappa - 1}{\\kappa + 1}\\right)^t$ theo số điều kiện $\\kappa$, ngưỡng ổn định bước học $\\eta < 2/L$ và thuật toán tìm bước tự động Armijo Backtracking Line Search.',
      outcomes: [
        'Sử dụng Descent Lemma chứng minh bước học $\\eta \\le 1/L$ bảo đảm $f(w_{t+1}) \\le f(w_t) - \\frac{\\eta}{2}\\|\\nabla f(w_t)\\|_2^2$.',
        'Tính chính xác ngưỡng phân kỳ $\\eta > 2/L$ và tốc độ hội tụ $O(\\kappa \\log(1/\\epsilon))$ cho hàm $\\mu$-lồi chặt.',
        'Cài đặt Armijo Backtracking Line Search bảo đảm giảm đơn điệu hàm mất mát mà không cần biết trước hằng số $L$.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 7/Lecture 12 -- Gradient Descent.pdf',
          section: 'Gradient Descent Convergence & Step Sizes',
          note: 'Phân tích bước học, định lý hội tụ và hội tụ tuyến tính theo số điều kiện κ.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§8.3.1–8.3.2 (Gradient Descent & Backtracking Line Search)',
          note: 'Điều kiện Armijo–Goldstein và phân tích tốc độ hội tụ.',
        },
      ],
      sections: [
        {
          heading: '1. Descent Lemma & Ngưỡng ổn định η < 2/L',
          body: String.raw`Với hàm $L$-smooth, khai triển Taylor bậc 2 bị chặn trên bởi parabol đồng nhất có độ cong $L$: $f(w - \eta g) \le f(w) - \eta \|g\|_2^2 + \frac{L \eta^2}{2}\|g\|_2^2$ (với $g = \nabla f(w)$). Khi $0 < \eta < \frac{2}{L}$, thừa số $\eta\left(1 - \frac{L\eta}{2}\right) > 0$, nên mọi bước đi có gradient khác $0$ đều **chắc chắn làm giảm $f$**. Điểm cực tiểu hóa vế phải là bước học chuẩn $\eta = \frac{1}{L}$, cho lượng giảm bảo chứng ít nhất $\frac{1}{2L}\|g\|_2^2$.`,
          math: String.raw`f(w_{t+1}) \le f(w_t) - \frac{1}{2L}\|\nabla f(w_t)\|_2^2 \quad \left(\text{tại }\eta = \frac{1}{L}\right)`,
          pitfall:
            'Nếu đặt bước học $\\eta > \\frac{2}{L}$, thành phần bậc hai $\\frac{L\\eta^2}{2}$ thắng thành phần bậc một, khiến bước nhảy vượt qua đáy parabol và dao động phân kỳ ra vô cực ($f(w_t) \\to +\\infty$).',
        },
        {
          heading: '2. Tốc độ hội tụ tuyến tính theo κ = L / μ',
          body: String.raw`Khi $f$ vừa $L$-smooth vừa $\mu$-strongly convex, sai số khoảng cách tới nghiệm $w^*$ sau $t$ bước với bước học chuẩn $\eta = 1/L$ giảm theo cấp số nhân (gọi là **hội tụ tuyến tính — linear convergence** vì số chữ số chính xác tăng tuyến tính theo số bước $t$). Chọn $\eta^* = \frac{2}{L + \mu}$ đạt hệ số co tối ưu:`,
          math: String.raw`\|w_t - w^*\|_2 \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|w_0 - w^*\|_2, \qquad f(w_t) - f(w^*) \le \left(1 - \frac{\mu}{L}\right)^t \big(f(w_0) - f(w^*)\big)`,
          intuition:
            'Để đạt sai số $f(w_t) - f(w^*) \\le \\varepsilon$, Gradient Descent cần $O\\big(\\kappa \\log(1/\\varepsilon)\\big)$ vòng lặp. Khi $\\kappa = 1000$, mỗi 1000 bước mới giảm được sai số đi $e^{-1}$ — thúc đẩy nhu cầu dùng Momentum, Adam hoặc Newton.',
        },
        {
          heading: '3. Tìm bước học thích nghi: Armijo Backtracking Line Search',
          body: String.raw`Trong thực tế ta hiếm khi biết trước hằng số $L$ toàn cục (nhất là với các hàm phi tuyến như Logistic Regression hay mạng sâu). **Armijo Backtracking Line Search** xuất phát từ cỡ bước lớn $\eta = 1.0$, rồi lặp nhân thu hẹp $\eta \leftarrow \beta \eta$ (với $\beta \in (0, 1)$, thường chọn $\beta = 0.5$) cho tới khi thỏa mãn **điều kiện giảm đủ Armijo**:`,
          math: String.raw`f\big(w_t - \eta \nabla f(w_t)\big) \le f(w_t) - c\, \eta \|\nabla f(w_t)\|_2^2 \qquad (c \in (0, 0.5),\;\text{thường } c = 10^{-4})`,
          intuition:
            'Armijo bảo đảm mỗi bước vừa giảm giá trị hàm ít nhất một tỷ lệ cố định $c$ so với đường tiếp tuyến, vừa tự động dùng bước lớn ở những vùng phẳng và thu nhỏ bước ngay lập tức khi gặp độ cong $L$ cục bộ lớn.',
        },
      ],
      simulationId: 'm3-u2',
      exercises: [
        {
          id: 'm3-u2-gd-armijo',
          title: 'Gradient Descent kết hợp Armijo Backtracking Line Search',
          prompt:
            'Cài đặt `gd_backtracking(f, grad_f, w0, c=1e-4, beta=0.5, steps=30)` tối ưu hàm `f(w)` có gradient `grad_f(w)` xuất phát từ `w0`. Tại mỗi bước (tổng `steps` bước), khởi đầu `eta = 1.0` và co `eta *= beta` cho tới khi thỏa điều kiện Armijo `f(w - eta * g) <= f(w) - c * eta * (g @ g)`. Trả về `(w_final, history)` trong đó `history` là danh sách `[f(w0), f(w1), ..., f(w_steps)]` gồm đúng `steps + 1` giá trị.',
          hint: 'Chuyển `w = np.asarray(w0, dtype=float).copy()`, khởi tạo `history = [float(f(w))]`. Trong vòng lặp `for _ in range(steps)`, tính `fw = float(f(w))`, `g = grad_f(w)`, `gnorm2 = float(g @ g)`, đặt `eta = 1.0`. Lặp `while eta > 1e-16 and float(f(w - eta * g)) > fw - c * eta * gnorm2: eta *= beta`, rồi cập nhật `w = w - eta * g` và `history.append(float(f(w)))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_f_quad = lambda w: 50.0 * (w[0] ** 2) + 0.5 * (w[1] ** 2)
_g_quad = lambda w: np.array([100.0 * w[0], 1.0 * w[1]])
`,
          starter: String.raw`def gd_backtracking(f, grad_f, w0, c=1e-4, beta=0.5, steps=30):
    raise NotImplementedError
`,
          solution: String.raw`def gd_backtracking(f, grad_f, w0, c=1e-4, beta=0.5, steps=30):
    w = np.asarray(w0, dtype=float).copy()
    history = [float(f(w))]
    for _ in range(steps):
        fw = float(f(w))
        g = np.asarray(grad_f(w), dtype=float)
        gnorm2 = float(g @ g)
        if gnorm2 < 1e-24:
            history.append(fw)
            continue
        eta = 1.0
        while eta > 1e-16 and float(f(w - eta * g)) > fw - c * eta * gnorm2:
            eta *= beta
        w = w - eta * g
        history.append(float(f(w)))
    return w, history
`,
          test: String.raw`_report('m3-u2-gd-armijo',
    _check('Tự động ổn định trên hàm L=100 (η=1.0 ban đầu vốn gây phân kỳ) và giảm đơn điệu hoàn toàn', lambda: (
        (lambda w_fin, hist: (
            len(hist) == 41 and all(hist[i + 1] <= hist[i] + 1e-12 for i in range(40)) and hist[-1] < 0.5
        ))(*gd_backtracking(_f_quad, _g_quad, np.array([1.0, 2.0]), steps=40)),
        'f(w) giảm từ 52.0 xuống <0.5 trong 40 bước và không tăng tại bất kỳ bước nào.'
    )),
    _check('Hội tụ tới đáy [0, 0] khi chạy đủ số vòng lặp (steps=300)', lambda: (
        np.allclose(gd_backtracking(_f_quad, _g_quad, np.array([1.0, 2.0]), steps=300)[0], [0.0, 0.0], atol=5e-3),
        'Kiểm tra sự hội tụ trên thung lũng điều kiện xấu κ=100.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'So sánh tốc độ hội tụ của hàm lồi trơn không lồi chặt ($\\mu = 0$, tốc độ $O(1/t)$) và hàm vừa trơn vừa lồi chặt ($\\mu > 0$, tốc độ $O(e^{-t/\\kappa})$). Một hàm phi lồi như mạng sâu có còn thỏa Descent Lemma không?',
          guidance:
            'Với $\\mu = 0$ (ví dụ vùng phẳng tiệm cận), khoảng cách nghiệm giảm chậm theo $O(1/t)$, cần tới $O(1/\\varepsilon)$ bước để đạt độ chính xác $\\varepsilon$, thay vì $O(\\log(1/\\varepsilon))$ như lồi chặt. Đối với hàm phi lồi nhưng khả vi Lipschitz ($L$-smooth), Descent Lemma vẫn đúng cục bộ, bảo đảm $\\min_{t < T} \\|\\nabla f(w_t)\\|_2^2 = O(1/T)$ (hội tụ tới điểm dừng / saddle / local min), nhưng không bảo đảm cực tiểu toàn cục.',
        },
      ],
    },
    {
      id: 'm3-u3',
      moduleId: 3,
      order: 3,
      title: 'SGD, Mini-batch, Momentum, Nesterov, RMSProp & Adam',
      summary:
        'Tối ưu hóa quy mô lớn và động lực học bậc nhất nâng cao: ước lượng gradient ngẫu nhiên không chệch (SGD) & sàn phương sai, cơ chế quán tính Heavy-Ball Momentum và Nesterov look-ahead, cùng chuẩn hóa bước học thích nghi từng tọa độ RMSProp và Adam kèm bias correction.',
      outcomes: [
        'Hiểu tính không chệch $\\mathbb{E}[\\tilde{g}] = \\nabla f(w)$ của mini-batch SGD và lý do cần lịch trình giảm learning rate (Robbins–Monro) khi không dùng variance reduction.',
        'Viết chính xác quy tắc cập nhật Heavy-Ball Momentum và Nesterov Accelerated Gradient (đạt chuẩn $O(1/t^2)$ cho hàm lồi trơn, tối ưu của Nesterov).',
        'Cài đặt Adam có hiệu chỉnh thiên lệch $(\\hat{m}_t, \\hat{v}_t)$ và lý giải vai trò chuẩn hóa từng tham số khi gradient giữa các chiều chênh lệch nhiều bậc độ lớn.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/Tuần 7/Lecture 13 -- Stochastic Gradient Descent.pdf',
          section: 'SGD, Mini-batch Variance & Momentum',
          note: 'Đánh đổi giữa chi phí mỗi bước O(B) với phương sai nhiễu gradient.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§8.4 (Stochastic Gradient Descent, Momentum, RMSProp & Adam)',
          note: 'Tổng hợp chuẩn xác công thức cập nhật, bias correction của Adam và lưu ý về AdamW.',
        },
      ],
      sections: [
        {
          heading: '1. Stochastic Gradient Descent (SGD) & Sàn phương sai',
          body: String.raw`Khi tập huấn luyện có $n = 10^6$ mẫu, mỗi bước Full-Batch Gradient Descent phải quét toàn bộ dữ liệu mất chi phí $O(nd)$. Mini-batch SGD chọn ngẫu nhiên tập con $B \subset \{1, \dots, n\}$ kích thước $|B| \ll n$: $\tilde{g}_t = \frac{1}{|B|}\sum_{i \in B} \nabla \ell_i(w_t)$. Vì $\mathbb{E}[\tilde{g}_t] = \nabla f(w_t)$, bước đi đúng kỳ vọng nhưng có phương sai $\text{Var}(\tilde{g}_t) \propto \frac{1}{|B|}$.

Nếu dùng learning rate cố định $\eta$, SGD hội tụ nhanh về gần đáy nhưng sau đó dao động mãi trong một quả cầu bán kính tỉ lệ với $\sqrt{\eta \sigma^2}$ (**variance floor**). Để hội tụ chính xác về $w^*$, điều kiện Robbins–Monro đòi hỏi $\sum \eta_t = \infty$ và $\sum \eta_t^2 < \infty$ (ví dụ $\eta_t = \eta_0 / (1 + c t)$ hoặc Cosine Annealing).`,
        },
        {
          heading: '2. Heavy-Ball Momentum & Nesterov Accelerated Gradient (NAG)',
          body: String.raw`Trong thung lũng có số điều kiện $\kappa \gg 1$, gradient theo chiều ngang triệt tiêu lẫn nhau (đổi dấu liên tục) trong khi gradient theo chiều dọc đáy giữ nguyên dấu. **Momentum** tích lũy trung bình động vận tốc $v_t = \beta v_{t-1} + g_t$ để cộng dồn thành phần đi dọc thung lũng và triệt tiêu dao động ngang:

- **Polyak Heavy-Ball**: $v_t = \beta v_{t-1} + \nabla f(w_{t-1}), \quad w_t = w_{t-1} - \eta v_t$.
- **Nesterov (NAG)**: Lấy gradient tại vị trí dự đoán trước ("look-ahead"): $g_t = \nabla f(w_{t-1} - \eta \beta v_{t-1})$, giảm tốc kịp thời trước khi leo ngược lên dốc bên kia, cải thiện tốc độ phụ thuộc $\kappa$ xuống còn $\sqrt{\kappa}$ ($O(1/t^2)$ cho hàm lồi trơn).`,
          math: String.raw`\|w_t - w^*\|_2 = O\!\left(\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t\right) \quad \text{(Cải thiện căn bậc hai từ } \kappa \text{ xuống } \sqrt{\kappa}\text{)}`,
        },
        {
          heading: '3. RMSProp & Adam (Adaptive Moment Estimation)',
          body: String.raw`Các đặc trưng thưa hoặc các tầng mạng nơ-ron khác nhau thường có biên độ gradient lệch nhau hàng nghìn lần. **Adam** kết hợp quán tính bậc 1 ($m_t$) của Momentum với ước lượng momen bậc 2 ($v_t$) của RMSProp để tự động chia bước học mỗi chiều cho căn bậc hai độ lớn gradient tích lũy, đồng thời chia $(1 - \beta^t)$ để **khử thiên lệch khởi tạo $0$** ở các bước đầu:`,
          math: String.raw`\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, & \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t, & \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
w_t &= w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}`,
          pitfall:
            'Trong các mô hình có phạt chuẩn tham số, **Adam + L2 penalty ≠ Weight Decay (AdamW)** vì thành phần $L_2$ đi vào $g_t$ bị chia cho $\\sqrt{\\hat{v}_t}$, làm các tham số có gradient lớn bị phạt nhẹ đi. Chuẩn hiện đại luôn dùng **Decoupled Weight Decay (AdamW)**.',
        },
      ],
      simulationId: 'm3-u3',
      exercises: [
        {
          id: 'm3-u3-adam',
          title: 'Cài đặt Adam có hiệu chỉnh thiên lệch (Bias Correction)',
          prompt:
            'Cài đặt `adam_minimize(grad_fn, w0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=200)` chạy đúng quy tắc cập nhật Adam (với $t = 1, 2, \\dots, \\text{steps}$) từ điểm đầu `w0` và trả về vector nghiệm cuối cùng. Bắt buộc phải thực hiện chia hiệu chỉnh thiên lệch $\\hat{m}_t = m_t / (1 - \\beta_1^t)$ và $\\hat{v}_t = v_t / (1 - \\beta_2^t)$.',
          hint: 'Khởi tạo `w = np.asarray(w0, dtype=float).copy()`, `m = np.zeros_like(w)`, `v = np.zeros_like(w)`. Trong vòng `for t in range(1, steps + 1)`: cập nhật `g`, `m`, `v`, tính `m_hat = m / (1.0 - beta1 ** t)`, `v_hat = v / (1.0 - beta2 ** t)`, rồi `w = w - lr * m_hat / (np.sqrt(v_hat) + eps)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_f_ill_a = lambda w: 50.0 * ((w[0] - 3.0) ** 2) + 0.5 * ((w[1] + 2.0) ** 2)
_g_ill_a = lambda w: np.array([100.0 * (w[0] - 3.0), 1.0 * (w[1] + 2.0)])
`,
          starter: String.raw`def adam_minimize(grad_fn, w0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=200):
    raise NotImplementedError
`,
          solution: String.raw`def adam_minimize(grad_fn, w0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=200):
    w = np.asarray(w0, dtype=float).copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(1, steps + 1):
        g = np.asarray(grad_fn(w), dtype=float)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - (beta1 ** t))
        v_hat = v / (1.0 - (beta2 ** t))
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w
`,
          test: String.raw`_report('m3-u3-adam',
    _check('Hội tụ chính xác về [3.0, -2.0] trên hàm κ = 100 trong 250 bước', lambda: (
        np.allclose(adam_minimize(_g_ill_a, np.array([-4.0, 5.0]), lr=0.1, steps=250), [3.0, -2.0], atol=1e-2),
        'Adam cân bằng hai trục có độ dốc chênh nhau 100 lần.'
    )),
    _check('Bước t=1 phản ánh đúng hiệu chỉnh thiên lệch: |Δw_j| ≈ lr khi eps nhỏ', lambda: (
        np.allclose(adam_minimize(_g_ill_a, np.array([0.0, 0.0]), lr=0.05, steps=1), [0.05, -0.05], atol=1e-6),
        'Tại t=1, m_hat = g và sqrt(v_hat) = |g|, nên bước cập nhật hướng về [3, -2] có biên độ gần như ±lr.'
    )),
)
`,
        },
        {
          id: 'm3-u3-sgd-momentum',
          title: 'Mini-batch SGD với Heavy-Ball Momentum cho Hồi quy tuyến tính',
          prompt:
            'Cài đặt `sgd_momentum_linear(X, y, lr=0.05, beta=0.9, batch_size=16, epochs=30, seed=0)` tối ưu MSE $\\frac{1}{n}\\|Xw + b - y\\|_2^2$. Mỗi epoch xáo trộn chỉ số bằng `rng.permutation(n)` (`rng = np.random.default_rng(seed)`), duyệt qua từng mini-batch kích thước `batch_size`, tính gradient trung bình của mini-batch đó và cập nhật vận tốc `v = beta * v + g`, `theta = theta - lr * v`. Trả về `(w, b)`.',
          hint: 'Khởi tạo `w = np.zeros(d)`, `b = 0.0`, `vw = np.zeros(d)`, `vb = 0.0`. Với mỗi batch `idx = order[s:s+batch_size]`, `err = X[idx] @ w + b - y[idx]`, `dw = (2.0 / len(idx)) * (X[idx].T @ err)`, `db = float(2.0 * np.mean(err))`. Cập nhật `vw = beta * vw + dw`, `vb = beta * vb + db`, rồi `w -= lr * vw`, `b -= lr * vb`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_sgd = np.random.default_rng(123)
_Xs = _rng_sgd.normal(size=(96, 3))
_ys = 2.5 * _Xs[:, 0] - 1.5 * _Xs[:, 1] + 0.8 * _Xs[:, 2] + 3.0 + _rng_sgd.normal(scale=0.05, size=96)
`,
          starter: String.raw`def sgd_momentum_linear(X, y, lr=0.05, beta=0.9, batch_size=16, epochs=30, seed=0):
    raise NotImplementedError
`,
          solution: String.raw`def sgd_momentum_linear(X, y, lr=0.05, beta=0.9, batch_size=16, epochs=30, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d, dtype=float)
    b = 0.0
    vw = np.zeros(d, dtype=float)
    vb = 0.0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            err = X[idx] @ w + b - y[idx]
            dw = (2.0 / len(idx)) * (X[idx].T @ err)
            db = float(2.0 * np.mean(err))
            vw = beta * vw + dw
            vb = beta * vb + db
            w = w - lr * vw
            b = b - lr * vb
    return w, float(b)
`,
          test: String.raw`_report('m3-u3-sgd-momentum',
    _check('Khôi phục chính xác hệ số [2.5, -1.5, 0.8] và bias ≈ 3.0 trên 96 mẫu', lambda: (
        (lambda w, b: np.allclose(w, [2.5, -1.5, 0.8], atol=0.1) and abs(b - 3.0) < 0.1)(
            *sgd_momentum_linear(_Xs, _ys, lr=0.03, beta=0.9, batch_size=16, epochs=30, seed=0)
        ),
        'Mini-batch SGD + Momentum hội tụ nhanh tới sát nghiệm thực.'
    )),
    _check('Tái lập hoàn toàn theo seed', lambda: (
        (lambda r1, r2: np.allclose(r1[0], r2[0]) and np.isclose(r1[1], r2[1]))(
            sgd_momentum_linear(_Xs, _ys, seed=5),
            sgd_momentum_linear(_Xs, _ys, seed=5),
        ),
        'Cùng seed phải cho ra trọng số giống hệt nhau.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong thực tế huấn luyện Deep Neural Networks, khi nào Mini-batch SGD + Momentum thường khái quát hóa (generalization) trên tập Test tốt hơn Adam mặc dù Adam có lỗi trên tập Train giảm nhanh hơn ở những epoch đầu?',
          guidance:
            'Nhiễu đẳng hướng của Mini-batch SGD kết hợp cấu trúc cập nhật không chuẩn hóa tọa độ có xu hướng đưa nghiệm về vùng cực tiểu phẳng rộng (flat minima) và tìm nghiệm lề lớn (max-margin implicit bias). Ngược lại, Adam thích nghi hóa tỷ lệ từng trục có thể hội tụ nhanh vào các cực tiểu nhọn hẹp (sharp minima) hoặc phụ thuộc nặng vào lựa chọn weight decay tách rời (AdamW).',
        },
      ],
    },
    {
      id: 'm3-u4',
      moduleId: 3,
      order: 4,
      title: 'Phương pháp Bậc 2: Newton–Raphson, IRLS & Quasi-Newton (L-BFGS)',
      summary:
        'Khai thác đạo hàm bậc hai (Hessian) để triệt tiêu ảnh hưởng của số điều kiện κ: xấp xỉ toàn phương cục bộ Newton–Raphson đạt tốc độ hội tụ bậc hai, mối liên hệ tương đương với Bình phương tối thiểu có trọng số lặp (IRLS) trên GLMs, và xấp xỉ Hessian chi phí O(md) của L-BFGS.',
      outcomes: [
        'Dẫn xuất bước cập nhật Newton $w_{t+1} = w_t - H_t^{-1} g_t$ từ khai triển Taylor bậc 2 và hiểu lý do bất biến với phép biến đổi tuyến tính đặc trưng.',
        'Cài đặt thuật toán Newton–Raphson / IRLS cho Logistic Regression có phạt $L_2$ đạt hội tụ máy ($\\|g\\| < 10^{-8}$) chỉ trong $\\le 10$ bước.',
        'So sánh đánh đổi chi phí $O(d^3)$ của Newton chính xác với chi phí $O(md)$ của L-BFGS trong bài toán nhiều chiều.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part I — Section 2.3 (Newton’s Method for Logistic Regression)',
          note: 'Dẫn xuất công thức Hessian của Logistic Regression và hội tụ bậc hai.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§8.3.3–8.3.4 (Newton’s Method, IRLS & Quasi-Newton / L-BFGS)',
          note: 'Giải thích IRLS cho GLMs và thuật toán L-BFGS lưu trữ m cặp hiệu gradient.',
        },
      ],
      sections: [
        {
          heading: '1. Bước cập nhật Newton–Raphson & Tính bất biến tỷ lệ',
          body: String.raw`Thay vì dùng xấp xỉ parabol đồng hướng $\frac{L}{2}\|v\|_2^2$ như Gradient Descent, phương pháp Newton tối thiểu hóa trực tiếp xấp xỉ Taylor bậc hai đầy đủ quanh điểm hiện tại $w_t$:

$$q(v) = f(w_t) + \nabla f(w_t)^\top v + \frac{1}{2} v^\top \nabla^2 f(w_t)\, v$$

Đặt đạo hàm theo bước nhảy $v$ bằng $0$ cho bước đi không cần chọn tham số learning rate (trừ khi ở xa đáy cần damped Newton):`,
          math: String.raw`w_{t+1} = w_t - \big[\nabla^2 f(w_t)\big]^{-1} \nabla f(w_t)`,
          intuition:
            'Ưu thế kỳ diệu nhất của Newton là **bất biến với phép biến đổi tuyến tính khả nghịch của đặc trưng** (Affine Invariance). Dù các cột đặc trưng có đơn vị lệch nhau $10^6$ lần ($\\kappa \\gg 1$), ma trận $H_t^{-1}$ tự động xoay và co dãn đưa thung lũng elip về dạng tròn hoàn hảo. Tốc độ hội tụ ở lân cận nghiệm là **bậc hai (quadratic)**: sai số sau mỗi bước bình phương lên ($\\|\\Delta_{t+1}\\| \\le C \\|\\Delta_t\\|^2$), tăng gấp đôi số chữ số chính xác sau mỗi lần lặp!',
        },
        {
          heading: '2. IRLS cho Logistic Regression & Bài toán quy mô lớn với L-BFGS',
          body: String.raw`Áp dụng Newton cho Logistic Regression có phạt $L_2$: $\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^n [\text{logaddexp}(0, X_{\text{aug}}\theta) - y_i (X_{\text{aug}}\theta)] + \frac{\lambda}{2}\|\theta\|_2^2$. Khi đó gradient và Hessian chính xác có dạng:

$$g = \frac{1}{n} X_{\text{aug}}^\top (p - y) + \lambda \theta, \qquad H = \frac{1}{n} X_{\text{aug}}^\top S X_{\text{aug}} + \lambda I, \quad S = \text{diag}\big(p_i(1 - p_i)\big)$$

Biến đổi $\theta_{t+1} = \theta_t - H^{-1} g$ thành một bài toán Weighted Least Squares với trọng số $S$ biến thiên theo từng bước gọi là **IRLS (Iteratively Reweighted Least Squares)**.`,
          math: String.raw`H \Delta\theta = g \quad \text{(Giải hệ tuyến tính bằng Cholesky / \texttt{np.linalg.solve} thay vì tính trực tiếp } H^{-1}\text{)}`,
          pitfall:
            'Chi phí tạo và giải hệ Hessian kích thước $d \\times d$ ở mỗi bước Newton là $O(n d^2 + d^3)$. Khi số chiều $d > 10\\,000$, Newton thuần túy trở nên bất khả thi; khi đó **L-BFGS** (Limited-memory BFGS) xấp xỉ $H^{-1} g$ chỉ bằng tích vô hướng với $m \\approx 10$ vector lịch sử gần nhất với chi phí $O(md)$ tuyến tính theo $d$.',
        },
      ],
      simulationId: 'm3-u4',
      exercises: [
        {
          id: 'm3-u4-newton-logistic',
          title: 'Newton–Raphson / IRLS cho Logistic Regression có phạt L2',
          prompt:
            'Cài đặt `newton_logistic(X, y, reg=1e-4, steps=10)` tối ưu Logistic Regression trên ma trận mở rộng $X_{\\text{aug}} = [X,\\mathbf{1}_n]$ với mục tiêu $J(\\theta) = \\frac{1}{n}\\sum_i (\\text{logaddexp}(0, z_i) - y_i z_i) + \\frac{\\text{reg}}{2}\\|\\theta\\|_2^2$ (với $z = X_{\\text{aug}}\\theta$). Khởi tạo $\\theta = \\mathbf{0}$, thực hiện `steps` bước giải `np.linalg.solve(H, g)` cập nhật $\\theta \\leftarrow \\theta - H^{-1} g$, và trả về `(w, b)` với `w = theta[:-1]`, `b = float(theta[-1])`.',
          hint: 'Tạo `Xa = np.column_stack([X, np.ones(n)])`, `theta = np.zeros(d + 1)`. Ở mỗi bước: tính `z = Xa @ theta`, `p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))`, `s = p * (1.0 - p)`, `g = (Xa.T @ (p - y)) / n + reg * theta`, `H = (Xa.T * s) @ Xa / n + reg * np.eye(d + 1)`. Giải `delta = np.linalg.solve(H, g)` và cập nhật `theta -= delta`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_nw = np.random.default_rng(77)
_Xn = np.column_stack([_rng_nw.normal(size=80) * 50.0, _rng_nw.normal(size=80) * 0.05])
_logits_n = 0.08 * _Xn[:, 0] - 40.0 * _Xn[:, 1] + 0.5
_yn = (_logits_n > 0).astype(float)

def _grad_norm_log(X, y, w, b, reg=1e-4):
    Xa = np.column_stack([X, np.ones(len(X))])
    th = np.r_[w, b]
    z = Xa @ th
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    g = (Xa.T @ (p - y)) / len(X) + reg * th
    return float(np.linalg.norm(g))
`,
          starter: String.raw`def newton_logistic(X, y, reg=1e-4, steps=10):
    raise NotImplementedError
`,
          solution: String.raw`def newton_logistic(X, y, reg=1e-4, steps=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    Xa = np.column_stack([X, np.ones(n, dtype=float)])
    theta = np.zeros(d + 1, dtype=float)
    I = np.eye(d + 1, dtype=float)
    for _ in range(steps):
        z = np.clip(Xa @ theta, -500.0, 500.0)
        p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        s = p * (1.0 - p)
        g = (Xa.T @ (p - y)) / n + reg * theta
        H = ((Xa.T * s) @ Xa) / n + reg * I
        delta = np.linalg.solve(H, g)
        theta = theta - delta
    return theta[:-1], float(theta[-1])
`,
          test: String.raw`_report('m3-u4-newton-logistic',
    _check('Hội tụ đạt chuẩn gradient ||∇J(θ)|| < 1e-8 trong ≤ 10 bước bất chấp 2 cột chênh lệch 1000 lần thang đo', lambda: (
        (lambda w, b: _grad_norm_log(_Xn, _yn, w, b, reg=1e-4) < 1e-8)(*newton_logistic(_Xn, _yn, reg=1e-4, steps=10)),
        'Chứng minh tính bất biến Affine và hội tụ bậc hai của phương pháp Newton.'
    )),
    _check('Tốc độ chính xác phân loại ≥ 95% trên tập dữ liệu tách biệt gần như tuyến tính', lambda: (
        (lambda w, b: np.mean(((_Xn @ w + b) > 0) == _yn) >= 0.95)(*newton_logistic(_Xn, _yn, reg=1e-4, steps=10)),
        'Kiểm tra độ chính xác dự đoán thực tế.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong thuật toán Newton cho hàm mất mát phi lồi (non-convex), ma trận Hessian $H(w_t)$ có thể xuất hiện trị riêng âm (điểm yên ngựa hoặc vùng lõm). Chuyện gì xảy ra nếu cứ thực hiện $-H^{-1} g$, và kỹ thuật nào khắc phục điều này?',
          guidance:
            'Tại hướng tương ứng trị riêng âm $\\lambda_k < 0$, bước nhảy $-H^{-1} g$ sẽ đi **ngược lên dốc** thay vì xuống dốc — thu hút nghiệm về thẳng điểm yên ngựa (saddle point) thay vì cực tiểu! Kỹ thuật sửa đổi phổ biến là Damped/Modified Newton: cộng thêm $\\gamma I$ (Levenberg–Marquardt) sao cho $H + \\gamma I \\succ 0$, hoặc Trust-Region Newton để bảo đảm hướng xuống dốc.',
        },
      ],
    },
  ],
};
