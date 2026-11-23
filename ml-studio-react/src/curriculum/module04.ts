import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_04: GradModule = {
  id: 4,
  title: 'Mô hình Sinh, Kernel & Support Vector Machines',
  englishTitle: 'Generative Classifiers, Kernel Methods & SVM',
  color: '#f28db9',
  status: 'ready',
  overview:
    'So sánh sâu sắc hai hệ hình Generative vs Discriminative qua GDA (LDA/QDA) và Multinomial Naive Bayes; thiết lập hình học lề cực đại Hard/Soft-Margin SVM từ bài toán gốc (Primal / Hinge Loss) sang đối ngẫu Lagrange và điều kiện KKT; cùng lý thuyết ánh xạ vào không gian Hilbert (RKHS), Định lý Mercer và Kernel Trick.',
  outcomes: [
    'Chứng minh biên quyết định của GDA với ma trận hiệp phương sai dùng chung (LDA) là tuyến tính theo x và có dạng bậc hai ở QDA.',
    'Cài đặt Multinomial Naive Bayes có làm mượt Laplace bằng tính toán trong không gian log ổn định số.',
    'Chuyển đổi bài toán Soft-Margin SVM từ Primal sang Dual Lagrange và khôi phục trực tiếp tham số $(w, b)$ từ hệ số đối ngẫu $\\alpha$ qua KKT.',
    'Chứng minh tính hợp lệ của hàm nhân theo Định lý Mercer và cài đặt Kernel Ridge Regression cho dữ liệu phi tuyến.',
  ],
  sources: [
    'Stanford CS229 — Machine Learning Complete Notes (Part II: Generative Learning & Part III: Kernel Methods / SVM)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 9: Discriminant Analysis & Naive Bayes; Ch. 17: Kernel Methods)',
    'Vũ Hữu Tiệp — Machine Learning cơ bản (Chương 19–22: SVM, Soft-Margin & Kernel SVM)',
  ],
  units: [
    {
      id: 'm4-u1',
      moduleId: 4,
      order: 1,
      title: 'Generative vs. Discriminative; GDA (LDA/QDA) & Naive Bayes',
      summary:
        'Sự khác biệt bản chất giữa học phân phối đồng thời $p(x, y) = p(x \\mid y)p(y)$ (Generative) và học trực tiếp hậu nghiệm $p(y \\mid x)$ (Discriminative); phân tích Gaussian Discriminant Analysis (LDA vs QDA) và Multinomial Naive Bayes cho văn bản.',
      outcomes: [
        'Phân biệt giả định thống kê và đánh đổi hiệu quả mẫu (sample complexity) giữa Logistic Regression và Naive Bayes / GDA.',
        'Dẫn xuất công thức hậu nghiệm tuyến tính chính xác của LDA khi hai lớp chia sẻ cùng ma trận hiệp phương sai $\\Sigma$.',
        'Cài đặt Multinomial Naive Bayes ổn định số với làm mượt Laplace $\\alpha$ trong không gian log.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part II — Generative Learning Algorithms (GDA & Naive Bayes)',
          note: 'Dẫn xuất MLE cho GDA và so sánh Ng–Jordan với Logistic Regression.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§9.2–9.3 (Gaussian Discriminant Analysis & Naive Bayes)',
          note: 'LDA, QDA, Regularized Covariance và Dirichlet-Multinomial smoothing.',
        },
      ],
      sections: [
        {
          heading: '1. Hệ hình Generative vs. Discriminative & Định lý Ng–Jordan',
          body: String.raw`Mô hình **Discriminative** (như Logistic Regression) mô hình hóa trực tiếp $p(y \mid x)$ hoặc ánh xạ $x \mapsto y$, không quan tâm cấu trúc phân phối của đặc trưng $x$. Ngược lại, mô hình **Generative** mô hình hóa mật độ sinh dữ liệu từng lớp $p(x \mid y=c)$ kết hợp tiên nghiệm $p(y=c)$, rồi suy ra hậu nghiệm bằng Định lý Bayes: $p(y=c \mid x) \propto p(x \mid y=c)p(y=c)$.

Kết quả kinh điển của Ng & Jordan (2001): nếu giả định sinh dữ liệu (như Gaussian hay độc lập có điều kiện) gần đúng, mô hình Generative hội tụ tới giới hạn tiệm cận chỉ với $O(\log d)$ mẫu — nhanh hơn $O(d)$ mẫu của Logistic Regression; nhưng khi $n \to \infty$, Logistic Regression thường đạt sai số tiệm cận thấp hơn do không bị thiên vị bởi giả định phân phối của $x$.`,
        },
        {
          heading: '2. Gaussian Discriminant Analysis: Từ QDA tới biên tuyến tính LDA',
          body: String.raw`GDA giả định $x \mid y=c \sim \mathcal{N}(\mu_c, \Sigma_c)$ và $y \sim \text{Bernoulli}(\phi)$ (hoặc Categorical). Nếu mỗi lớp có $\Sigma_c$ riêng, hiệu log-mật độ giữa hai lớp là hàm bậc hai theo $x$, sinh ra biên cong **QDA (Quadratic Discriminant Analysis)**.

Nếu áp đặt các lớp chia sẻ chung ma trận hiệp phương sai $\Sigma_0 = \Sigma_1 = \Sigma$ (**LDA — Linear Discriminant Analysis**), số hạng bậc hai $-\frac{1}{2}x^\top \Sigma^{-1} x$ triệt tiêu hoàn toàn, khiến hậu nghiệm $p(y=1 \mid x)$ quy về đúng hàm Sigmoid của Logistic Regression:`,
          math: String.raw`p(y=1 \mid x) = \sigma(w^\top x + b), \qquad w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad b = -\frac{1}{2}\mu_1^\top\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_0^\top\Sigma^{-1}\mu_0 + \ln\frac{\phi}{1-\phi}`,
          intuition:
            'GDA và Logistic Regression chia sẻ cùng dạng hàm hậu nghiệm $p(y=1 \\mid x) = \\sigma(w^\\top x + b)$, nhưng ước lượng hai bộ $(w, b)$ khác nhau hoàn toàn do tối ưu các hàm hợp lý khác nhau ($p(x, y)$ vs $p(y \\mid x)$).',
        },
        {
          heading: '3. Multinomial Naive Bayes & Làm mượt Laplace trong không gian Log',
          body: String.raw`Với dữ liệu đếm từ vựng $x \in \mathbb{N}^d$, Naive Bayes giả định các tọa độ độc lập có điều kiện theo lớp $y$: $p(x \mid y=c) \propto \prod_{j=1}^d \theta_{c, j}^{x_j}$. Để tránh xác suất $0$ tuyệt đối (zero-frequency problem) khi một từ trong test chưa từng xuất hiện ở lớp $c$ trong train, ta thêm giả đếm $\alpha > 0$ (Laplace/Lidstone smoothing):`,
          math: String.raw`\hat{\theta}_{c, j} = \frac{\sum_{i: y_i = c} x_{ij} + \alpha}{\sum_{i: y_i = c}\sum_{v=1}^d x_{iv} + \alpha d}, \qquad \log p(y=c, x) = \log \hat{\pi}_c + \sum_{j=1}^d x_j \log \hat{\theta}_{c, j}`,
          pitfall:
            'Nhân trực tiếp hàng trăm xác suất $\\hat{\\theta}_{c, j}$ trong văn bản dài sẽ sụp đổ về đúng `0.0` (floating-point underflow). Luôn tính tổng trong không gian `log` và chuẩn hóa bằng Log-Sum-Exp.',
        },
      ],
      simulationId: 'm4-u1',
      exercises: [
        {
          id: 'm4-u1-lda',
          title: 'Binary LDA bằng Pooled Covariance MLE',
          prompt:
            'Cài đặt `fit_predict_lda(X_train, y_train, X_test)` cho bài toán nhị phân `y_train` $\\in \\{0, 1\\}$. Tính các tham số MLE: $\\phi = \\text{mean}(y)$, $\\mu_0, \\mu_1$, hiệp phương sai gộp $\\hat{\\Sigma} = \\frac{1}{n}(\\sum_{y_i=0}(x_i - \\mu_0)(x_i - \\mu_0)^\\top + \\sum_{y_i=1}(x_i - \\mu_1)(x_i - \\mu_1)^\\top) + 10^{-6} I_d$, sau đó trả về vector xác suất hậu nghiệm $P(y=1 \\mid x)$ cho `X_test`.',
          hint: 'Tách `X0 = X_train[y_train == 0]`, `X1 = X_train[y_train == 1]`, tính `mu0, mu1`, `Sigma = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / len(y_train) + 1e-6 * np.eye(d)`. Giải `w = np.linalg.solve(Sigma, mu1 - mu0)` và `b = -0.5 * (mu1 @ np.linalg.solve(Sigma, mu1) - mu0 @ np.linalg.solve(Sigma, mu0)) + np.log(phi / (1 - phi))`, rồi áp dụng Sigmoid.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_lda = np.random.default_rng(21)
_cov_true = np.array([[1.2, 0.7], [0.7, 0.9]])
_X0_l = _rng_lda.multivariate_normal([-1.2, -0.8], _cov_true, size=60)
_X1_l = _rng_lda.multivariate_normal([1.3, 1.0], _cov_true, size=60)
_X_lda_tr = np.vstack([_X0_l, _X1_l])
_y_lda_tr = np.r_[np.zeros(60), np.ones(60)]
_X_lda_te = np.array([[-2.0, -1.5], [0.05, 0.1], [2.0, 1.8]])

def _bayes_lda_post1(X_tr, y_tr, X_te):
    X0, X1 = X_tr[y_tr == 0], X_tr[y_tr == 1]
    phi = np.mean(y_tr)
    mu0, mu1 = X0.mean(0), X1.mean(0)
    S = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / len(y_tr) + 1e-6 * np.eye(X_tr.shape[1])
    invS = np.linalg.inv(S)
    l0 = -0.5 * np.einsum('ni,ij,nj->n', X_te - mu0, invS, X_te - mu0) + math.log(1.0 - phi)
    l1 = -0.5 * np.einsum('ni,ij,nj->n', X_te - mu1, invS, X_te - mu1) + math.log(phi)
    return 1.0 / (1.0 + np.exp(l0 - l1))
`,
          starter: String.raw`def fit_predict_lda(X_train, y_train, X_test):
    raise NotImplementedError
`,
          solution: String.raw`def fit_predict_lda(X_train, y_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    n, d = X_train.shape
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    phi = float(np.mean(y_train))
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    D0 = X0 - mu0
    D1 = X1 - mu1
    Sigma = (D0.T @ D0 + D1.T @ D1) / n + 1e-6 * np.eye(d)
    inv_mu1 = np.linalg.solve(Sigma, mu1)
    inv_mu0 = np.linalg.solve(Sigma, mu0)
    w = inv_mu1 - inv_mu0
    b = -0.5 * float(mu1 @ inv_mu1 - mu0 @ inv_mu0) + math.log(phi / (1.0 - phi))
    z = np.clip(X_test @ w + b, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))
`,
          test: String.raw`_report('m4-u1-lda',
    _check('Khớp chính xác công thức hậu nghiệm Bayes Gaussian đa biến (< 1e-6)', lambda: (
        np.allclose(fit_predict_lda(_X_lda_tr, _y_lda_tr, _X_lda_te), _bayes_lda_post1(_X_lda_tr, _y_lda_tr, _X_lda_te), atol=1e-6),
        'Đối chiếu dạng tuyến tính σ(wᵀx + b) với mật độ chuẩn đa biến trực tiếp.'
    )),
    _check('Đạt độ chính xác Bayes tối ưu (≥ 85%) trên hai cụm Gaussian chồng lấn', lambda: (
        (lambda p: p[0] < 0.15 and 0.2 < p[1] < 0.8 and p[2] > 0.85)(fit_predict_lda(_X_lda_tr, _y_lda_tr, _X_lda_te))
        and np.mean((fit_predict_lda(_X_lda_tr, _y_lda_tr, _X_lda_tr) >= 0.5) == _y_lda_tr) >= 0.85,
        'Đánh giá khả năng phân tách hai cụm Gaussian có hiệp phương sai tương quan.'
    )),
)
`,
        },
        {
          id: 'm4-u1-nb',
          title: 'Multinomial Naive Bayes với Làm mượt Laplace (Log-Space)',
          prompt:
            'Cài đặt `multinomial_nb_log_probs(X_counts, y, X_test, alpha=1.0)` cho dữ liệu đếm không âm `X_counts` `(n, d)`, nhãn lớp `y` thuộc `{0, ..., K-1}` (với $K = \\max(y) + 1$). Tính tiên nghiệm $\\log\\hat{\\pi}_c$ và xác suất từ có làm mượt Laplace $\\alpha$, rồi trả về ma trận xác suất hậu nghiệm `(n_test, K)` bằng phép chuẩn hóa Log-Sum-Exp.',
          hint: String.raw`Với mỗi lớp $c \in [0..K-1]$, cộng tổng vector từ ` + '`counts_c = X_counts[y == c].sum(axis=0)`' + ', tính `log_theta[c] = np.log((counts_c + alpha) / (counts_c.sum() + alpha * d))` và `log_pi[c] = np.log(np.mean(y == c))`. Tính `log_joint = X_test @ log_theta.T + log_pi`, rồi trừ row-max trước khi `exp` và chuẩn hóa.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xnb = np.array([
    [3., 2., 0., 0.],
    [2., 4., 1., 0.],
    [0., 0., 3., 2.],
    [0., 1., 2., 4.],
])
_ynb = np.array([0, 0, 1, 1])
_Xnb_te = np.array([[500., 500., 0., 0.], [0., 0., 5., 5.], [0., 0., 0., 10.]])
`,
          starter: String.raw`def multinomial_nb_log_probs(X_counts, y, X_test, alpha=1.0):
    raise NotImplementedError
`,
          solution: String.raw`def multinomial_nb_log_probs(X_counts, y, X_test, alpha=1.0):
    X_counts = np.asarray(X_counts, dtype=float)
    y = np.asarray(y, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    n, d = X_counts.shape
    K = int(np.max(y)) + 1
    log_theta = np.zeros((K, d), dtype=float)
    log_pi = np.zeros(K, dtype=float)
    for c in range(K):
        Xc = X_counts[y == c]
        counts_c = Xc.sum(axis=0)
        log_theta[c] = np.log((counts_c + alpha) / (counts_c.sum() + alpha * d))
        log_pi[c] = np.log(len(Xc) / n)
    log_joint = X_test @ log_theta.T + log_pi
    shifted = log_joint - np.max(log_joint, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)
`,
          test: String.raw`_report('m4-u1-nb',
    _check('An toàn trước văn bản cực dài (500 lần lặp từ, không underflow về 0/0)', lambda: (
        (lambda p: np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1.0) and p[0, 0] > 0.99)(
            multinomial_nb_log_probs(_Xnb, _ynb, _Xnb_te, alpha=1.0)
        ),
        'Log-Sum-Exp ngăn chặn underflow tuyệt đối.'
    )),
    _check('Làm mượt Laplace α > 0 ngăn xác suất lớp 0 sụp về 0 dù từ vựng vắng mặt', lambda: (
        (lambda p: p[2, 1] > p[2, 0] and p[2, 0] > 0.0)(
            multinomial_nb_log_probs(_Xnb, _ynb, _Xnb_te, alpha=1.0)
        ),
        'Kiểm tra smoothing cho từ chưa xuất hiện trong tập huấn luyện của lớp 0.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Chứng minh mối liên hệ chính xác về giả định phân phối khiến GDA dẫn tới hàm hậu nghiệm Logistic Regression: nếu $p(x \\mid y=c)$ là thành viên thuộc Exponential Family có chung tham số phân tán (như Gaussian đồng phương sai $\\Sigma$ hoặc Poisson hoặc Bernoulli độc lập), log-odds $\\ln\\frac{p(y=1\\mid x)}{p(y=0\\mid x)}$ luôn tuyến tính theo $x$.',
          guidance:
            'Với $p(x \\mid y=c) = b(x)\\exp(\\eta_c^\\top x - a(\\eta_c))$, trong tỉ số $p(x \\mid y=1) / p(x \\mid y=0)$, nhân tử $b(x)$ (chứa toàn bộ số hạng bậc hai $x^\\top \\Sigma^{-1} x$ đối với Gaussian đồng $\\Sigma$) triệt tiêu hoàn toàn! Phần còn lại trong số mũ chỉ còn $(\\eta_1 - \\eta_0)^\\top x - (a(\\eta_1) - a(\\eta_0)) + \\ln\\frac{\\pi_1}{\\pi_0}$, chính là hàm tuyến tính $w^\\top x + b$.',
        },
      ],
    },
    {
      id: 'm4-u2',
      moduleId: 4,
      order: 2,
      title: 'Maximum Margin Classifier & Hard/Soft-Margin SVM (Primal & Dual)',
      summary:
        'Tối ưu hóa lề hình học cực đại (Maximum Margin): công thức hóa Primal dưới góc nhìn Hinge Loss + $L_2$, dẫn xuất bài toán đối ngẫu Lagrange QP (Dual), ý nghĩa thưa của điều kiện KKT xác định Support Vectors và cách khôi phục $(w, b)$ từ nghiệm đối ngẫu $\\alpha$.',
      outcomes: [
        'Chứng minh khoảng cách hình học hai phía lề là $\\frac{2}{\\|w\\|_2}$, từ đó dẫn xuất mục tiêu tối thiểu hóa $\\frac{1}{2}\\|w\\|_2^2$.',
        'Thiết lập hàm mất mát Soft-Margin Hinge Loss không ràng buộc và cài đặt bước Subgradient Descent.',
        'Viết điều kiện đối ngẫu KKT cho Soft-Margin SVM và giải thích ba nhóm điểm dữ liệu ($\\alpha_i = 0$, $0 < \\alpha_i < C$, $\\alpha_i = C$).',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part III — Support Vector Machines (Primal, Dual & KKT)',
          note: 'Lề hình học, lề hàm số và biến đổi Lagrange QP chuẩn mực.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/book_ML.pdf',
          section: 'Chương 19–20: Support Vector Machine & Soft Margin SVM',
          note: 'Giải thích trực quan hình học biên lề và biến giả (slack variables) ξᵢ.',
        },
      ],
      sections: [
        {
          heading: '1. Từ Lề Hình học Cực đại tới Hinge Loss (Primal Formulation)',
          body: String.raw`Với nhãn $y_i \in \{-1, +1\}$ và siêu phẳng $w^\top x + b = 0$, bằng cách chuẩn hóa tỷ lệ $(w, b)$ sao cho điểm gần biên nhất đạt lề hàm số $|w^\top x + b| = 1$, khoảng cách hình học giữa hai siêu phẳng biên $w^\top x + b = \pm 1$ đúng bằng $\frac{2}{\|w\|_2}$. Cực đại hóa lề tương đương tối thiểu hóa $\frac{1}{2}\|w\|_2^2$.

Khi dữ liệu không tách biệt tuyến tính hoàn toàn hoặc có nhiễu, ta đưa thêm biến trượt (slack variables) $\xi_i = \max(0, 1 - y_i(w^\top x_i + b))$ với hệ số phạt $C > 0$, thu được dạng tối ưu không ràng buộc **Hinge Loss + L2 Regularization**:`,
          math: String.raw`\min_{w, b} \; \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^n \max\!\big(0,\; 1 - y_i(w^\top x_i + b)\big)`,
          intuition:
            'Bất kỳ điểm nào nằm đúng phía và vượt ra ngoài hành lang lề ($y_i(w^\\top x_i + b) > 1$) có Hinge Loss bằng $0$ và dưới vi phân đóng góp bằng $0$: chúng không hề ảnh hưởng tới $w$ và $b$.',
        },
        {
          heading: '2. Đối ngẫu Lagrange & Điều kiện KKT cho Support Vectors',
          body: String.raw`Đưa ràng buộc $y_i(w^\top x_i + b) \ge 1 - \xi_i$, $\xi_i \ge 0$ vào hàm Lagrange với nhân tử đối ngẫu $\alpha_i \ge 0$, đạo hàm theo biến gốc $(w, b, \xi)$ cho điều kiện stationarity $w = \sum_{i=1}^n \alpha_i y_i x_i$ và $\sum_{i=1}^n \alpha_i y_i = 0$, đưa về bài toán Quy hoạch toàn phương (QP) thuần túy theo tích vô hướng cặp $x_i^\top x_j$:`,
          math: String.raw`\max_{0 \le \alpha_i \le C} \;\sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j (x_i^\top x_j) \quad \text{s.t. } \sum_{i=1}^n \alpha_i y_i = 0`,
          intuition:
            'Điều kiện bù KKT $\\alpha_i [y_i(w^\\top x_i + b) - 1 + \\xi_i] = 0$ phân tách mọi mẫu dữ liệu làm ba nhóm rõ rệt:\n- $\\alpha_i = 0$: Điểm an toàn nằm ngoài hành lang lề ($y_i f(x_i) > 1$).\n- $0 < \\alpha_i < C$ (**Free Support Vectors**): Nằm **chính xác trên đường biên lề** với $\\xi_i = 0$ và $y_i(w^\\top x_i + b) = 1$, dùng để khôi phục trực tiếp hệ số chặn $b$.\n- $\\alpha_i = C$: Vi phạm biên lề nằm trong hành lang hoặc bị phân loại sai ($\\xi_i > 0$).',
        },
      ],
      simulationId: 'm4-u2',
      exercises: [
        {
          id: 'm4-u2-hinge-svm',
          title: 'Bước Subgradient Descent trên Soft-Margin SVM Primal',
          prompt:
            'Cài đặt `svm_hinge_subgrad_step(X, y_pm1, w, b, C=1.0, lr=0.01)` thực hiện **một bước cập nhật dưới gradient** tối ưu hàm mục tiêu $J(w, b) = \\frac{1}{2}\\|w\\|_2^2 + C \\sum_{i=1}^n \\max(0, 1 - y_i(w^\\top x_i + b))$ với `y_pm1` $\\in \\{-1, +1\\}$. Trả về `(w_new, b_new)`.',
          hint: 'Tính margin `m = y_pm1 * (X @ w + float(b))`, mặt nạ `active = m < 1.0`. Dưới gradient `dw = w.copy()` và `db = 0.0`; nếu `np.any(active)`: `dw = dw - C * (X[active].T @ y_pm1[active])`, `db = -C * float(np.sum(y_pm1[active]))`. Trả về `w - lr * dw, float(b - lr * db)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xsv = np.array([[2., 2.], [1.5, 2.5], [3., 1.], [-2., -1.5], [-1.5, -2.5], [-2.5, -1.]])
_ysv = np.array([1., 1., 1., -1., -1., -1.])
def _svm_obj(X, y, w, b, C):
    return 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - y * (X @ w + b))))
def _run_hinge_svm(X, y, C=2.0, lr=0.01, steps=200):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        w, b = svm_hinge_subgrad_step(X, y, w, b, C=C, lr=lr)
    return w, b
`,
          starter: String.raw`def svm_hinge_subgrad_step(X, y_pm1, w, b, C=1.0, lr=0.01):
    raise NotImplementedError
`,
          solution: String.raw`def svm_hinge_subgrad_step(X, y_pm1, w, b, C=1.0, lr=0.01):
    X = np.asarray(X, dtype=float)
    y_pm1 = np.asarray(y_pm1, dtype=float)
    w = np.asarray(w, dtype=float)
    margins = y_pm1 * (X @ w + float(b))
    active = margins < 1.0
    dw = w.copy()
    db = 0.0
    if np.any(active):
        dw = dw - float(C) * (X[active].T @ y_pm1[active])
        db = -float(C) * float(np.sum(y_pm1[active]))
    return w - lr * dw, float(b - lr * db)
`,
          test: String.raw`_report('m4-u2-hinge-svm',
    _check('Khi điểm đã vượt biên (margin >= 1), chỉ còn thành phần co cụm weight decay w ← (1 - lr)w và b giữ nguyên', lambda: (
        (lambda wn, bn: np.allclose(wn, 0.9 * np.array([2.0, 0.0])) and np.isclose(bn, 0.5))(
            *svm_hinge_subgrad_step(np.array([[2.0, 0.0], [-2.0, 0.0]]), np.array([1.0, -1.0]), np.array([2.0, 0.0]), 0.5, C=10.0, lr=0.1)
        ),
        'Các điểm thỏa margin >= 1 không đóng góp gradient cho Hinge Loss.'
    )),
    _check('Lặp 200 bước giảm mạnh mục tiêu Soft-Margin và phân tách 100% tập dữ liệu', lambda: (
        (lambda w, b: _svm_obj(_Xsv, _ysv, w, b, 2.0) < 0.8 and np.mean(np.sign(_Xsv @ w + b) == _ysv) == 1.0)(
            *_run_hinge_svm(_Xsv, _ysv, C=2.0, lr=0.01, steps=200)
        ),
        'Subgradient hội tụ về bộ phân tách có lề ổn định.'
    )),
)
`,
        },
        {
          id: 'm4-u2-dual-reconstruct',
          title: 'Khôi phục Primal (w, b) từ Nghiệm Đối ngẫu α & Điều kiện KKT',
          prompt:
            'Cho bộ nghiệm đối ngẫu `alphas` thu được từ giải QP với hằng số `C`. Cài đặt `svm_primal_from_dual(X, y_pm1, alphas, C)` tính $w = \\sum_{i=1}^n \\alpha_i y_i x_i$, sau đó khôi phục hệ số chặn $b$ bằng cách lấy trung bình $y_i - w^\\top x_i$ trên các **Free Support Vectors** ($10^{-6} < \\alpha_i < C - 10^{-6}$). Nếu tập Free SV rỗng, lấy trung bình trên các điểm $\\alpha_i > 10^{-6}$.',
          hint: 'Tính `w = X.T @ (alphas * y_pm1)`. Tạo `free = (alphas > 1e-6) & (alphas < C - 1e-6)`; nếu không có phần tử nào thỏa thì lấy `free = alphas > 1e-6`. Trả về `w, float(np.mean(y_pm1[free] - X[free] @ w))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xd = np.array([[1., 1.], [-1., -1.], [3., 3.], [-3., -4.]])
_yd = np.array([1., -1., 1., -1.])
# Exact Hard/Soft margin solution for +/- [1, 1]: w = [0.5, 0.5], b = 0, alphas = [0.25, 0.25, 0, 0]
_ad = np.array([0.25, 0.25, 0.0, 0.0])
`,
          starter: String.raw`def svm_primal_from_dual(X, y_pm1, alphas, C):
    raise NotImplementedError
`,
          solution: String.raw`def svm_primal_from_dual(X, y_pm1, alphas, C):
    X = np.asarray(X, dtype=float)
    y_pm1 = np.asarray(y_pm1, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    w = X.T @ (alphas * y_pm1)
    free = (alphas > 1e-6) & (alphas < C - 1e-6)
    if not np.any(free):
        free = alphas > 1e-6
    b = float(np.mean(y_pm1[free] - X[free] @ w)) if np.any(free) else 0.0
    return w, b
`,
          test: String.raw`_report('m4-u2-dual-reconstruct',
    _check('Khôi phục chính xác w = [0.5, 0.5] và b = 0.0 (độ rộng lề 2/||w|| = 2√2)', lambda: (
        (lambda w, b: np.allclose(w, [0.5, 0.5]) and abs(b) < 1e-9)(*svm_primal_from_dual(_Xd, _yd, _ad, C=10.0)),
        'Kiểm tra KKT y_i(wᵀx_i + b) = 1 tại các Support Vector.'
    )),
    _check('Chỉ dùng Free Support Vectors (bỏ qua mẫu α = C bị ràng buộc hộp)', lambda: (
        (lambda w, b: abs(b - 0.0) < 1e-9)(*svm_primal_from_dual(np.vstack([_Xd, [0.0, 0.0]]), np.r_[_yd, 1.0], np.array([0.25, 0.25, 0.0, 0.0, 1.0]), C=1.0)),
        'Mẫu có α = C = 1.0 là outlier nằm sai lề không thỏa đẳng thức = 1.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao tăng siêu tham số $C$ trong Soft-Margin SVM lại tương ứng với **giảm** mức độ regularization (giảm Bias nhưng tăng Variance)? So sánh tương ứng giữa $C$ của SVM và $\\lambda$ của Ridge Regression.',
          guidance:
            'Trong mục tiêu $\\frac{1}{2}\\|w\\|_2^2 + C \\sum \\xi_i$, chia hai vế cho $C$ cho thấy hệ số nhân trước $\\|w\\|_2^2$ chính là $\\lambda = \\frac{1}{C}$. Vì vậy $C$ càng lớn thì phạt chuẩn $\\|w\\|_2^2$ càng nhỏ, buộc mô hình gồng mình sửa bằng mọi giá các điểm nhiễu gần biên (lề hẹp, dễ overfit); ngược lại $C$ nhỏ ưu tiên lề rộng hơn là cố phân tách sạch mọi điểm.',
        },
      ],
    },
    {
      id: 'm4-u3',
      moduleId: 4,
      order: 3,
      title: 'Kernel Trick, Định lý Mercer & RKHS (RBF, Polynomial Kernels)',
      summary:
        'Biến đổi phi tuyến vào không gian đặc trưng vô hạn chiều chỉ với chi phí $O(n^2)$: Định lý Biểu diễn (Representer Theorem), Định lý Mercer cho ma trận Gram nửa xác định dương, RBF Kernel và Kernel Ridge Regression.',
      outcomes: [
        'Phát biểu Representer Theorem giải thích vì sao nghiệm tối ưu nằm hoàn toàn trong bao tuyến tính của các điểm dữ liệu qua hàm nhân: $f(x) = \\sum_{i=1}^n \\alpha_i K(x_i, x)$.',
        'Kiểm tra tính hợp lệ của một hàm nhân bất kỳ bằng điều kiện Mercer (đối xứng và ma trận Gram $K \\succeq 0$).',
        'Cài đặt hàm nhân RBF vector hóa và thuật toán Kernel Ridge Regression giải dữ liệu phi tuyến.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part III — Kernels & Mercer’s Theorem',
          note: 'Ánh xạ đặc trưng φ(x), đa thức bậc cao và điều kiện ma trận Gram K ≽ 0.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§17.1–17.3 (Kernel Methods, RKHS & Kernel Ridge Regression)',
          note: 'Representer Theorem và Dual Trick biến đổi từ không gian d chiều sang n×n.',
        },
      ],
      sections: [
        {
          heading: '1. Kernel Trick & Định lý Mercer',
          body: String.raw`Xét ánh xạ phi tuyến đưa dữ liệu vào không gian chiều rất cao $\phi: \mathbb{R}^d \to \mathcal{H}$. Nếu một thuật toán (như SVM Dual, Ridge Dual, PCA, Perceptron) chỉ phụ thuộc vào tích vô hướng $\langle \phi(x), \phi(z) \rangle$, ta không cần tính tường minh $\phi(x)$ (vốn có thể vô hạn chiều!) mà chỉ cần đánh giá hàm nhân **Kernel** $K(x, z) = \langle \phi(x), \phi(z) \rangle$.

- **Polynomial Kernel**: $K(x, z) = (x^\top z + c)^p$ ánh xạ tương đương toàn bộ đơn thức tới bậc $p$ trong chi phí $O(d)$ thay vì $O\binom{d+p}{p}$.
- **Gaussian RBF Kernel**: $K(x, z) = \exp(-\gamma \|x - z\|_2^2)$ tương ứng không gian Hilbert vô hạn chiều ($\text{dim}(\mathcal{H}) = \infty$).

**Định lý Mercer**: Một hàm đối xứng $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ là hàm nhân hợp lệ (tồn tại tích vô hướng $\langle \phi(x), \phi(z)\rangle$ tương ứng) khi và chỉ khi với mọi tập hữu hạn điểm $\{x_1, \dots, x_n\}$, ma trận Gram $K_{ij} = K(x_i, x_j)$ đều **nửa xác định dương ($K \succeq 0$)**.`,
        },
        {
          heading: '2. Representer Theorem & Kernel Ridge Regression',
          body: String.raw`Theo **Representer Theorem**, nghiệm cực tiểu của mọi hàm mất mát điều chuẩn trên Không gian Hilbert tái sinh (RKHS) có dạng $\min_{f \in \mathcal{H}_K} \frac{1}{n}\sum_{i=1}^n \ell(f(x_i), y_i) + \lambda \|f\|_{\mathcal{H}_K}^2$ luôn viết được dưới dạng tổ hợp hữu hạn tại chính các điểm huấn luyện: $f(x) = \sum_{i=1}^n \alpha_i K(x_i, x)$. Áp dụng cho bài toán bình phương (Kernel Ridge Regression), ta thu được nghiệm đóng giải trên ma trận $n \times n$:`,
          math: String.raw`\alpha = (K_{\text{train}} + \lambda I_n)^{-1} y, \qquad \hat{y}_{\text{test}} = K_{\text{test, train}}\, \alpha`,
          pitfall:
            'Kernel methods dịch chuyển độ phức tạp huấn luyện từ không gian chiều đặc trưng $d$ sang số lượng mẫu $n$ ($O(n^3)$ để nghịch đảo ma trận $n \\times n$). Khi $n > 100\\,000$, Kernel chuẩn rất chậm; các kỹ thuật xấp xỉ như **Random Fourier Features** hoặc **Nyström** biến đổi ngược về đặc trưng ngẫu nhiên $D \\ll n$ chiều.',
        },
      ],
      simulationId: 'm4-u3',
      exercises: [
        {
          id: 'm4-u3-rbf-krr',
          title: 'Vectorized RBF Kernel & Kernel Ridge Regression',
          prompt:
            'Cài đặt hai hàm:\n1. `rbf_kernel(A, B, gamma)`: nhận `A` `(n, d)` và `B` `(m, d)`, trả về ma trận `(n, m)` có phần tử $\\exp(-\\gamma \\|a_i - b_j\\|_2^2)$ không dùng vòng lặp Python.\n2. `fit_predict_krr(X_train, y_train, X_test, gamma=1.0, lam=1e-2)`: giải nghiệm đối ngẫu $(K_{\\text{train}} + \\lambda I_n)\\alpha = y_{\\text{train}}$ và trả về vector dự đoán $K_{\\text{test, train}} \\alpha$.',
          hint: 'Dùng broadcasting `sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)` để tạo `np.exp(-gamma * sq)`. Sau đó giải `alpha = np.linalg.solve(K_tr + lam * np.eye(len(X_train)), y_train)` và trả về `rbf_kernel(X_test, X_train, gamma) @ alpha`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_k = np.random.default_rng(19)
_th_k = np.linspace(0, 2 * np.pi, 32, endpoint=False)
_X_in = 0.5 * np.column_stack([np.cos(_th_k), np.sin(_th_k)])
_X_out = 2.0 * np.column_stack([np.cos(_th_k), np.sin(_th_k)])
_Xk_tr = np.vstack([_X_in, _X_out])
_yk_tr = np.r_[np.ones(32), -np.ones(32)]

def _no_loops(fn):
    forbidden = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    return not any(isinstance(node, forbidden) for node in ast.walk(ast.parse(inspect.getsource(fn))))
`,
          starter: String.raw`def rbf_kernel(A, B, gamma=1.0):
    raise NotImplementedError

def fit_predict_krr(X_train, y_train, X_test, gamma=1.0, lam=1e-2):
    raise NotImplementedError
`,
          solution: String.raw`def rbf_kernel(A, B, gamma=1.0):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    sq_dists = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-float(gamma) * sq_dists)

def fit_predict_krr(X_train, y_train, X_test, gamma=1.0, lam=1e-2):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    K_tr = rbf_kernel(X_train, X_train, gamma)
    alpha = np.linalg.solve(K_tr + float(lam) * np.eye(len(X_train)), y_train)
    K_te = rbf_kernel(X_test, X_train, gamma)
    return K_te @ alpha
`,
          test: String.raw`_report('m4-u3-rbf-krr',
    _check('Ma trận Gram K_tr đối xứng, đường chéo bằng 1, nửa xác định dương (λ_min ≥ 0) & vector hóa không vòng lặp', lambda: (
        (lambda K: (
            _no_loops(rbf_kernel)
            and K.shape == (64, 64)
            and np.allclose(np.diag(K), 1.0)
            and np.allclose(K, K.T)
            and np.min(np.linalg.eigvalsh(K)) >= -1e-9
        ))(rbf_kernel(_Xk_tr, _Xk_tr, gamma=1.0)),
        'Kiểm chứng định lý Mercer trên RBF Kernel.'
    )),
    _check('KRR phân tách hoàn toàn 100% bài toán hai vòng tròn đồng tâm phi tuyến', lambda: (
        (lambda pred: np.mean(np.sign(pred) == _yk_tr) == 1.0 and pred[0] > 0.7 and pred[40] < -0.7)(
            fit_predict_krr(_Xk_tr, _yk_tr, _Xk_tr, gamma=1.5, lam=1e-2)
        ),
        'Mô hình tuyến tính thường chỉ đạt 50% trên hai vòng đồng tâm, trong khi RBF KRR đạt 100%.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Giải thích vì sao khi $\\gamma \\to \\infty$ trong RBF Kernel $K(x, z) = \\exp(-\\gamma\\|x - z\\|_2^2)$, ma trận Gram tiến về ma trận đơn vị $I_n$. Điều này ảnh hưởng thế nào tới hành vi khái quát hóa (Overfitting vs Underfitting) của Kernel SVM / KRR?',
          guidance:
            'Khi $\\gamma$ quá lớn, hàm nhân co thành đỉnh nhọn cực hẹp quanh từng điểm huấn luyện: bất kỳ điểm test $x$ nào nằm hơi lệch khỏi điểm train $x_i$ đều có $K(x, x_i) \\approx 0$. Kết quả là mô hình ghi nhớ thuộc lòng 100% tập train (đạt train accuracy 100%) nhưng mất hoàn toàn khả năng nội suy trên tập test (High Variance / Overfitting cực đoan).',
        },
      ],
    },
  ],
};
