import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_06: GradModule = {
  id: 6,
  title: 'Học Không giám sát & Mô hình Biến tiềm ẩn',
  englishTitle: 'Unsupervised Learning & Latent Variable Models',
  color: '#ff9b6a',
  status: 'ready',
  overview:
    'Thuật toán phân cụm K-means, Hierarchical và Spectral Clustering (Graph Cut); ước lượng mật độ KDE và Gaussian Mixture Models (GMM); chứng minh hội tụ thuật toán EM qua Bất đẳng thức Jensen; giảm chiều PCA qua SVD, Kernel PCA, ICA và trực quan hóa phi tuyến t-SNE/UMAP.',
  outcomes: [
    'Phân tích hình học phân cụm phi lồi bằng chuẩn hóa phổ Graph Laplacian $L_{\\text{sym}}$ và thuật toán K-means.',
    'Ước lượng mật độ phi tham số (KDE) và tính log-likelihood mô hình hỗn hợp GMM ổn định số học với Log-Sum-Exp.',
    'Chứng minh bước E và bước M trong thuật toán EM đảm bảo Log-Likelihood không giảm nhờ Bất đẳng thức Jensen.',
    'Liên hệ PCA dưới hai góc nhìn tương đương: cực đại hóa phương sai chiếu và tối thiểu hóa sai số tái tạo qua SVD.',
  ],
  sources: [
    'UC Berkeley CS189 (Lecture 04–05: K-means, Density Estimation & GMM)',
    'Stanford CS229 (Part IV: Unsupervised Learning, PCA, EM)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 20–21)',
  ],
  units: [
    {
      id: 'm6-u1',
      moduleId: 6,
      order: 1,
      title: 'Phân cụm: K-means, Hierarchical & Spectral Clustering',
      summary:
        'Bài toán khám phá cấu trúc tiềm ẩn không nhãn: tối ưu hóa luân phiên trên hàm méo dạng $K$-means (và khởi tạo $K$-means++), phân cụm phân cấp Agglomerative và phép nhúng phổ Graph Laplacian cắt đồ thị cho các cụm phi lồi.',
      outcomes: [
        'Chứng minh K-means giảm đơn điệu hàm mục tiêu nội cụm qua hai bước gán nhãn và cập nhật tâm cụm.',
        'Phân biệt các tiêu chí liên kết phân cấp (Single, Complete, Average, Ward) và hiệu ứng chuỗi (chaining).',
        'Thiết lập ma trận Normalized Symmetric Laplacian $L_{\\text{sym}} = I - D^{-1/2} W D^{-1/2}$ để phân cụm dữ liệu phi lồi.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/',
          section: 'Lecture 04 (K-means & Hierarchical Clustering)',
          note: 'Tính hội tụ của Lloyd algorithm và đánh đổi giữa các phép đo liên kết phân cấp.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§21.3–21.5 (Clustering, Spectral Clustering & Normalized Cuts)',
          note: 'Liên hệ giữa vector riêng của Graph Laplacian và bài toán Normalized Cut.',
        },
      ],
      sections: [
        {
          heading: '1. K-means (Thuật toán Lloyd) & Tính hội tụ đơn điệu',
          body: String.raw`Cho tập dữ liệu không nhãn $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, thuật toán **$K$-means** tìm phân hoạch thành $K$ cụm với nhãn $c_i \in \{1,\dots,K\}$ và các tâm cụm $\mu_k \in \mathbb{R}^d$ nhằm cực tiểu hóa tổng bình phương khoảng cách nội cụm (Within-Cluster Sum of Squares / Distortion):`,
          math: String.raw`J(c, \mu) = \sum_{i=1}^N \|x_i - \mu_{c_i}\|_2^2`,
          intuition:
            'Đây là bài toán tối ưu hóa không lồi đồng thời trên $(c, \\mu)$. Lloyd dùng Coordinate Descent gồm hai bước luân phiên: (1) Cố định $\\mu$, gán $c_i = \\arg\\min_k \\|x_i - \\mu_k\\|_2^2$; (2) Cố định $c$, cập nhật $\\mu_k = \\frac{1}{|C_k|}\\sum_{i \\in C_k} x_i$. Vì mỗi bước đều giảm (hoặc giữ nguyên) $J$ và số phân hoạch hữu hạn ($\\le K^N$), thuật toán chắc chắn dừng tại một cực tiểu địa phương.',
          pitfall:
            'K-means giả định ngầm các cụm có dạng cầu đồng phương sai (Voronoi cells lồi) và rất nhạy cảm với tâm khởi tạo ngẫu nhiên. Khởi tạo $K$-means++ chọn tâm mới theo xác suất tỷ lệ với $D(x_i)^2$ giúp đạt chặn đảm bảo $O(\\log K)$ so với nghiệm tối ưu toàn cục.',
        },
        {
          heading: '2. Phân cụm phân cấp (Hierarchical Agglomerative Clustering)',
          body: String.raw`Thay vì cố định trước $K$, phân cụm phân cấp tích tụ (Agglomerative) bắt đầu với $N$ cụm đơn lẻ và gộp dần hai cụm gần nhất tạo cây Dendrogram. Khoảng cách giữa hai cụm $A, B$ quyết định hình dạng phân hoạch:
- **Single-linkage**: $\min_{a \in A, b \in B} d(a, b)$ — tách được biên phi tuyến nhưng dễ mắc hiệu ứng chuỗi (chaining) khi chỉ có vài điểm nhiễu nối hai cụm.
- **Complete-linkage**: $\max_{a \in A, b \in B} d(a, b)$ — ưu tiên cụm chặt, nhạy cảm với ngoại lệ.
- **Ward linkage**: gộp hai cụm làm tăng tổng phương sai nội cụm ít nhất.`,
          math: String.raw`\Delta(A, B) = \frac{|A|\,|B|}{|A| + |B|} \|\bar{x}_A - \bar{x}_B\|_2^2 \quad (\text{Ward linkage})`,
        },
        {
          heading: '3. Spectral Clustering & Normalized Graph Laplacian',
          body: String.raw`Khi các cụm có dạng vành đồng tâm hoặc mặt trăng đan xen (phi lồi), cả $K$-means lẫn khoảng cách tâm cụm đều thất bại. Ta xây dựng đồ thị tương đồng với ma trận kề đối xứng không âm $W_{ij} \ge 0$ (ví dụ nhân RBF $W_{ij} = \exp(-\|x_i - x_j\|_2^2 / 2\sigma^2)$) và ma trận bậc $D_{ii} = \sum_j W_{ij}$. Với mọi vector $f \in \mathbb{R}^N$, dạng toàn phương của Unnormalized Laplacian $L = D - W$ đo độ trơn qua các cạnh cắt:`,
          math: String.raw`f^\top L f = \frac{1}{2} \sum_{i,j=1}^N W_{ij} (f_i - f_j)^2, \qquad L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}`,
          intuition:
            'Nới lỏng liên tục bài toán tối ưu Normalized Cut dẫn tới tìm $k$ vector riêng ứng với $k$ trị riêng nhỏ nhất của $L_{\\text{sym}}$. Nếu đồ thị có $k$ thành phần liên thông tách biệt, $0$ là trị riêng bội $k$ và không gian con riêng gắn tọa độ trực giao hoàn hảo cho từng cụm.',
          pitfall:
            'Sau khi ghép $k$ vector riêng đầu tiên thành ma trận $V \\in \\mathbb{R}^{N \\times k}$, bước chuẩn hóa từng hàng $U_{i,:} = V_{i,:} / \\|V_{i,:}\\|_2$ (Ng-Jordan-Weiss) là bắt buộc trước khi chạy $K$-means nhằm khử sai khác bậc đỉnh.',
        },
      ],
      simulationId: 'm6-u1',
      exercises: [
        {
          id: 'm6-u1-kmeans',
          title: 'Cài đặt K-means Lloyd Algorithm',
          prompt:
            'Cài đặt hàm `kmeans_fit(X, init_centroids, max_iter=50)` nhận `X` shape `(n, d)` và `init_centroids` shape `(k, d)`.\n- Mỗi vòng lặp:\n  1. Gán nhãn `labels` bằng chỉ số tâm cụm gần nhất theo khoảng cách Euclid bình phương (dùng `np.argmin`).\n  2. Cập nhật `centroids[k]` là trung bình các điểm thuộc cụm `k` (nếu cụm rỗng thì giữ nguyên tâm cũ).\n  3. Dừng sớm nếu `labels` không đổi.\n- Trả về `(labels, centroids, distortion)` với `distortion = float(np.sum((X - centroids[labels]) ** 2))`.',
          hint: 'Tính khoảng cách bình phương vector hóa bằng `sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)` rồi lấy `new_labels = np.argmin(sq, axis=1)`. Cuối vòng lặp, tính `labels` chuẩn theo `centroids` cuối cùng trước khi tính `distortion`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_km = np.random.default_rng(7)
_X_km = np.vstack([
    _rng_km.normal([-4.0, 0.0], 0.35, size=(12, 2)),
    _rng_km.normal([4.0, 3.0], 0.35, size=(12, 2)),
    _rng_km.normal([0.0, -4.5], 0.35, size=(12, 2)),
])
_init_km = np.array([[-0.5, 0.2], [0.5, 0.2], [0.0, -0.3]])
`,
          starter: String.raw`def kmeans_fit(X, init_centroids, max_iter=50):
    raise NotImplementedError
`,
          solution: String.raw`def kmeans_fit(X, init_centroids, max_iter=50):
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    k = centroids.shape[0]
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(sq, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            mask = new_labels == c
            if np.any(mask):
                new_centroids[c] = X[mask].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            centroids = new_centroids
            labels = new_labels
            break
        centroids = new_centroids
        labels = new_labels
    labels = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
    distortion = float(np.sum((X - centroids[labels]) ** 2))
    return labels, centroids, distortion
`,
          test: String.raw`def _run_checks():
    c1 = _check('Shape & kieu dau ra', lambda: (
        lambda res: (
            len(res) == 3 and res[0].shape == (36,) and res[1].shape == (3, 2) and isinstance(res[2], float),
            f'shapes={getattr(res[0], "shape", None)}, {getattr(res[1], "shape", None)}, J={res[2]}'
        )
    )(kmeans_fit(_X_km, _init_km)))
    c2 = _check('Tach chinh xac 3 cum & Distortion thap', lambda: (
        lambda labels, centroids, J: (
            len(set(labels[:12])) == 1 and len(set(labels[12:24])) == 1 and len(set(labels[24:])) == 1
            and len({labels[0], labels[12], labels[24]}) == 3 and J < 15.0,
            f'unique_per_block={[labels[0], labels[12], labels[24]]}, J={J:.3f}'
        )
    )(*kmeans_fit(_X_km, _init_km)))
    init_J = float(np.sum((_X_km - _init_km[np.argmin(np.sum((_X_km[:, None, :] - _init_km[None, :, :])**2, axis=2), axis=1)])**2))
    c3 = _check('Ham meo dang giam sau hoi tu', lambda: (
        lambda _, __, J: (J < init_J * 0.2, f'J_final={J:.2f} vs J_init={init_J:.2f}')
    )(*kmeans_fit(_X_km, _init_km)))
    _report('m6-u1-kmeans', c1, c2, c3)
_run_checks()
`,
        },
        {
          id: 'm6-u1-spectral',
          title: 'Normalized Laplacian & Spectral Embedding',
          prompt:
            'Cài đặt `normalized_laplacian_embedding(W, k)` nhận ma trận tương đồng đối xứng `W` `(n, n)`:\n1. Tính ma trận chuẩn hóa $L_{\\text{sym}} = I - D^{-1/2} W D^{-1/2}$ với $D_{ii} = \\sum_j W_{ij}$.\n2. Phân rã trị riêng đối xứng `np.linalg.eigh(0.5 * (L_sym + L_sym.T))`, chọn $k$ vector riêng ứng với $k$ trị riêng nhỏ nhất.\n3. Chuẩn hóa L2 từng hàng của ma trận `(n, k)` thu được và trả về `U`.',
          hint: 'Tính `deg = W.sum(axis=1)` rồi `d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)`. Nhân phát sóng `L_sym = np.eye(n) - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]`. Chuẩn hóa từng hàng bằng `V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_W_spec = np.full((8, 8), 0.01)
_W_spec[:4, :4] = 0.9
_W_spec[4:, 4:] = 0.9
np.fill_diagonal(_W_spec, 0.0)
`,
          starter: String.raw`def normalized_laplacian_embedding(W, k):
    raise NotImplementedError
`,
          solution: String.raw`def normalized_laplacian_embedding(W, k):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    deg = W.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L_sym = 0.5 * (L_sym + L_sym.T)
    _, vecs = np.linalg.eigh(L_sym)
    V = vecs[:, :k]
    norms = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return V / norms
`,
          test: String.raw`def _run_checks():
    c1 = _check('Shape & chuan hoa tung hang L2 = 1', lambda: (
        lambda U: (
            U.shape == (8, 2) and np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-6),
            f'shape={U.shape}, max_norm_err={np.max(np.abs(np.linalg.norm(U, axis=1) - 1.0)):.2e}'
        )
    )(normalized_laplacian_embedding(_W_spec, 2)))
    c2 = _check('Cung cum gan nhau & Khac cum truc giao', lambda: (
        lambda U: (
            abs(float(U[0] @ U[1])) > 0.98 and abs(float(U[0] @ U[4])) < 0.08,
            f'cos_same={float(U[0] @ U[1]):.4f}, cos_diff={float(U[0] @ U[4]):.4f}'
        )
    )(normalized_laplacian_embedding(_W_spec, 2)))
    _report('m6-u1-spectral', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao trong Spectral Clustering, ma trận $L_{\\text{sym}}$ có tổng hàng bằng 0 đối với Unnormalized $L$, còn $L_{\\text{sym}}$ lại có vector riêng ứng với $\\lambda_1 = 0$ là $D^{1/2}\\mathbf{1}$ (không hằng số)? Điều đó giải thích vì sao cần bước chuẩn hóa hàng Ng-Jordan-Weiss?',
          guidance:
            'Kiểm tra $L_{\\text{sym}}(D^{1/2}\\mathbf{1}) = D^{1/2}\\mathbf{1} - D^{-1/2}W\\mathbf{1} = D^{1/2}\\mathbf{1} - D^{-1/2}D\\mathbf{1} = 0$. Các phần tử vector riêng bị tỷ lệ theo $\\sqrt{d_i}$ của từng đỉnh, nên phép chiếu lên mặt cầu đơn vị theo từng hàng mới quy tụ mọi nút trong cùng thành phần liên thông về đúng một điểm.',
        },
        {
          question:
            'Nếu $W$ là ma trận tương đồng RBF của $N = 100{,}000$ điểm, chi phí bộ nhớ và thời gian của `eigh` là bao nhiêu, và có thể xấp xỉ hóa bằng kỹ thuật nào?',
          guidance:
            'Ma trận dày $W$ tốn $O(N^2)$ bộ nhớ (~80 GB float64) và $O(N^3)$ cho phân rã trị riêng. Trong thực tế ta dùng đồ thị $k$-NN thưa với solver Lanczos (`eigsh`), phương pháp Nyström với $m \\ll N$ mốc mẫu, hoặc Power Iteration Clustering.',
        },
      ],
    },
    {
      id: 'm6-u2',
      moduleId: 6,
      order: 2,
      title: 'Ước lượng mật độ (KDE) & Gaussian Mixture Models (GMM)',
      summary:
        'Chuyển từ phân cụm cứng sang mô hình hóa phân phối sinh không nhãn: ước lượng mật độ phi tham số Kernel Density Estimation (KDE) với tham số băng thông $h$, mô hình hỗn hợp Gaussian (GMM) với biến tiềm ẩn phân loại và ổn định số học bằng Log-Sum-Exp.',
      outcomes: [
        'Phân tích đánh đổi Bias–Variance của tham số băng thông $h$ trong KDE.',
        'Viết hàm log-likelihood biên của GMM bằng tích hợp biến tiềm ẩn rời rạc $z_i \\in \\{1,\\dots,K\\}$.',
        'Áp dụng biến đổi Log-Sum-Exp tránh tràn số dưới (underflow) khi tính trách nhiệm hậu nghiệm và log-likelihood.',
      ],
      references: [
        {
          source: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/',
          section: 'Lecture 05 (Density Estimation, Histograms, KDE & GMMs)',
          note: 'Ước lượng Parzen window 1D/ND và bài toán phân rã hỗn hợp Gaussian.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§3.5 & §21.4 (Gaussian Mixture Models & KDE)',
          note: 'Biểu diễn sinh $z_i \\sim \\text{Cat}(\\pi),\\; x_i \\mid z_i \\sim \\mathcal{N}(\\mu_{z_i}, \\Sigma_{z_i})$ và hiện tượng suy biến MLE.',
        },
      ],
      sections: [
        {
          heading: '1. Kernel Density Estimation (KDE) & Đánh đổi Bandwidth',
          body: String.raw`Histogram rời rạc phụ thuộc mạnh vào điểm gốc chia bin và không khả vi, đồng thời số bin tăng cấp số mũ $M^d$ theo số chiều. **KDE** thay thế bằng tổng chuẩn hóa các hàm nhân đối xứng tích phân bằng $1$ đặt tại từng điểm mẫu $\{x_i\}_{i=1}^N$:`,
          math: String.raw`\hat{p}_h(x) = \frac{1}{N} \sum_{i=1}^N \frac{1}{h^d}\, K\!\left(\frac{x - x_i}{h}\right), \qquad K_{\text{Gauss}}(u) = \frac{1}{(2\pi)^{d/2}} e^{-\|u\|_2^2 / 2}`,
          intuition:
            'Khi $h$ quá nhỏ, $\\hat{p}_h(x)$ xuất hiện gai nhọn tại mọi điểm mẫu (phương sai cao, overfit). Khi $h$ quá lớn, hai đỉnh phân phối bị hòa lẫn thành một bướu rộng (độ chệch Bias $\\propto h^2$ cao, underfit). Chuẩn chọn Silverman cân bằng phương sai tiệm cận $\\text{MISE}$.',
        },
        {
          heading: '2. Mô hình hỗn hợp Gaussian (GMM) & Phép sinh hai bước',
          body: String.raw`Thay vì lưu toàn bộ $N$ điểm như KDE, mô hình tham số **GMM** biểu diễn mật độ đa đỉnh bằng tổ hợp lồi của $K$ phân phối chuẩn với trọng số tiên nghiệm $\pi_k \ge 0,\; \sum_{k=1}^K \pi_k = 1$:`,
          math: String.raw`z_i \sim \text{Cat}(\pi), \quad x_i \mid (z_i = k) \sim \mathcal{N}(\mu_k, \Sigma_k) \implies p(x_i \mid \theta) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)`,
          pitfall:
            'Khác với đơn phân phối chuẩn có nghiệm MLE đóng $\\hat{\\mu}=\\bar{x}$, hàm log-likelihood biên của GMM là tổng bên trong log nên không lồi và không có nghiệm đóng: $\\ell(\\theta) = \\sum_{i=1}^N \\log \\sum_{k=1}^K \\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k)$. Hơn nữa, đặt $\\mu_1 = x_1$ và cho $\\sigma_1 \\to 0$ sẽ đẩy hàm likelihood tiến ra $+\\infty$ (điểm kỳ dị — singularity) nếu không có điều chuẩn $\\Sigma_k + \\epsilon I$ hoặc MAP prior.',
        },
        {
          heading: '3. Kỹ thuật Log-Sum-Exp chống tràn số Underflow',
          body: String.raw`Trong không gian nhiều chiều hoặc tại vùng đuôi xa tâm cụm, mật độ chuẩn $\mathcal{N}(x_i \mid \mu_k, \Sigma_k) \approx e^{-5000}$ sẽ làm tròn thành $0.0$ trong chuẩn IEEE-754 float64, kéo theo $\log(0) = -\infty$. Ta luôn làm việc trực tiếp trong không gian log với danh tính Log-Sum-Exp:`,
          math: String.raw`a_k = \log \pi_k + \log \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \log \sum_{k=1}^K e^{a_k} = m + \log \sum_{k=1}^K e^{a_k - m}, \quad m = \max_k a_k`,
        },
      ],
      simulationId: 'm6-u2',
      exercises: [
        {
          id: 'm6-u2-kde-gmm-ll',
          title: 'KDE Gaussian 1D & Log-Sum-Exp GMM Log-Likelihood',
          prompt:
            'Cài đặt hai hàm thuần NumPy:\n1. `kde_gaussian_1d(x_grid, samples, h)`: trả về mảng mật độ 1D `(len(x_grid),)` dùng nhân Gaussian với băng thông `h > 0`.\n2. `gmm_log_likelihood_1d(x, pis, mus, sigmas)`: trả về tổng log-likelihood vô hướng $\\sum_{i=1}^N \\log \\sum_{k=1}^K \\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\sigma_k^2)$. Bắt buộc dùng Log-Sum-Exp trong miền log để không bị `-inf` khi $|x_i - \\mu_k|/\\sigma_k$ lớn.',
          hint: 'Với `gmm_log_likelihood_1d`: tạo ma trận `log_w` shape `(N, K)` chứa `np.log(pis)[None, :] - 0.5*np.log(2*np.pi) - np.log(sigmas)[None, :] - 0.5*((x[:, None] - mus[None, :]) / sigmas[None, :])**2`, sau đó rút `m = np.max(log_w, axis=1, keepdims=True)` và tính `np.sum(m[:, 0] + np.log(np.sum(np.exp(log_w - m), axis=1)))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_samples_kde = np.array([-2.0, -1.5, -1.0, 2.0, 2.5, 3.0])
_grid_kde = np.linspace(-10.0, 10.0, 2001)
_pis_1d = np.array([0.4, 0.6])
_mus_1d = np.array([0.0, 10.0])
_sigmas_1d = np.array([1.0, 1.0])
`,
          starter: String.raw`def kde_gaussian_1d(x_grid, samples, h):
    raise NotImplementedError

def gmm_log_likelihood_1d(x, pis, mus, sigmas):
    raise NotImplementedError
`,
          solution: String.raw`def kde_gaussian_1d(x_grid, samples, h):
    x_grid = np.asarray(x_grid, dtype=float)
    samples = np.asarray(samples, dtype=float)
    diff = (x_grid[:, None] - samples[None, :]) / h
    kernels = np.exp(-0.5 * diff ** 2) / (np.sqrt(2.0 * np.pi) * h)
    return kernels.mean(axis=1)

def gmm_log_likelihood_1d(x, pis, mus, sigmas):
    x = np.asarray(x, dtype=float)
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    log_norm = -0.5 * np.log(2.0 * np.pi) - np.log(sigmas)
    z = (x[:, None] - mus[None, :]) / sigmas[None, :]
    log_joint = np.log(pis)[None, :] + log_norm[None, :] - 0.5 * (z ** 2)
    m = np.max(log_joint, axis=1, keepdims=True)
    log_p = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))
    return float(np.sum(log_p))
`,
          test: String.raw`def _run_checks():
    _area = lambda y, x: float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
    c1 = _check('KDE tich phan xap xi 1 & khop gia tri giai tich', lambda: (
        lambda dens: (
            dens.shape == _grid_kde.shape and abs(_area(dens, _grid_kde) - 1.0) < 1e-4,
            f'integral={_area(dens, _grid_kde):.6f}'
        )
    )(kde_gaussian_1d(_grid_kde, _samples_kde, 0.6)))
    _ref_ll0 = float(np.log(0.4 * (1/np.sqrt(2*np.pi)) + 0.6 * (1/np.sqrt(2*np.pi)) * np.exp(-50.0)))
    c2 = _check('GMM 1D log-likelihood dung tai diem test', lambda: (
        lambda ll: (
            abs(ll - _ref_ll0) < 1e-9,
            f'got={ll:.8f}, expected={_ref_ll0:.8f}'
        )
    )(gmm_log_likelihood_1d([0.0], _pis_1d, _mus_1d, _sigmas_1d)))
    c3 = _check('Log-Sum-Exp khong bi underflow khi x = 100', lambda: (
        lambda ll: (
            np.isfinite(ll) and abs(ll - (np.log(0.6) - 0.5*np.log(2*np.pi) - 0.5*(90.0**2))) < 1e-6,
            f'll_extreme={ll}'
        )
    )(gmm_log_likelihood_1d([100.0], _pis_1d, _mus_1d, _sigmas_1d)))
    _report('m6-u2-kde-gmm-ll', c1, c2, c3)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong thực tế, tại sao không dùng trực tiếp KDE với $d = 50$ chiều mặc dù KDE là công cụ ước lượng phi tham số nhất quán theo nghĩa tiệm cận?',
          guidance:
            'Tốc độ hội tụ MISE tối ưu của KDE giảm theo $O(N^{-4/(d+4)})$. Khi $d = 50$, số mũ chỉ còn $-4/54 \\approx -0.074$, đòi hỏi lượng mẫu khổng lồ tăng theo hàm mũ của $d$ để giữ sai số không đổi (Curse of Dimensionality).',
        },
        {
          question:
            'So sánh ý nghĩa của điểm kỳ dị $\\Sigma_k \\to 0$ trong GMM MLE với quá khớp thông thường. Làm sao phân biệt bằng tiêu chuẩn thông tin BIC/AIC hoặc Cross-Validation?',
          guidance:
            'Điểm kỳ dị tạo hàm mật độ vô hạn tại một mẫu duy nhất nhưng log-likelihood trên tập Validation lại sụp đổ về $-\\infty$ tại những điểm ngoài tâm đó. Đặt sàn phương sai $+\\epsilon I$ (hoặc Inverse-Wishart prior trong MAP) cùng BIC giúp chọn $K$ hợp lý.',
        },
      ],
    },
    {
      id: 'm6-u3',
      moduleId: 6,
      order: 3,
      title: 'Thuật toán Expectation-Maximization (EM) & Bất đẳng thức Jensen',
      summary:
        'Khung lý thuyết tổng quát để học MLE khi có biến tiềm ẩn: dẫn xuất chặn dưới ELBO từ Bất đẳng thức Jensen, chứng minh Log-Likelihood tăng không giảm sau mỗi vòng lặp EM và cài đặt EM vector hóa cho GMM nhiều chiều.',
      outcomes: [
        'Dẫn xuất bất đẳng thức ELBO từ tính lõm nghiêm ngặt của hàm log và Bất đẳng thức Jensen.',
        'Chứng minh hai bước E và M đảm bảo $\\ell(\\theta^{(t+1)}) \\ge \\ell(\\theta^{(t)})$ với mọi mô hình biến tiềm ẩn.',
        'Cài đặt bước E và bước M đầy đủ cho GMM nhiều chiều ổn định bằng `slogdet` và `linalg.solve`.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part IV — Ch. 9–10 (The EM Algorithm, Jensen’s Inequality & GMM)',
          note: 'Chứng minh chặt chẽ sự hội tụ của EM thông qua phân bố phụ $Q_i(z_i)$ và ELBO.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§8.7 & §21.4 (The EM Algorithm for Latent Variable Models)',
          note: 'Phân tích mối liên hệ $\\ell(\\theta) = \\text{ELBO}(q, \\theta) + \\text{KL}(q(z) \\parallel p(z \\mid x; \\theta))$.',
        },
      ],
      sections: [
        {
          heading: '1. ELBO & Bất đẳng thức Jensen',
          body: String.raw`Xét bài toán cực đại hóa log-likelihood biên $\ell(\theta) = \sum_{i=1}^N \log \sum_{z_i} p(x_i, z_i; \theta)$. Với bất kỳ họ phân bố thử tùy ý $q_i(z_i)$ sao cho $q_i(z_i) > 0$, áp dụng tính lõm của hàm $\log$ theo Bất đẳng thức Jensen ($\log \mathbb{E}_q[Y] \ge \mathbb{E}_q[\log Y]$):`,
          math: String.raw`\ell(\theta) = \sum_{i=1}^N \log \sum_{z_i} q_i(z_i) \frac{p(x_i, z_i; \theta)}{q_i(z_i)} \;\ge\; \sum_{i=1}^N \sum_{z_i} q_i(z_i) \log \frac{p(x_i, z_i; \theta)}{q_i(z_i)} \;=\; \text{ELBO}(q, \theta)`,
          intuition:
            'Khoảng cách chính xác giữa hai vế chính là tổng phân kỳ KL không âm: $\\ell(\\theta) - \\text{ELBO}(q, \\theta) = \\sum_i \\text{KL}(q_i(z_i) \\parallel p(z_i \\mid x_i; \\theta)) \\ge 0$. Do đó, làm khít chặn dưới tức là ép $q_i$ trùng đúng bằng phân bố hậu nghiệm.',
        },
        {
          heading: '2. Chứng minh EM không bao giờ làm giảm Log-Likelihood',
          body: String.raw`Thuật toán EM tối ưu luân phiên hai tọa độ trên cận dưới ELBO tại vòng lặp $t$:
- **E-step (siết chặt chặn dưới tại $\theta^{(t)}$)**: chọn $q_i^{(t)}(z_i) := p(z_i \mid x_i; \theta^{(t)})$, khiến $\text{KL} = 0$ và $\text{ELBO}(q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})$.
- **M-step (cực đại hóa chặn dưới theo $\theta$)**: cập nhật $\theta^{(t+1)} = \arg\max_\theta \text{ELBO}(q^{(t)}, \theta)$.`,
          math: String.raw`\ell(\theta^{(t+1)}) \;\ge\; \text{ELBO}\big(q^{(t)}, \theta^{(t+1)}\big) \;\ge\; \text{ELBO}\big(q^{(t)}, \theta^{(t)}\big) \;=\; \ell(\theta^{(t)})`,
          pitfall:
            'Dãy $\\ell(\\theta^{(t)})$ hội tụ đơn điệu tới một giá trị hữu hạn (nếu bị chặn trên), nhưng EM **không đảm bảo tìm ra cực đại toàn cục** mà có thể kẹt tại cực đại địa phương hoặc điểm yên ngựa tùy thuộc khởi tạo.',
        },
        {
          heading: '3. Công thức cập nhật E-step & M-step cho GMM nhiều chiều',
          body: String.raw`Đối với GMM nhiều chiều, tại bước E ta tính ma trận trách nhiệm hậu nghiệm $\gamma_{ik} = p(z_i = k \mid x_i; \theta^{(t)})$, trong bước M ta cập nhật bằng trung bình có trọng số với số điểm hiệu dụng $N_k = \sum_{i=1}^N \gamma_{ik}$:`,
          math: String.raw`\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}, \qquad \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i, \qquad \Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik}(x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^\top + \epsilon I`,
        },
      ],
      simulationId: 'm6-u3',
      exercises: [
        {
          id: 'm6-u3-gmm-em-step',
          title: 'GMM Expectation-Maximization (Multi-D Vectorized)',
          prompt:
            'Cài đặt hàm `gmm_em_step(X, pis, mus, covs, reg=1e-6)` thực hiện đúng 1 vòng E-step và M-step cho GMM chiều $d$:\n- `X` shape `(N, d)`, `pis` `(K,)`, `mus` `(K, d)`, `covs` `(K, d, d)`.\n- E-step: dùng `np.linalg.slogdet` và `np.linalg.solve` tính log-mật độ từng thành phần và dùng Log-Sum-Exp tính `ll_before` (tổng log-likelihood trên `X` trước cập nhật) cùng ma trận trách nhiệm `gamma` shape `(N, K)`.\n- M-step: tính `pis_new`, `mus_new`, `covs_new` (cộng `+ reg * np.eye(d)` giữ khả nghịch và đối xứng PSD).\n- Trả về `(pis_new, mus_new, covs_new, gamma, ll_before)`.',
          hint: 'Với từng cụm `k`: `diff = X - mus[k]` `(N, d)`, `sol = np.linalg.solve(covs[k], diff.T)` `(d, N)`, dạng toàn phương là `quad = np.sum(diff.T * sol, axis=0)`. Tổng hiệp phương sai cập nhật: `((gamma[:, k][:, None] * diff_new).T @ diff_new) / Nk[k] + reg * np.eye(d)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_em = np.random.default_rng(12)
_X_em = np.vstack([
    _rng_em.multivariate_normal([-2.5, -1.0], [[0.6, 0.2], [0.2, 0.5]], size=25),
    _rng_em.multivariate_normal([2.5, 1.5], [[0.7, -0.3], [-0.3, 0.6]], size=25),
])
_pis_em0 = np.array([0.5, 0.5])
_mus_em0 = np.array([[-0.5, -0.5], [0.5, 0.5]])
_covs_em0 = np.array([np.eye(2), np.eye(2)])
`,
          starter: String.raw`def gmm_em_step(X, pis, mus, covs, reg=1e-6):
    raise NotImplementedError
`,
          solution: String.raw`def gmm_em_step(X, pis, mus, covs, reg=1e-6):
    X = np.asarray(X, dtype=float)
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    covs = np.asarray(covs, dtype=float)
    N, d = X.shape
    K = pis.shape[0]
    log_joint = np.empty((N, K), dtype=float)
    for k in range(K):
        sign, logdet = np.linalg.slogdet(covs[k])
        if sign <= 0:
            raise ValueError('Ma tran hiep phuong sai khong xac dinh duong.')
        diff = X - mus[k]
        sol = np.linalg.solve(covs[k], diff.T)
        quad = np.sum(diff.T * sol, axis=0)
        log_joint[:, k] = np.log(pis[k]) - 0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

    m = np.max(log_joint, axis=1, keepdims=True)
    log_marg = m[:, 0] + np.log(np.sum(np.exp(log_joint - m), axis=1))
    ll_before = float(np.sum(log_marg))
    gamma = np.exp(log_joint - log_marg[:, None])

    Nk = np.maximum(gamma.sum(axis=0), 1e-12)
    pis_new = Nk / N
    mus_new = np.empty_like(mus)
    covs_new = np.empty_like(covs)
    for k in range(K):
        mus_new[k] = (gamma[:, k][:, None] * X).sum(axis=0) / Nk[k]
        diff_new = X - mus_new[k]
        cov_k = (diff_new.T @ (gamma[:, k][:, None] * diff_new)) / Nk[k] + reg * np.eye(d)
        covs_new[k] = 0.5 * (cov_k + cov_k.T)
    return pis_new, mus_new, covs_new, gamma, ll_before
`,
          test: String.raw`def _run_checks():
    c1 = _check('Responsibilities & tham so hop le sau 1 buoc EM', lambda: (
        lambda pis1, mus1, covs1, gamma1, ll0: (
            gamma1.shape == (50, 2)
            and np.allclose(gamma1.sum(axis=1), 1.0, atol=1e-7)
            and np.allclose(pis1.sum(), 1.0, atol=1e-7)
            and np.all(np.linalg.eigvalsh(covs1[0]) > 0)
            and np.all(np.linalg.eigvalsh(covs1[1]) > 0),
            f'pis={pis1.round(3)}, ll0={ll0:.3f}'
        )
    )(*gmm_em_step(_X_em, _pis_em0, _mus_em0, _covs_em0)))

    def _run_5():
        p, mu, cv = _pis_em0.copy(), _mus_em0.copy(), _covs_em0.copy()
        lls = []
        for _ in range(6):
            p, mu, cv, _, ll = gmm_em_step(_X_em, p, mu, cv)
            lls.append(ll)
        non_dec = all(lls[i + 1] >= lls[i] - 1e-9 for i in range(len(lls) - 1))
        means_sorted = np.sort(mu[:, 0])
        return (
            non_dec and lls[-1] > lls[0] and abs(means_sorted[0] + 2.5) < 0.5 and abs(means_sorted[1] - 2.5) < 0.5,
            f'lls={[round(v, 2) for v in lls]}, mu_x={means_sorted.round(2)}'
        )
    c2 = _check('Jensen: Log-Likelihood tang khong giam & hoi tu ve dung tam cum', _run_5)
    _report('m6-u3-gmm-em-step', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'K-means có thể xem là trường hợp suy biến nào của thuật toán EM cho GMM?',
          guidance:
            'Khi cố định $\\Sigma_k = \\epsilon I$ và cho $\\epsilon \\to 0^+$ (với $\\pi_k$ cân bằng), trách nhiệm hậu nghiệm $\\gamma_{ik} \\propto \\exp(-\\|x_i - \\mu_k\\|^2 / 2\\epsilon)$ tập trung thành vector one-hot tại tâm cụm gần nhất (E-step $\\to$ Hard Assignment), còn bước M quy về lấy trung bình cụm.',
        },
        {
          question:
            'Trong EM cho GMM, nếu bước M không cập nhật $\\mu_k$ và $\\Sigma_k$ đồng thời mà dùng $\\mu_k^{(t)}$ khi tính $\\Sigma_k^{(t+1)}$ thì định lý không giảm $\\ell(\\theta)$ có còn đúng không?',
          guidance:
            'Vẫn không giảm theo định lý **Generalized EM (GEM)**: bất kỳ cập nhật $\\theta^{(t+1)}$ nào làm tăng $\\text{ELBO}(q^{(t)}, \\theta) \\ge \\text{ELBO}(q^{(t)}, \\theta^{(t)})$ (dù chưa đạt cực đại tuyệt đối của bước M) đều vẫn bảo toàn $\\ell(\\theta^{(t+1)}) \\ge \\ell(\\theta^{(t)})$.',
        },
      ],
    },
    {
      id: 'm6-u4',
      moduleId: 6,
      order: 4,
      title: 'Giảm chiều: PCA (SVD), Kernel PCA, ICA & t-SNE/UMAP',
      summary:
        'Giảm chiều dữ liệu và biểu diễn ẩn tuyến tính/phi tuyến: hai cách chứng minh tương đương của PCA qua phương sai cực đại và sai số tái tạo tối thiểu theo định lý Eckart–Young, tính ổn định bằng SVD, mở rộng phi tuyến Kernel PCA, tách nguồn độc lập ICA và t-SNE/UMAP.',
      outcomes: [
        'Chứng minh hai bài toán tối ưu PCA (Max Variance vs Min Reconstruction Error) cùng cho các vector riêng trội của hiệp phương sai.',
        'Cài đặt PCA chính xác số học bằng SVD trên ma trận đã trừ trung bình mà không bình phương hóa điều kiện số $\\kappa(X)$.',
        'Phân biệt giả định của PCA (trực giao, bậc hai), ICA (độc lập thống kê, phi Gauss) và t-SNE/UMAP (bảo toàn đồ thị lân cận phi tuyến).',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part IV — Ch. 11–12 (Principal Components Analysis & Independent Components Analysis)',
          note: 'Dẫn xuất tối ưu lồi trên mặt cầu đơn vị, SVD và hàm mục tiêu InfoMax / ICA.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§20.1–20.4 (PCA, Kernel PCA, Factor Analysis & Manifold Learning / t-SNE)',
          note: 'Liên hệ giữa PCA, Probabilistic PCA và bảo toàn khoảng cách lân cận phi tuyến.',
        },
      ],
      sections: [
        {
          heading: '1. Hai góc nhìn tương đương của PCA',
          body: String.raw`Sau khi tâm hóa dữ liệu $\tilde{X} = X - \mathbf{1}\bar{x}^\top \in \mathbb{R}^{N \times d}$ (với ma trận hiệp phương sai mẫu $S = \frac{1}{N-1}\tilde{X}^\top \tilde{X}$), trục chính thứ nhất $u_1$ đồng thời là nghiệm của hai bài toán tương đương nhau:
1. **Cực đại phương sai chiếu**: $\max_{\|u\|_2=1} u^\top S u \implies S u_1 = \lambda_1 u_1$.
2. **Tối thiểu sai số tái tạo hạng $k$**: tìm ma trận trực chuẩn $U_k \in \mathbb{R}^{d \times k}$ ($U_k^\top U_k = I_k$) cực tiểu hóa $\|\tilde{X} - \tilde{X} U_k U_k^\top\|_F^2$.`,
          math: String.raw`\min_{U_k^\top U_k = I_k} \|\tilde{X} - \tilde{X} U_k U_k^\top\|_F^2 = \sum_{j=k+1}^d \sigma_j^2, \qquad \text{Tỷ lệ phương sai giữ lại} = \frac{\sum_{j=1}^k \sigma_j^2}{\sum_{j=1}^r \sigma_j^2}`,
          intuition:
            'Bằng định lý Pythagoras từng điểm mẫu, $\\|\\tilde{x}_i\\|_2^2 = \\|U_k^\\top \\tilde{x}_i\\|_2^2 + \\|\\tilde{x}_i - U_k U_k^\\top \\tilde{x}_i\\|_2^2$. Do tổng bình phương chuẩn mẫu cố định, tối đa hóa năng lượng hình chiếu bắt buộc tối thiểu hóa bình phương sai số trực giao còn dư.',
        },
        {
          heading: '2. Tính PCA bằng SVD & Lợi thế điều kiện số',
          body:
            String.raw`Phân rã giá trị kỳ dị rút gọn $\tilde{X} = U \Sigma V^\top$ cho trực tiếp các hướng chính là $k$ hàng đầu của $V^\top$ (` +
            '`components = Vt[:k]`' +
            String.raw`) và tọa độ ẩn $Z = \tilde{X} V_k = U_k \Sigma_k$. Làm việc thẳng trên $\tilde{X}$ giữ nguyên số điều kiện $\kappa(\tilde{X})$ thay vì bình phương thành $\kappa(\tilde{X})^2$ như khi tạo tường minh $\tilde{X}^\top \tilde{X}$.`,
          math: String.raw`\tilde{X} = U \Sigma V^\top \implies Z = \tilde{X} V_k, \qquad \hat{X}_{\text{recon}} = \mathbf{1}\bar{x}^\top + Z V_k^\top`,
          pitfall:
            'Quên trừ trung bình $\\bar{x}$ (Centering) trước khi chạy SVD khiến thành phần đầu tiên bị kéo về hướng vector trung bình của dữ liệu thay vì hướng biến thiên lớn nhất. Ngoài ra, luôn fit vector $\\bar{x}$ và $V_k$ duy nhất trên tập Train.',
        },
        {
          heading: '3. Kernel PCA, ICA & Trực quan hóa phi tuyến (t-SNE / UMAP)',
          body: String.raw`- **Kernel PCA**: tâm hóa ma trận Gram $\tilde{K} = H K H$ (với $H = I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top$) và tìm vector riêng trong không gian RKHS để bẻ thẳng các đa tạp phi tuyến.
- **ICA (Independent Component Analysis)**: thay vì chỉ triệt tiêu tương quan bậc 2 ($U^\top S U$ chéo) vốn bất định với phép xoay trực giao trên dữ liệu trắng, ICA giả định tín hiệu nguồn $s_j$ độc lập thống kê và **phi Gauss** (Cocktail Party problem), cực đại hóa entropy/non-Gaussianity.
- **t-SNE & UMAP**: ánh xạ từ phân phối xác suất lân cận trong $\mathbb{R}^d$ sang phân phối Student-t $1$ bậc tự do đuôi nặng trong $\mathbb{R}^2$ nhằm khắc phục **Crowding Problem** của không gian chiều cao.`,
          pitfall:
            'Không bao giờ diễn giải kích thước tuyệt đối của cụm hay khoảng cách giữa hai cụm xa nhau trên biểu đồ t-SNE như thước đo hình học Euclid thực, bởi perplexity làm co giãn mật độ cục bộ và hàm KL chỉ ưu tiên giữ láng giềng gần.',
        },
      ],
      simulationId: 'm6-u4',
      exercises: [
        {
          id: 'm6-u4-pca-svd',
          title: 'PCA qua SVD & Chứng thực Định lý Eckart–Young',
          prompt:
            'Cài đặt `pca_svd(X, k)` nhận ma trận `X` shape `(n, d)` và số chiều giữ lại `k` (`1 <= k <= min(n, d)`):\n1. Tâm hóa dữ liệu: `mean = X.mean(axis=0)`, `Xc = X - mean`.\n2. Chạy `U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`.\n3. Trả về 4 giá trị `(Z, components, explained_variance_ratio, X_recon)` với `components = Vt[:k]` `(k, d)`, `Z = Xc @ components.T` `(n, k)`, `explained_variance_ratio = (S[:k] ** 2) / np.sum(S ** 2)` `(k,)`, và `X_recon = mean + Z @ components` `(n, d)`.',
          hint: 'Chú ý không dùng `np.linalg.eigh(Xc.T @ Xc)` mà gọi trực tiếp `np.linalg.svd(Xc, full_matrices=False)`. Nếu tổng `S ** 2` bằng `0`, tránh chia cho `0`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_pca = np.random.default_rng(19)
_t_pca = _rng_pca.normal(0.0, 2.5, size=(40, 2))
_W_mix = np.array([[1.2, -0.7, 0.9, 0.4], [-0.5, 1.4, 0.2, 1.1]])
_X_pca = _t_pca @ _W_mix + np.array([10.0, -5.0, 3.0, 1.0]) + _rng_pca.normal(0.0, 0.04, size=(40, 4))
`,
          starter: String.raw`def pca_svd(X, k):
    raise NotImplementedError
`,
          solution: String.raw`def pca_svd(X, k):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    Z = Xc @ components.T
    total_energy = float(np.sum(S ** 2))
    if total_energy > 0.0:
        explained_variance_ratio = (S[:k] ** 2) / total_energy
    else:
        explained_variance_ratio = np.zeros(k, dtype=float)
    X_recon = mean + Z @ components
    return Z, components, explained_variance_ratio, X_recon
`,
          test: String.raw`def _run_checks():
    _, S_ref, _ = np.linalg.svd(_X_pca - _X_pca.mean(axis=0), full_matrices=False)
    expected_err = float(np.sum(S_ref[2:] ** 2))
    c1 = _check('Shape & tinh truc chuan cua cac thanh phan chinh', lambda: (
        lambda Z, comp, ratio, Xr: (
            Z.shape == (40, 2) and comp.shape == (2, 4) and ratio.shape == (2,) and Xr.shape == (40, 4)
            and np.allclose(comp @ comp.T, np.eye(2), atol=1e-7),
            f'shapes={(Z.shape, comp.shape, ratio.shape)}, ortho_err={np.max(np.abs(comp @ comp.T - np.eye(2))):.2e}'
        )
    )(*pca_svd(_X_pca, 2)))
    c2 = _check('Eckart-Young: sai so tai tao Frobenius khop tong gia tri ky di bi bo', lambda: (
        lambda Z, comp, ratio, Xr: (
            abs(float(np.sum((_X_pca - Xr) ** 2)) - expected_err) < 1e-6 and float(ratio.sum()) > 0.995,
            f'recon_err={np.sum((_X_pca - Xr)**2):.6f} vs expected={expected_err:.6f}, var_kept={ratio.sum():.5f}'
        )
    )(*pca_svd(_X_pca, 2)))
    _report('m6-u4-pca-svd', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Khi số mẫu $N = 100$ nhưng số chiều đặc trưng $d = 20{,}000$ (như dữ liệu biểu hiện gen hoặc ảnh flatten), nên tính PCA bằng cách phân rã hiệp phương sai $d \\times d$ hay dùng ma trận Gram $N \\times N$ / SVD rút gọn?',
          guidance:
            'Ma trận $S = \\frac{1}{N-1}\\tilde{X}^\\top\\tilde{X}$ có kích thước $20{,}000 \\times 20{,}000$ cực kỳ tốn bộ nhớ ($> 3$ GB) và chậm $O(d^3)$. Vì $\\tilde{X}\\tilde{X}^\\top$ ($100 \\times 100$) chia sẻ cùng tập trị riêng khác 0 với $\\tilde{X}^\\top\\tilde{X}$, gọi `svd(Xc, full_matrices=False)` hoặc tìm vector riêng $u_j$ của $\\tilde{X}\\tilde{X}^\\top$ rồi suy ra $v_j = \\tilde{X}^\\top u_j / \\sigma_j$ chỉ tốn $O(N^2 d)$.',
        },
        {
          question:
            'Tại sao ICA không thể tách hai tín hiệu nguồn có cùng phân phối chuẩn Gaussian độc lập nhau?',
          guidance:
            'Nếu $s \\sim \\mathcal{N}(0, I)$, phép xoay trực giao bất kỳ $Q$ tạo ra $s\' = Q s$ vẫn có phân phối $\\mathcal{N}(0, Q Q^\\top) = \\mathcal{N}(0, I)$ hoàn toàn không phân biệt được (rotational symmetry). Phi Gauss mới phá vỡ đối xứng xoay này để xác định duy nhất ma trận trộn $A$ (xem CS229 Ch. 12).',
        },
      ],
    },
  ],
};
