import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_05: GradModule = {
  id: 5,
  title: 'Phương pháp Phi tham số & Học Tổ hợp',
  englishTitle: 'Nonparametric Models & Ensemble Learning',
  color: '#a8a7ff',
  status: 'ready',
  overview:
    'Các mô hình phi tham số và tổ hợp cây quyết định mạnh nhất cho dữ liệu dạng bảng: K-Nearest Neighbors dưới lời nguyền chiều cao (Curse of Dimensionality) và khoảng cách Mahalanobis; cây quyết định CART (Gini, Entropy, tỉa nhánh Cost-Complexity Pruning); Bagging, Out-of-Bag Error & Random Forests; cùng các thuật toán Boosting tuần tự (AdaBoost, Gradient Boosting và XGBoost bậc hai).',
  outcomes: [
    'Phân tích sự tập trung thể tích vỏ siêu cầu và suy giảm tương phản khoảng cách khi số chiều $d \\to \\infty$, cài đặt KNN vector hóa và khoảng cách Mahalanobis.',
    'Cài đặt bộ tách nút tham lam tối ưu cho cây quyết định CART theo Gini Impurity và giải thích tỉa nhánh $R_\\alpha(T)$.',
    'Chứng minh định lý giảm phương sai của tập hợp tương quan $\\text{Var}(\\bar{f}) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ và ước lượng Out-of-Bag.',
    'Cài đặt bước cập nhật trọng số AdaBoost và công thức trọng số lá tối ưu kèm Split Gain bậc hai của XGBoost.',
  ],
  sources: [
    'ISLR 7th Printing — Ch. 8 (Tree-Based Methods, Bagging, Random Forests, Boosting)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 16 & Ch. 18)',
  ],
  units: [
    {
      id: 'm5-u1',
      moduleId: 5,
      order: 1,
      title: 'KNN, Metric Learning & Curse of Dimensionality',
      summary:
        'Học phi tham số dựa trên bộ nhớ (Instance-based Learning): hội tụ không đồng nhất của K-Nearest Neighbors, sự sụp đổ của trực giác hình học Euclid trong không gian nhiều chiều (Curse of Dimensionality) và chuẩn hóa hiệp phương sai qua khoảng cách Mahalanobis.',
      outcomes: [
        'Chứng minh bằng giải tích lý do phần trăm thể tích quả cầu $d$ chiều tập trung gần như 100% ở lớp vỏ ngoài mỏng $\\epsilon$ khi $d$ lớn.',
        'Tính tỉ số $(d_{\\max} - d_{\\min})/d_{\\min} \\to 0$ làm mất khái niệm "láng giềng gần nhất" nếu nhiều chiều không mang thông tin.',
        'Cài đặt KNN vector hóa và khoảng cách Mahalanobis $d_M(x, z)^2 = (x - z)^\\top \\Sigma^{-1} (x - z)$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§16.1 (k-Nearest Neighbors, Distance Metrics & Curse of Dimensionality)',
          note: 'Hiện tượng tập trung khối lượng ở vỏ siêu cầu và Metric Learning.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§2.2.3 & §4.7.6 (K-Nearest Neighbors)',
          note: 'Đánh đổi Bias–Variance khi thay đổi K và giới hạn ở số chiều cao.',
        },
      ],
      sections: [
        {
          heading: '1. Lời nguyền chiều cao (Curse of Dimensionality)',
          body: String.raw`Xét hình lập phương đơn vị $[0, 1]^d$ trong không gian $\mathbb{R}^d$. Để bao phủ được tỷ lệ $p = 10\%$ thể tích tổng thể bằng một khối siêu lập phương nhỏ bao quanh điểm truy vấn, cạnh mỗi chiều bắt buộc phải dài $l = p^{1/d} = 0.1^{1/d}$: với $d = 10$, cạnh cần đạt $l \approx 0.794$; với $d = 100$, cạnh phải đạt $0.977$ (bao trùm gần hết mọi đặc trưng!).

Tương tự với siêu cầu đơn vị bán kính $1$: tỷ lệ thể tích nằm ngoài lớp lõi bán kính $1 - \varepsilon$ là $1 - (1 - \varepsilon)^d \xrightarrow{d \to \infty} 1$. Khi $d$ lớn, mọi điểm dữ liệu đều bị đẩy ra rìa biên, dẫn đến hai hệ quả nghiêm trọng cho KNN:`,
          math: String.raw`\frac{d_{\max}(x) - d_{\min}(x)}{d_{\min}(x)} \xrightarrow{p} 0 \quad (d \to \infty), \qquad \text{Khối lượng lớp vỏ ngoài } = 1 - (1 - \varepsilon)^d \to 1`,
          intuition:
            'Nếu có 1000 đặc trưng nhiễu cộng vào 2 đặc trưng thật, tổng khoảng cách bình phương sẽ bị nhiễu áp đảo, khiến láng giềng gần nhất theo Euclid không còn chia sẻ cùng nhãn.',
        },
        {
          heading: '2. Chuẩn hóa hiệp phương sai & Khoảng cách Mahalanobis',
          body: String.raw`Khoảng cách Euclid chuẩn $d_E(x, z) = \sqrt{(x - z)^\top I (x - z)}$ giả định mọi đặc trưng độc lập và có cùng phương sai đơn vị. Để loại bỏ lệch thang đo lẫn tương quan tuyến tính giữa các trục, **Khoảng cách Mahalanobis** đưa vào ma trận xác định dương $M \succeq 0$ (kinh điển chọn $M = \Sigma^{-1}$):`,
          math: String.raw`d_{\text{Mahal}}(x, z)^2 = (x - z)^\top \Sigma^{-1} (x - z) = \big\|\Sigma^{-1/2}(x - z)\big\|_2^2`,
          pitfall:
            'Tính $k$ láng giềng gần nhất bằng khoảng cách Euclid trên dữ liệu chưa chuẩn hóa sẽ để đặc trưng có biên độ số học lớn nhất (ví dụ Thu nhập tính bằng VNĐ so với Tuổi) chi phối hoàn toàn kết quả phân loại.',
        },
      ],
      simulationId: 'm5-u1',
      exercises: [
        {
          id: 'm5-u1-knn-mahalanobis',
          title: 'Vectorized KNN Classifier & Khoảng cách Mahalanobis',
          prompt:
            'Cài đặt hai hàm:\n1. `knn_predict(X_train, y_train, X_test, k=3)`: tính ma trận khoảng cách bình phương `(n_test, n_train)` bằng broadcasting, lấy `k` láng giềng gần nhất và bầu chọn đa số (ưu tiên nhãn số nhỏ nhất nếu hòa).\n2. `mahalanobis_sq_dist(x, z, cov_inv)`: tính khoảng cách bình phương Mahalanobis $(x - z)^\\top \\Sigma^{-1} (x - z)$.',
          hint: 'Trong `knn_predict`: tính `sq = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)`, lấy `idx = np.argsort(sq, axis=1, kind="mergesort")[:, :k]`, rồi dùng `np.bincount(y_train[row]).argmax()` cho từng dòng để tự động ưu tiên nhãn nhỏ nhất khi hòa.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xknn_tr = np.array([[0., 0.], [0.2, 0.1], [-0.1, 0.2], [5., 5.], [5.2, 4.9], [4.9, 5.1]])
_yknn_tr = np.array([0, 0, 0, 1, 1, 1])
_Xknn_te = np.array([[0.1, 0.1], [5.0, 5.0], [10.0, 0.0]])
_S_cov = np.array([[4.0, 0.0], [0.0, 0.25]])
_S_inv = np.linalg.inv(_S_cov)
`,
          starter: String.raw`def knn_predict(X_train, y_train, X_test, k=3):
    raise NotImplementedError

def mahalanobis_sq_dist(x, z, cov_inv):
    raise NotImplementedError
`,
          solution: String.raw`def knn_predict(X_train, y_train, X_test, k=3):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    sq_dists = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    nn_idx = np.argsort(sq_dists, axis=1, kind='mergesort')[:, :k]
    preds = [int(np.bincount(y_train[row]).argmax()) for row in nn_idx]
    return np.array(preds, dtype=int)

def mahalanobis_sq_dist(x, z, cov_inv):
    diff = np.asarray(x, dtype=float) - np.asarray(z, dtype=float)
    cov_inv = np.asarray(cov_inv, dtype=float)
    return float(diff @ cov_inv @ diff)
`,
          test: String.raw`_report('m5-u1-knn-mahalanobis',
    _check('knn_predict phân loại chính xác các cụm tâm [0,0] và [5,5]', lambda: (
        np.array_equal(knn_predict(_Xknn_tr, _yknn_tr, _Xknn_te[:2], k=3), [0, 1]),
        'Kiểm tra k=3 láng giềng gần nhất.'
    )),
    _check('Mahalanobis chuẩn hóa theo phương sai: Δ=(2, 0) với var=4 tương đương Δ=(0, 0.5) với var=0.25', lambda: (
        np.isclose(mahalanobis_sq_dist([2., 0.], [0., 0.], _S_inv), 1.0)
        and np.isclose(mahalanobis_sq_dist([0., 0.5], [0., 0.], _S_inv), 1.0),
        'Khoảng cách theo thang đo độ lệch chuẩn σ đều bằng 1².'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong thực tế tìm kiếm láng giềng gần nhất trên tập $n = 10^7$ vector embedding (hệ thống RAG), tại sao cấu trúc cây không gian KD-Tree vẫn nhanh vượt trội ở số chiều $d = 3$ nhưng suy biến ngang vét cạn $O(n)$ khi $d > 20$, và giải pháp công nghiệp nào thay thế nó?',
          guidance:
            'Để KD-Tree cắt tỉa nhánh hiệu quả, cầu bán kính láng giềng chỉ nên cắt ít hộp trục chia không gian; nhưng ở $d > 20$, theo lời nguyền chiều cao, hình cầu truy vấn cắt gần như toàn bộ $2^d$ ô lân cận khiến thuật toán phải duyệt hầu hết cây. Do đó tìm kiếm vector quy mô lớn dùng thuật toán Approximate Nearest Neighbor (ANN) như **HNSW (Hierarchical Navigable Small World)** hoặc **IVF-PQ**.',
        },
      ],
    },
    {
      id: 'm5-u2',
      moduleId: 5,
      order: 2,
      title: 'Decision Trees (CART, Gini/Entropy & Pruning)',
      summary:
        'Cây quyết định phi tuyến chia cắt không gian đặc trưng theo trục tọa độ: thuật toán tham lam CART cực tiểu hóa độ bất thuần (Gini Impurity, Shannon Entropy), tìm ngưỡng trung điểm tối ưu, kiểm soát độ phức tạp và tỉa nhánh Cost-Complexity Pruning $R_\\alpha(T)$.',
      outcomes: [
        'Tính chính xác Gini Impurity và Entropy, chứng minh cả hai hàm đều lõm chặt để độ giảm bất thuần khi tách nhánh luôn không âm.',
        'Cài đặt tìm kiếm tách nút tối ưu (best split) trên toàn bộ đặc trưng và tập ngưỡng trung điểm hợp lệ.',
        'Hiểu cơ chế tỉa nhánh Weakest-Link Cost-Complexity Pruning cân bằng lỗi lá và số lá $|T|$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§8.1 (The Basics of Decision Trees & Tree Pruning)',
          note: 'Định nghĩa Recursive Binary Splitting và Cost-Complexity Pruning.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§18.1 (Classification and Regression Trees — CART)',
          note: 'Hàm bất thuần Gini, Information Gain và đánh giá chia nhánh tham lam.',
        },
      ],
      sections: [
        {
          heading: '1. Tiêu chí tách nhánh tham lam: Gini Impurity & Information Gain',
          body: String.raw`Xét nút $m$ chứa $N_m$ mẫu với phân bố lớp $\hat{p}_{mk} = \frac{1}{N_m}\sum_{i \in R_m} \mathbb{I}(y_i = k)$. Hai thước đo độ bất thuần (impurity) phổ biến nhất là:
- **Gini Impurity**: $G(R_m) = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk}) = 1 - \sum_{k=1}^K \hat{p}_{mk}^2$ (xác suất phân loại nhầm nếu gán nhãn ngẫu nhiên theo $\hat{p}_m$).
- **Shannon Entropy**: $H(R_m) = -\sum_{k=1}^K \hat{p}_{mk}\log_2 \hat{p}_{mk}$.

Với mỗi đặc trưng $j$ và ngưỡng cắt $t$, ta chia tập mẫu thành nhánh trái $L = \{x_{ij} \le t\}$ và nhánh phải $R = \{x_{ij} > t\}$, chọn cặp $(j^*, t^*)$ tối đa hóa độ giảm bất thuần có trọng số:`,
          math: String.raw`\Delta I(j, t) = I(R_m) - \left(\frac{|L|}{N_m} I(L) + \frac{|R|}{N_m} I(R)\right) \ge 0`,
          intuition:
            'Nhờ tính chất hàm lõm (concave) của cả Gini lẫn Entropy, bất đẳng thức Jensen bảo đảm $\\Delta I(j, t)$ luôn $\\ge 0$ và chỉ bằng $0$ khi phân bố lớp của $L$ và $R$ y hệt nút cha.',
        },
        {
          heading: '2. Khống chế quá khớp bằng Cost-Complexity Pruning R_α(T)',
          body: String.raw`Nếu tiếp tục tách đến khi mọi lá thuần nhất ($N_m = 1$), cây sẽ học vẹt toàn bộ nhiễu (High Variance). Thay vì dừng sớm cận thị (có thể bỏ lỡ tách nhánh tốt ở độ sâu sau đó), CART nuôi cây lớn $T_0$ rồi tỉa dần các cây con yếu nhất bằng hàm mục tiêu điều chuẩn số lá $|T|$:`,
          math: String.raw`R_\alpha(T) = \sum_{m=1}^{|T|} N_m \, I(R_m) + \alpha |T| \qquad (\alpha \ge 0 \text{ chọn bằng Cross-Validation})`,
          pitfall:
            'Cây quyết định tiêu chuẩn chỉ tạo siêu phẳng **song song trục tọa độ**, nên để xấp xỉ ranh giới chéo $x_1 + x_2 > 0$, cây đơn lẻ cần nhiều tầng bậc thang gãy khúc.',
        },
      ],
      simulationId: 'm5-u2',
      exercises: [
        {
          id: 'm5-u2-best-split',
          title: 'Hàm Gini Impurity & Bộ tách nút tối ưu CART',
          prompt:
            'Cài đặt hai hàm:\n1. `gini_impurity(y)`: trả về $1 - \\sum_k p_k^2$ (`0.0` nếu `len(y) == 0`).\n2. `best_stump_split(X, y)`: duyệt qua mọi đặc trưng `j` và mọi ngưỡng trung điểm giữa hai giá trị phân biệt liên tiếp sau khi sắp xếp, tìm chia nhánh `x_j <= t` vs `x_j > t` đạt độ giảm Gini có trọng số $\\Delta G$ lớn nhất. Trả về tuple `(best_feat, best_thresh, best_gain)` (ưu tiên chỉ số feature nhỏ hơn, ngưỡng nhỏ hơn nếu hòa).',
          hint: 'Với mỗi cột `j`: lấy `vals = np.unique(X[:, j])`; nếu `len(vals) < 2` thì bỏ qua cột này. Tính `mids = 0.5 * (vals[:-1] + vals[1:])`. Với mỗi `t` trong `mids`, tách `left = y[X[:, j] <= t]`, `right = y[X[:, j] > t]`, tính `gain = parent_gini - (len(left)*gini_impurity(left) + len(right)*gini_impurity(right)) / n`, cập nhật nếu `gain > best_gain + 1e-12`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Xdt = np.array([
    [1.0, 2.0],
    [1.0, 4.0],
    [1.0, 6.0],
    [2.0, 1.0],
    [2.0, 5.0],
    [2.0, 7.0],
    [3.0, 3.0],
    [3.0, 8.0],
])
_ydt = np.array([0, 0, 1, 0, 1, 1, 0, 1]) # Exact threshold on feature 1 at midpoint 4.5!
`,
          starter: String.raw`def gini_impurity(y):
    raise NotImplementedError

def best_stump_split(X, y):
    raise NotImplementedError
`,
          solution: String.raw`def gini_impurity(y):
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return float(1.0 - np.sum(probs ** 2))

def best_stump_split(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape
    parent = gini_impurity(y)
    best_feat = 0
    best_thresh = 0.0
    best_gain = -1.0
    for j in range(d):
        vals = np.unique(X[:, j])
        if len(vals) < 2:
            continue
        mids = 0.5 * (vals[:-1] + vals[1:])
        for t in mids:
            mask = X[:, j] <= t
            nL = int(np.sum(mask))
            nR = n - nL
            if nL == 0 or nR == 0:
                continue
            weighted = (nL * gini_impurity(y[mask]) + nR * gini_impurity(y[~mask])) / n
            gain = parent - weighted
            if gain > best_gain + 1e-12:
                best_gain = float(gain)
                best_feat = j
                best_thresh = float(t)
    return best_feat, best_thresh, max(0.0, best_gain)
`,
          test: String.raw`_report('m5-u2-best-split',
    _check('gini_impurity đạt 0.0 khi thuần khiết và 0.5 khi cân bằng hai lớp', lambda: (
        np.isclose(gini_impurity([1, 1, 1]), 0.0) and np.isclose(gini_impurity([0, 0, 1, 1]), 0.5) and np.isclose(gini_impurity([]), 0.0),
        'Kiểm tra giá trị chuẩn của hàm bất thuần.'
    )),
    _check('Tìm đúng đặc trưng thứ 1 (cột index=1), ngưỡng trung điểm 4.5 và độ giảm Gini = 0.5', lambda: (
        (lambda feat, th, gain: feat == 1 and np.isclose(th, 4.5) and np.isclose(gain, 0.5))(
            *best_stump_split(_Xdt, _ydt)
        ),
        'Phát hiện chính xác nhát cắt phân tách hoàn hảo hai lớp.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Nếu nhân giá trị một cột đặc trưng liên tục $x_j$ lên $1000$ lần hoặc biến đổi đơn điệu dương tùy ý $x_j \\mapsto \\exp(x_j)$, cấu trúc và dự đoán của Cây quyết định CART có thay đổi không? So sánh tính chất này với Ridge/SVM/KNN.',
          guidance:
            'Cây quyết định CART **hoàn toàn bất biến** với mọi phép biến đổi đơn điệu tăng của từng đặc trưng riêng lẻ, vì nhát cắt chỉ phụ thuộc vào **thứ tự xếp hạng** ($x_{ij} \\le t$), không phụ thuộc độ lớn khoảng cách! Ngược lại, SVM, Ridge và KNN biến động mạnh nếu không chuẩn hóa thang đo.',
        },
      ],
    },
    {
      id: 'm5-u3',
      moduleId: 5,
      order: 3,
      title: 'Bagging & Random Forests (Giảm phương sai & OOB)',
      summary:
        'Giảm phương sai không làm tăng mạnh thiên vị qua Học Tổ hợp song song: lấy mẫu hoàn lại Bootstrap, ước lượng không chệch Out-of-Bag (OOB ≈ 36.8% dữ liệu bị bỏ lại), phân tích toán học định lý tương quan cặp $\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ và kỹ thuật ngẫu nhiên hóa tập đặc trưng tại mỗi nút của Random Forest.',
      outcomes: [
        'Chứng minh xác suất một mẫu bất kỳ rơi vào tập Out-of-Bag là $(1 - 1/n)^n \\to e^{-1} \\approx 36.8\\%$.',
        'Tính phương sai trung bình của $B$ cây có tương quan cặp $\\rho$ và lý giải ngưỡng sàn không thể giảm thêm nếu không phá vỡ tương quan.',
        'Giải thích cơ chế chọn ngẫu nhiên $m \\approx \\sqrt{d}$ đặc trưng tại mỗi nút trong Random Forest giúp hạ $\\rho$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§8.2.1–8.2.2 (Bagging, Out-of-Bag Error & Random Forests)',
          note: 'Cơ chế Bootstrap, OOB validation miễn phí và chọn ngẫu nhiên m đặc trưng.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§18.2–18.3 (Bagging & Random Forests)',
          note: 'Phân tích giảm phương sai trên các bộ ước lượng tương quan.',
        },
      ],
      sections: [
        {
          heading: '1. Lấy mẫu Bootstrap & Xác suất Out-of-Bag (OOB ≈ 36.8%)',
          body: String.raw`Mỗi cây trong Bagging / Random Forest được huấn luyện trên một mẫu Bootstrap kích thước $n$ rút ngẫu nhiên **có hoàn lại** từ tập huấn luyện gốc. Xác suất một quan sát $i$ không được bốc trong bất kỳ lượt nào trong $n$ lượt là:

$$\lim_{n \to \infty} \left(1 - \frac{1}{n}\right)^n = e^{-1} \approx 0.3679$$

Các mẫu bị bỏ lại này tạo thành tập **Out-of-Bag (OOB)** của riêng cây đó. Nhờ gộp dự đoán của mỗi mẫu $i$ từ chỉ $\approx 36.8\% \cdot B$ cây không hề nhìn thấy $i$ khi huấn luyện, **OOB Error** cung cấp ước lượng lỗi tổng quát hóa tương đương K-Fold Cross-Validation hoàn toàn miễn phí!`,
        },
        {
          heading: '2. Định lý Phương sai Tổ hợp tương quan & Vai trò của Random Forest',
          body: String.raw`Giả sử ta lấy trung bình dự đoán của $B$ cây phân bố giống nhau, mỗi cây có phương sai $\sigma^2$ và hệ số tương quan cặp dương $\rho$ giữa hai cây bất kỳ. Phương sai của bộ dự đoán trung bình $\bar{f} = \frac{1}{B}\sum_{b=1}^B f_b$ bằng:`,
          math: String.raw`\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^B f_b\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2`,
          intuition:
            'Khi tăng số cây $B \\to \\infty$, số hạng thứ hai $\\frac{1-\\rho}{B}\\sigma^2 \\to 0$, **nhưng số hạng thứ nhất $\\rho \\sigma^2$ vẫn còn nguyên!** Trong Bagging thuần túy, nếu dữ liệu có một đặc trưng quá trội, mọi cây đều tách ngay tại nút gốc đó khiến tương quan $\\rho$ rất cao. **Random Forest** chỉ xét ngẫu nhiên $m \\approx \\sqrt{d}$ đặc trưng ở mỗi nút, chủ động triệt tiêu các đặc trưng trội ở một số cây để kéo giảm mạnh $\\rho$ và mở khóa giảm phương sai sâu hơn nữa.',
        },
      ],
      simulationId: 'm5-u3',
      exercises: [
        {
          id: 'm5-u3-oob-bagging',
          title: 'Lấy mẫu Bootstrap/OOB & Công thức Phương sai Tổ hợp',
          prompt:
            'Cài đặt hai hàm:\n1. `bootstrap_oob_indices(n, seed=0)`: dùng `rng = np.random.default_rng(seed)` bốc `in_bag = rng.integers(0, n, size=n)`, tìm mảng các chỉ số Out-of-Bag `oob` không xuất hiện trong `in_bag` (sắp xếp tăng dần), trả về `(in_bag, oob)`.\n2. `ensemble_variance(sigma2, rho, B)`: trả về $\\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2$.',
          hint: 'Dùng `rng = np.random.default_rng(seed)`, `in_bag = rng.integers(0, n, size=n)`, và `oob = np.setdiff1d(np.arange(n), in_bag)`. Với `ensemble_variance`, trả về `float(rho * sigma2 + ((1.0 - rho) / B) * sigma2)`.',
          setup: COMMON_SETUP,
          starter: String.raw`def bootstrap_oob_indices(n, seed=0):
    raise NotImplementedError

def ensemble_variance(sigma2, rho, B):
    raise NotImplementedError
`,
          solution: String.raw`def bootstrap_oob_indices(n, seed=0):
    rng = np.random.default_rng(seed)
    in_bag = rng.integers(0, n, size=n)
    oob = np.setdiff1d(np.arange(n), in_bag)
    return in_bag, oob

def ensemble_variance(sigma2, rho, B):
    return float(rho * sigma2 + ((1.0 - rho) / B) * sigma2)
`,
          test: String.raw`_report('m5-u3-oob-bagging',
    _check('Tỷ lệ OOB trên n=1000 xấp xỉ e⁻¹ ≈ 0.368, hợp của in_bag và oob phủ trọn 0..n-1 và rời nhau', lambda: (
        (lambda ib, oob: (
            len(ib) == 1000
            and 0.32 < len(oob) / 1000 < 0.42
            and len(set(ib) | set(oob)) == 1000
            and len(set(ib) & set(oob)) == 0
        ))(*bootstrap_oob_indices(1000, seed=42)),
        'Kiểm chứng giới hạn cổ điển (1 - 1/n)^n ≈ 36.8%.'
    )),
    _check('Phương sai tiến về sàn ρσ² khi B lớn và giảm mạnh khi hạ tương quan ρ', lambda: (
        np.isclose(ensemble_variance(10.0, 0.4, 10000), 4.0, atol=1e-2)
        and ensemble_variance(10.0, 0.2, 50) < ensemble_variance(10.0, 0.6, 50),
        'Chứng minh lợi thế cốt lõi của Random Forest so với Bagging.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao tăng số lượng cây $B$ trong Random Forest lên hàng nghìn cây hầu như **không bao giờ gây Overfitting**, trái ngược với việc tăng số cây trong Gradient Boosting ($M$)?',
          guidance:
            'Random Forest cộng **trung bình** các cây có phân bố đồng nhất: tăng $B$ chỉ triệt tiêu thành phần phương sai $\\frac{1-\\rho}{B}\\sigma^2$, giữ nguyên kỳ vọng mô hình và hội tụ về một giới hạn cố định theo Luật Số lớn. Ngược lại, Boosting là cộng **dồn tích lũy tuần tự** khớp phần dư; nếu chạy quá nhiều vòng $M$, các cây sau bắt đầu khớp cả nhiễu ngẫu nhiên trên tập train và làm tăng phương sai.',
        },
      ],
    },
    {
      id: 'm5-u4',
      moduleId: 5,
      order: 4,
      title: 'Boosting: AdaBoost & Gradient Boosting (GBDT / XGBoost)',
      summary:
        'Học tổ hợp tuần tự hướng giảm Bias: AdaBoost tái trọng số mẫu khó theo hàm mất mát Exponential Loss; Gradient Boosting (GBDT) thực hiện Gradient Descent trong không gian hàm qua pseudo-residuals; và công thức XGBoost khai triển Taylor bậc hai cho trọng số lá tối ưu và Gain cắt nhánh.',
      outcomes: [
        'Cài đặt quy tắc tính hệ số mô hình $\\alpha_m = \\frac{1}{2}\\ln\\frac{1 - \\varepsilon_m}{\\varepsilon_m}$ và tái trọng số mẫu của AdaBoost, hiểu vì sao Boosting hạ lỗi mẫu trọng số xuống 0.5 sau mỗi bước.',
        'Giải thích Gradient Boosting dưới góc nhìn Functional Gradient Descent (khớp $h_m(x)$ vào $-\\partial \\ell / \\partial F_{m-1}(x)$).',
        'Dẫn xuất và cài đặt công thức giải tích bậc hai của XGBoost cho trọng số lá tối ưu $w_j^*$ và điểm lợi ích cắt nhánh $\\text{Gain}$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§18.4–18.6 (AdaBoost, Gradient Boosting & XGBoost)',
          note: 'Cơ sở Exponential Loss của AdaBoost, Pseudo-residuals của GBDT và Second-order Taylor của XGBoost.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§8.2.3 (Boosting)',
          note: 'Các siêu tham số shrinkage η, độ sâu cây d và số vòng B.',
        },
      ],
      sections: [
        {
          heading: '1. AdaBoost & Tối ưu tham lam trên Exponential Loss',
          body: String.raw`**AdaBoost** huấn luyện tuần tự $M$ bộ phân loại yếu $G_m(x) \in \{-1, +1\}$ trên tập dữ liệu được gán trọng số $w_i^{(m)}$ (tổng bằng $1$), rồi kết hợp $\text{sign}(\sum_{m=1}^M \alpha_m G_m(x))$. Dưới góc nhìn thống kê (Friedman et al., 2000), AdaBoost chính là phép tối ưu Stagewise trên hàm mất mát mũ $L(y, F(x)) = \exp(-y F(x))$:`,
          math: String.raw`\varepsilon_m = \sum_{y_i \neq G_m(x_i)} w_i^{(m)}, \qquad \alpha_m = \frac{1}{2}\ln\!\left(\frac{1 - \varepsilon_m}{\varepsilon_m}\right), \qquad w_i^{(m+1)} = \frac{w_i^{(m)} \exp(-\alpha_m y_i G_m(x_i))}{Z_m}`,
          intuition:
            'Sau phép tái trọng số và chuẩn hóa bởi $Z_m$, những mẫu bị đoán sai được nhân phình trọng số lên sao cho bộ phân loại vừa học $G_m$ nếu đem áp dụng lại trên phân bố trọng số mới sẽ có độ chính xác đúng bằng $50\\%$ (bằng đoán mò) — buộc cây $G_{m+1}$ phải tập trung giải quyết những ca khó mới.',
        },
        {
          heading: '2. Gradient Boosting (GBDT) & Khai triển bậc hai XGBoost',
          body: String.raw`Thay vì cố định Exponential Loss, **Gradient Boosting (GBDT)** mở rộng cho mọi hàm mất mát khả vi $\ell(y_i, F(x_i))$: ở bước $m$, ta xem giá trị dự đoán $F_{m-1}(x_i)$ là biến số và khớp cây $h_m(x)$ hướng theo **âm gradient hàm số** (*pseudo-residuals* $r_{im} = -\frac{\partial \ell(y_i, F)}{\partial F}\big|_{F_{m-1}(x_i)}$). Với MSE Loss, pseudo-residual chính là phần dư $y_i - F_{m-1}(x_i)$!

**XGBoost** (Chen & Guestrin, 2016) tiến xa hơn bằng khai triển Taylor **bậc hai** gồm cả gradient $g_i = \partial_{\hat{y}}\ell$ và Hessian $h_i = \partial_{\hat{y}}^2\ell$, cộng phạt điều chuẩn phức tạp $\Omega(f) = \gamma T + \frac{\lambda}{2}\sum_{j=1}^T w_j^2$, cho ra công thức nghiệm đóng tuyệt đẹp cho từng lá $j$ và điểm lợi ích cắt nhánh:`,
          math: String.raw`w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma`,
          pitfall:
            'Nếu hệ số pruning tối thiểu $\\gamma > 0$ vượt quá mức giảm mất mát $\\frac{1}{2}[\\dots]$, Gain mang giá trị âm và nhánh đó bị cắt bỏ ngay tại thời điểm dựng cây.',
        },
      ],
      simulationId: 'm5-u4',
      exercises: [
        {
          id: 'm5-u4-adaboost-step',
          title: 'Tính hệ số α_m & Cập nhật trọng số mẫu trong AdaBoost',
          prompt:
            'Cài đặt `adaboost_weight_update(y_pm1, pred_pm1, sample_weights)` nhận nhãn `y_pm1` $\\in \\{-1, +1\\}$, dự đoán `pred_pm1` $\\in \\{-1, +1\\}$ và vector trọng số chuẩn hóa `sample_weights` (tổng $= 1$). Tính sai số trọng số $\\varepsilon$, hệ số $\\alpha = \\frac{1}{2}\\ln\\frac{1 - \\varepsilon}{\\varepsilon}$ (chặn $\\varepsilon$ trong $[10^{-12}, 1 - 10^{-12}]$), cập nhật $w_{\\text{next}} \\propto w \\odot \\exp(-\\alpha y \\odot \\hat{y})$ và chuẩn hóa lại để tổng bằng $1$. Trả về `(eps, alpha, w_next)`.',
          hint: 'Tính `w = np.asarray(sample_weights, dtype=float) / np.sum(sample_weights)`, `eps = float(np.clip(np.sum(w[y_pm1 != pred_pm1]), 1e-12, 1 - 1e-12))`, `alpha = 0.5 * math.log((1.0 - eps) / eps)`. Cập nhật `w_next = w * np.exp(-alpha * y_pm1 * pred_pm1)`, rồi chia cho `np.sum(w_next)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_y_ab = np.array([1., 1., -1., -1., 1.])
_p_ab = np.array([1., -1., -1., -1., 1.]) # 1 error out of 5 -> eps = 0.2
_w_ab = np.ones(5) / 5.0
`,
          starter: String.raw`def adaboost_weight_update(y_pm1, pred_pm1, sample_weights):
    raise NotImplementedError
`,
          solution: String.raw`def adaboost_weight_update(y_pm1, pred_pm1, sample_weights):
    y_pm1 = np.asarray(y_pm1, dtype=float)
    pred_pm1 = np.asarray(pred_pm1, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    w = w / np.sum(w)
    eps = float(np.clip(np.sum(w[y_pm1 != pred_pm1]), 1e-12, 1.0 - 1e-12))
    alpha = 0.5 * math.log((1.0 - eps) / eps)
    w_next = w * np.exp(-alpha * y_pm1 * pred_pm1)
    w_next = w_next / np.sum(w_next)
    return eps, float(alpha), w_next
`,
          test: String.raw`_report('m5-u4-adaboost-step',
    _check('Tính đúng eps = 0.2, alpha = 0.5 * ln(4), và tổng trọng số mới bằng 1', lambda: (
        (lambda eps, alpha, wn: (
            np.isclose(eps, 0.2) and np.isclose(alpha, 0.5 * math.log(4.0)) and np.isclose(np.sum(wn), 1.0)
        ))(*adaboost_weight_update(_y_ab, _p_ab, _w_ab)),
        'Kiểm tra giá trị giải tích bước cập nhật AdaBoost.'
    )),
    _check('Bất biến lý thuyết: sai số của chính bộ phân loại cũ trên trọng số mới bằng đúng 0.5', lambda: (
        (lambda eps, alpha, wn: np.isclose(np.sum(wn[_y_ab != _p_ab]), 0.5))(
            *adaboost_weight_update(_y_ab, _p_ab, _w_ab)
        ),
        'Các mẫu sai nhận tổng trọng số đúng bằng 50% để ép cây tiếp theo học hướng trực giao.'
    )),
)
`,
        },
        {
          id: 'm5-u4-xgb-leaf',
          title: 'Trọng số lá tối ưu & Split Gain bậc hai của XGBoost',
          prompt:
            'Cài đặt `xgboost_leaf_and_gain(g_left, h_left, g_right, h_right, lam=1.0, gamma=0.0)` nhận các mảng/danh sách gradient bậc 1 (`g`) và Hessian bậc 2 (`h`) rơi vào nút trái và nút phải, cùng hai tham số điều chuẩn `lam` ($\\lambda$) và `gamma` ($\\gamma$). Tính trọng số lá tối ưu $w_L^*, w_R^*$ và $\\text{Gain}$ bậc hai, trả về `(w_left, w_right, split_gain)`.',
          hint: 'Gọi `GL = float(np.sum(g_left))`, `HL = float(np.sum(h_left))`, `GR = float(np.sum(g_right))`, `HR = float(np.sum(h_right))`. Tính `w_left = -GL / (HL + lam)`, `w_right = -GR / (HR + lam)`, và `split_gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR)**2 / (HL + HR + lam)) - gamma`.',
          setup: COMMON_SETUP,
          starter: String.raw`def xgboost_leaf_and_gain(g_left, h_left, g_right, h_right, lam=1.0, gamma=0.0):
    raise NotImplementedError
`,
          solution: String.raw`def xgboost_leaf_and_gain(g_left, h_left, g_right, h_right, lam=1.0, gamma=0.0):
    GL = float(np.sum(g_left))
    HL = float(np.sum(h_left))
    GR = float(np.sum(g_right))
    HR = float(np.sum(h_right))
    w_left = -GL / (HL + float(lam))
    w_right = -GR / (HR + float(lam))
    score_left = (GL ** 2) / (HL + float(lam))
    score_right = (GR ** 2) / (HR + float(lam))
    score_parent = ((GL + GR) ** 2) / (HL + HR + float(lam))
    split_gain = 0.5 * (score_left + score_right - score_parent) - float(gamma)
    return float(w_left), float(w_right), float(split_gain)
`,
          test: String.raw`_report('m5-u4-xgb-leaf',
    _check('Tính chính xác trọng số lá tối ưu w* = -G / (H + λ) khi phạt co cụm λ > 0', lambda: (
        (lambda wL, wR, gain: np.isclose(wL, 4.0 / 3.0) and np.isclose(wR, -4.0 / 5.0))(
            *xgboost_leaf_and_gain([-2.0, -2.0], [1.0, 1.0], [3.0, 1.0], [2.0, 2.0], lam=1.0, gamma=0.0)
        ),
        'Tách mẫu gradient trái ngược dấu (-4 vs +4) tạo giá trị dự đoán đối nghịch.'
    )),
    _check('Split Gain trừ đúng phạt γ và phản ánh độ lợi thực tế khi tách nhánh', lambda: (
        (lambda wL, wR, g0, g5: np.isclose(g0 - g5, 5.0) and g0 > 4.0 and g5 < 0.0)(
            *xgboost_leaf_and_gain([-2.0, -2.0], [1.0, 1.0], [3.0, 1.0], [2.0, 2.0], lam=1.0, gamma=0.0)[:2],
            xgboost_leaf_and_gain([-2.0, -2.0], [1.0, 1.0], [3.0, 1.0], [2.0, 2.0], lam=1.0, gamma=0.0)[2],
            xgboost_leaf_and_gain([-2.0, -2.0], [1.0, 1.0], [3.0, 1.0], [2.0, 2.0], lam=1.0, gamma=5.0)[2],
        ),
        'Khi γ > gain gốc, gain âm kích hoạt cơ chế không tách nhánh / tỉa.'
    )),
)
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong XGBoost với Binary Cross-Entropy Loss $\\ell(y, \\hat{y}) = \\ln(1 + e^{\\hat{y}}) - y \\hat{y}$, giá trị $g_i$ và $h_i$ truyền vào hàm `xgboost_leaf_and_gain` ở mỗi vòng boosting có biểu thức tường minh là gì? Tại sao $h_i$ gần $0$ nếu xác suất $p_i \\to 0$ hoặc $p_i \\to 1$?',
          guidance:
            'Với $p_i = \\sigma(\\hat{y}_i)$, đạo hàm bậc 1 là $g_i = p_i - y_i$, và đạo hàm bậc 2 là Hessian $h_i = p_i(1 - p_i) \\in (0, 0.25]$. Khi mô hình đã rất tự tin ($p_i \\to 0$ hoặc $1$), $h_i$ rất nhỏ; tham số `min_child_weight` trong XGBoost đặt ngưỡng tối thiểu trên tổng $\\sum_{i \\in I_j} h_i$ nhằm ngăn việc tạo lá quá nhỏ ở các vùng tự tin cao.',
        },
      ],
    },
  ],
};
