import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

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
      bigPicture:
        'Xuyên suốt Buổi 1, bạn sẽ làm việc trực tiếp trên bộ dữ liệu nhà đất thật **UCI Real Estate Valuation** (`414` giao dịch bất động sản tại Tân Bắc, Đài Loan — nạp sẵn trong Python tại `data/real_estate.csv`). Mỗi căn nhà là một hàng gồm tuổi nhà, khoảng cách tới ga tàu điện MRT, số cửa hàng tiện lợi xung quanh; ghép `414` căn lại thành **Ma trận thiết kế $X$**. Từ bảng thật này, bạn sẽ đi từng bước: chuẩn hóa cột khác đơn vị → nhân trọng số dự đoán giá → tính Gradient → tự huấn luyện mô hình ở cuối bài.',
      bigPictureTable: {
        caption: ">>> df = pd.read_csv('data/real_estate.csv'); df[['age_years', 'mrt_distance_m', 'stores', 'price_10k_ntd_per_ping']].head(4)   # 4 trong 414 căn thật",
        columns: ['Dòng (i)', 'age_years (tuổi)', 'mrt_distance_m (m)', 'stores (cửa hàng)', 'price y (10k NTD/ping)'],
        rows: [
          ['0 (x₀)', 32.0, 84.88, 10, 37.9],
          ['1 (x₁)', 19.5, 306.59, 9, 42.2],
          ['2 (x₂)', 13.3, 561.98, 5, 47.3],
          ['3 (x₃)', 13.3, 561.98, 5, 54.8],
        ],
        highlightTargetCol: 4,
        footnote:
          'Ba cột đặc trưng (`X = df[["age_years", "mrt_distance_m", "stores"]].to_numpy()`) tạo thành **Ma trận $X \\in \\mathbb{R}^{414 \\times 3}$**; cột tô vàng cuối cùng (`y = df["price_10k_ntd_per_ping"].to_numpy()`) là **Vector giá nhà $y \\in \\mathbb{R}^{414}$**.',
      },
      summary:
        'Biểu diễn tập dữ liệu dạng ma trận thiết kế $X \\in \\mathbb{R}^{n \\times d}$, hình học của chuẩn và phép chiếu trực giao, kỹ thuật vector hóa NumPy và tính gradient dạng ma trận trên bộ dữ liệu UCI Real Estate Valuation.',
      outcomes: [
        'Phân biệt hình học giữa các chuẩn $L_1, L_2, L_\\infty$ và chuẩn hóa Z-score cho các cột khác đơn vị trong `real_estate.csv`.',
        'Sử dụng broadcasting NumPy để chuẩn hóa ma trận và đo khoảng cách giữa các căn nhà không cần vòng lặp.',
        'Tính chính xác gradient theo vector tham số $w$ cho hàm mất mát bình phương trung bình $\\|Xw - y\\|_2^2$ và huấn luyện dự đoán giá nhà.',
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
          heading: '1. Ma trận thiết kế, Chuẩn vector & Chuẩn hóa Z-score',
          realWorld:
            'Nhìn vào bảng dữ liệu thật `data/real_estate.csv` ở trên: cột **`mrt_distance_m`** (khoảng cách ra ga tàu) dao động từ `23.4 m` tới `6488.0 m`, còn cột **`stores`** (số cửa hàng tiện lợi) chỉ chạy từ `0` tới `10`!\n\nNếu bạn đo khoảng cách giữa hai căn nhà mà để nguyên số thô, độ chênh vài trăm mét ra ga tàu sẽ **nuốt chửng hoàn toàn** sự khác biệt `1–2` cửa hàng tiện lợi chỉ vì đơn vị mét có con số to hơn. Vì vậy ta phải **(1) Đưa mọi cột về cùng thước đo** bằng Z-score và **(2) Chọn thước đo độ dài (Chuẩn — Norm)** để so sánh các căn nhà công bằng.',
          workedExample:
            String.raw`Lấy đúng cột ` + '`age_years`' + String.raw` (tuổi nhà) của **3 căn đầu tiên trong ` + '`real_estate.csv`' + String.raw`**: ` + '`[32.0, 19.5, 13.3]`' + String.raw`.

- **Bước 1 — Trung bình ($\mu$)**: $(32.0 + 19.5 + 13.3) / 3 = 21.6$.
- **Bước 2 — Độ lệch khỏi trung bình**: ` + '`[32.0 - 21.6, 19.5 - 21.6, 13.3 - 21.6] = [+10.4, -2.1, -8.3]`' + String.raw`.
- **Bước 3 — Độ phân tán chuẩn ($\sigma$)**: $\sqrt{\frac{10.4^2 + (-2.1)^2 + (-8.3)^2}{3}} = \sqrt{\frac{181.46}{3}} \approx 7.777$.
- **Bước 4 — Chia lấy Z-score**: ` + '`[+10.4 / 7.777, -2.1 / 7.777, -8.3 / 7.777] = [+1.337, -0.270, -1.067]`' + String.raw`.

Khi áp dụng công thức đó trên cả **414 căn nhà thật**, hai căn đầu có vector Z-score ` + '`z_0 = [1.2556, -0.7925, 2.0074]`' + String.raw` và ` + '`z_1 = [0.1571, -0.6166, 1.6675]`' + String.raw`, hiệu ` + '`diff = z_0 - z_1 = [1.0985, -0.1759, 0.3399]`' + String.raw`:
- **Đường chim bay ($L_2$)**: $\sqrt{1.0985^2 + (-0.1759)^2 + 0.3399^2} \approx 1.163$.
- **Đi theo trục tọa độ ($L_1$)**: $|1.0985| + |-0.1759| + |0.3399| \approx 1.614$.
- **Chênh lệch lớn nhất trên một cột ($L_\infty$)**: $\max(|1.0985|, |-0.1759|, |0.3399|) \approx 1.099$.`,
          workedExampleCode: `import numpy as np
import pandas as pd

df = pd.read_csv('data/real_estate.csv')
print("3 căn đầu từ data/real_estate.csv:")
print(df[['age_years', 'mrt_distance_m', 'stores', 'price_10k_ntd_per_ping']].head(3))

# 1. Tính Z-score cho age_years của 3 căn đầu
age3 = df['age_years'].iloc[:3].to_numpy(dtype=float)
z3 = (age3 - age3.mean()) / age3.std()
print("\\nZ-score 3 căn đầu:", np.round(z3, 3))

# 2. Chuẩn hóa 3 cột đặc trưng của toàn bộ 414 căn và đo khoảng cách căn 0 vs căn 1
X = df[['age_years', 'mrt_distance_m', 'stores']].to_numpy(dtype=float)
Z = (X - X.mean(axis=0)) / X.std(axis=0)
diff = Z[0] - Z[1]
print(f"L2 = {np.linalg.norm(diff, 2):.3f}, L1 = {np.linalg.norm(diff, 1):.3f}, Linf = {np.linalg.norm(diff, np.inf):.3f}")`,
          body: String.raw`Trong học có giám sát, mỗi quan sát được biểu diễn dưới dạng vector cột $x_i \in \mathbb{R}^d$, ghép theo từng hàng thành ma trận thiết kế $X \in \mathbb{R}^{n \times d}$, cùng vector nhãn mục tiêu $y \in \mathbb{R}^n$.

Để đo độ lớn sai số hoặc ràng buộc độ phức tạp của tham số $w \in \mathbb{R}^d$, ta sử dụng chuẩn $\ell_p$:
- **Chuẩn $\ell_2$ (Euclidean)**: hình cầu đẳng hướng, trơn khả vi tại mọi điểm khác $0$.
- **Chuẩn $\ell_1$ (Manhattan)**: mặt cầu là khối bát diện có góc nhọn trên các trục tọa độ, thúc đẩy tính thưa (sparsity).
- **Chuẩn $\ell_\infty$**: đo sai lệch lớn nhất theo một tọa độ đơn lẻ.`,
          math: String.raw`\|x\|_p = \left(\sum_{j=1}^d |x_j|^p\right)^{1/p}, \quad \|x\|_1 = \sum_{j=1}^d |x_j|, \quad \|x\|_2 = \sqrt{x^\top x}, \quad \|x\|_\infty = \max_{j} |x_j|`,
          symbols: [
            {
              symbol: '$X \\in \\mathbb{R}^{n \\times d}$',
              readAs: 'Ma trận thiết kế $n$ dòng, $d$ cột',
              meaning: 'Bảng dữ liệu gồm $n$ mẫu (414 căn nhà), mỗi mẫu có $d$ đặc trưng (tuổi nhà, khoảng cách MRT, số cửa hàng).',
            },
            {
              symbol: '$\\|x\\|_p$',
              readAs: 'Chuẩn bậc $p$ của vector $x$',
              meaning: 'Thước đo độ dài tổng quát biến mảng nhiều số thành đúng 1 con số không âm.',
            },
          ],
          intuition:
            'Khoảng cách Euclidean bình phương giữa hai tập điểm $A \\in \\mathbb{R}^{n \\times d}$ và $B \\in \\mathbb{R}^{m \\times d}$ có thể khai triển qua tích vô hướng hoặc tính trực tiếp bằng broadcasting 3 chiều $(n, 1, d) - (1, m, d)$ mà không cần vòng lặp Python.',
          pitfall:
            'Chuẩn hóa Z-score với cột hằng (độ lệch chuẩn $\\sigma_j = 0$) sẽ tạo ra `NaN` hoặc `Inf` nếu không thay mẫu số bằng $1$ trước khi chia.',
        },
        {
          heading: '2. Tích vô hướng (Nhân cặp rồi cộng) & Phép chiếu',
          realWorld:
            'Sau khi chuẩn hóa Z-score 3 cột của `data/real_estate.csv`, mô hình tuyến tính định giá một căn nhà bằng cách giữ một bộ **trọng số tác động** `w = [-3.0, -7.0, 4.0]` (tuổi càng cao giảm giá `-3.0`, càng xa ga MRT giảm mạnh `-7.0`, càng nhiều cửa hàng tiện lợi tăng giá `+4.0`) cộng với mức giá trung bình nền `b = 38.0`.\n\nVới mỗi căn nhà, máy tính nhân từng đặc trưng đã chuẩn hóa với trọng số tương ứng rồi cộng dồn lại (`Z[i] @ w + b`). Động tác "nhân từng cặp cột rồi cộng" đó chính là **Tích vô hướng (Dot Product)**!',
          workedExample:
            String.raw`Thử đoán giá cho **2 căn đầu tiên trong ` + '`real_estate.csv`' + String.raw`** (đã chuẩn hóa Z-score) với ` + '`w = [-3.0, -7.0, 4.0]`' + String.raw` và ` + '`b = 38.0`' + String.raw`:

- **Căn 0** (` + '`z_0 = [1.2556, -0.7925, 2.0074]`' + String.raw`, giá thật ` + '`y_0 = 37.9`' + String.raw`):
  $\hat{y}_0 = 1.2556\cdot(-3) + (-0.7925)\cdot(-7) + 2.0074\cdot 4 + 38.0 = -3.767 + 5.548 + 8.030 + 38.0 \approx 47.81$.
- **Căn 1** (` + '`z_1 = [0.1571, -0.6166, 1.6675]`' + String.raw`, giá thật ` + '`y_1 = 42.2`' + String.raw`):
  $\hat{y}_1 = 0.1571\cdot(-3) + (-0.6166)\cdot(-7) + 1.6675\cdot 4 + 38.0 = -0.471 + 4.316 + 6.670 + 38.0 \approx 48.52$.

Trong Python/NumPy, chỉ một biểu thức ` + '`Z @ w + b`' + String.raw` sẽ tính đồng thời tích vô hướng cho toàn bộ ` + '`414`' + String.raw` căn nhà mà không cần bất kỳ vòng lặp ` + '`for`' + String.raw` nào.`,
          workedExampleCode: `import numpy as np
import pandas as pd

df = pd.read_csv('data/real_estate.csv')
X = df[['age_years', 'mrt_distance_m', 'stores']].to_numpy(dtype=float)
y = df['price_10k_ntd_per_ping'].to_numpy(dtype=float)
Z = (X - X.mean(axis=0)) / X.std(axis=0)

w = np.array([-3.0, -7.0, 4.0])
b = 38.0
y_hat = Z @ w + b   # Tính tích vô hướng cho cả 414 căn cùng lúc

for i in range(5):
    print(f"Căn {i}: Giá thật = {y[i]:5.1f} | Dự đoán = {y_hat[i]:5.2f} | Sai lệch = {y_hat[i] - y[i]:+6.2f}")`,
          body: String.raw`Mô hình tuyến tính cơ bản thực chất tính tích vô hướng $x^\top w + b$:
- Về đại số, đây là tổ hợp tuyến tính của các đặc trưng.
- Về hình học, $(w^\top x) / \|w\|_2$ là độ dài hình chiếu có dấu của vector dữ liệu $x$ lên phương của vector pháp tuyến $w$. Siêu phẳng quyết định $\{x : w^\top x + b = 0\}$ luôn vuông góc với vector $w$.`,
          math: String.raw`\text{proj}_w(x) = \frac{w^\top x}{\|w\|_2^2}\, w, \qquad Xw + b\mathbf{1}_n = \hat{y} \in \mathbb{R}^n`,
          intuition:
            'Trong lập trình ma trận, quy ước một mẫu nằm trên một hàng của $X$ cho phép biểu thức `X @ w + b` tự động broadcast bias vô hướng $b$ ra toàn bộ $n$ mẫu cùng lúc.',
        },
        {
          heading: '3. Gradient: Làm sao máy tính biết đường giảm sai số?',
          realWorld:
            'Nếu ban đầu máy tính chưa biết giá trị hợp lý của `w` (ví dụ đặt toàn số `0`: `w = [0, 0, 0]`, `b = 0.0`), mức sai số bình phương trung bình trên `414` căn nhà thật sẽ rất lớn (`MSE = 1627.18`). Làm sao máy tính tự biết phải tăng hay giảm từng trọng số `w[0], w[1], w[2]`?\n\nHãy tưởng tượng bạn đứng trên một thung lũng sương mù, độ cao là **sai số MSE**. Độ dốc của mặt đất theo từng hướng chính là **Gradient ($\nabla_w$)**. Máy tính chỉ cần bước ngược chiều độ dốc (`w = w - lr * grad`) là sai số giảm xuống đều đặn sau mỗi vòng lặp!',
          workedExample:
            String.raw`Thử tính Gradient bằng tay trên **2 căn đầu tiên của ` + '`real_estate.csv`' + String.raw`** tại ` + '`w = [0, 0, 0]`' + String.raw`, ` + '`b = 38.0`' + String.raw` (giá thật ` + '`y = [37.9, 42.2]`' + String.raw`):

- **Bước 1 (Đoán thử & Phần dư)**: $\hat{y}_0 = \hat{y}_1 = 38.0 \implies e_0 = 38.0 - 37.9 = +0.1$, $e_1 = 38.0 - 42.2 = -4.2$.
- **Bước 2 (MSE của 2 căn)**: $L = \frac{0.1^2 + (-4.2)^2}{2} = \frac{0.01 + 17.64}{2} = 8.825$.
- **Bước 3 (Gradient từng đặc trưng $\frac{2}{2}(z_{0,j}e_0 + z_{1,j}e_1)$)**:
  - Tuổi nhà ($j=0$): $1.2556\cdot(0.1) + 0.1571\cdot(-4.2) \approx -0.534$.
  - Khoảng cách MRT ($j=1$): $-0.7925\cdot(0.1) + (-0.6166)\cdot(-4.2) \approx +2.511$.
  - Số cửa hàng ($j=2$): $2.0074\cdot(0.1) + 1.6675\cdot(-4.2) \approx -6.803$.

Trong Python bên cạnh, ta vừa đối chiếu Gradient trên 2 căn này với công thức kiểm tra số ` + '`(L(w+h) - L(w-h)) / (2*h)`' + String.raw`, vừa chạy ` + '`10`' + String.raw` bước Gradient Descent trên **toàn bộ 414 căn nhà thật** để thấy MSE tụt thẳng từ ` + '`1627.18`' + String.raw` xuống ` + '`101.81`' + String.raw`!`,
          workedExampleCode: `import numpy as np
import pandas as pd

df = pd.read_csv('data/real_estate.csv')
X = df[['age_years', 'mrt_distance_m', 'stores']].to_numpy(dtype=float)
y = df['price_10k_ntd_per_ping'].to_numpy(dtype=float)
Z = (X - X.mean(axis=0)) / X.std(axis=0)

# 1. Kiểm chứng tính tay 2 căn đầu bằng công thức giải tích & sai phân số
Z2, y2 = Z[:2], y[:2]
w0, b0 = np.zeros(3), 38.0
err2 = (Z2 @ w0 + b0) - y2
grad_w2 = (2.0 / len(y2)) * (Z2.T @ err2)
h = 1e-4
num_grad = np.array([
    (np.mean((Z2 @ (w0 + e) + b0 - y2) ** 2) - np.mean((Z2 @ (w0 - e) + b0 - y2) ** 2)) / (2 * h)
    for e in np.eye(3) * h
])
print("MSE 2 căn đầu:", round(float(np.mean(err2 ** 2)), 3))
print("Gradient w giải tích:", np.round(grad_w2, 3), "| Sai phân số:", np.round(num_grad, 3))

# 2. Chạy 10 bước Gradient Descent trên TOÀN BỘ 414 căn nhà thật
w, b, lr = np.zeros(3), 0.0, 0.1
print("\\nMSE ban đầu (414 căn):", round(float(np.mean((Z @ w + b - y) ** 2)), 2))
for _ in range(10):
    err = (Z @ w + b) - y
    w -= lr * (2.0 / len(y)) * (Z.T @ err)
    b -= lr * (2.0 / len(y)) * np.sum(err)
print("MSE sau 10 bước:", round(float(np.mean((Z @ w + b - y) ** 2)), 2), "| w =", np.round(w, 3))`,
          body: String.raw`Thay vì lấy đạo hàm riêng từng phần tử $w_1, \dots, w_d$ rồi ghép lại, ta đạo hàm trực tiếp trên không gian vector $\mathbb{R}^d$ (theo quy ước mẫu số — cùng kích thước với tham số $w$).

Ba đồng nhất thức cốt lõi dùng xuyên suốt các mô hình học máy:
- Gradient của dạng tuyến tính: $\nabla_w (w^\top x) = x$.
- Gradient của dạng toàn phương tổng quát: $\nabla_w (w^\top A w) = (A + A^\top)w$ (rút gọn thành $2Aw$ nếu $A$ đối xứng).
- Gradient của hàm mất mát bình phương trung bình (MSE) với $L(w) = \frac{1}{n}\|Xw - y\|_2^2$.`,
          math: String.raw`L(w) = \frac{1}{n}(Xw - y)^\top(Xw - y) \implies \nabla_w L(w) = \frac{2}{n} X^\top (Xw - y) \in \mathbb{R}^d`,
          symbols: [
            {
              symbol: '$\\nabla_w L(w)$',
              readAs: 'Gradient (độ dốc) của hàm sai số $L$ theo tham số $w$',
              meaning: 'Mảng chứa đạo hàm riêng theo từng trọng số, cho biết chiều cần nhích để giảm sai số.',
            },
            {
              symbol: '$X^\\top (Xw - y)$',
              readAs: 'Chuyển vị $X$ nhân với phần dư',
              meaning: 'Cách viết ma trận ngắn gọn của tổng `sum_i err_i * X[i, j]` cho mọi đặc trưng $j$ cùng lúc.',
            },
          ],
          intuition:
            'Chú ý sự khớp chiều (dimensionality sanity check): $X$ có kích thước $(n, d)$, phần dư $(Xw - y)$ có kích thước $(n,)$, do đó $X^\\top (Xw - y)$ có kích thước $(d, n) \\times (n,) = (d,)$ — trùng khớp chính xác với shape của $w$.',
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
        {
          id: 'm1-u1-housing-pipeline',
          title: 'Thực hành trọn quy trình: Xử lý dữ liệu & Huấn luyện dự đoán giá nhà (UCI Real Estate)',
          prompt:
            'Áp dụng toàn bộ 3 phần vừa học lên bảng `df` (đã nạp sẵn tại `df_housing = pd.read_csv(\'data/real_estate.csv\')`). Cài đặt `train_housing_gd(df, lr=0.1, steps=200)`: (1) tách `X` gồm 3 cột `[\'age_years\', \'mrt_distance_m\', \'stores\']` và `y` là `\'price_10k_ntd_per_ping\'` (không sửa `df` gốc); (2) chia 80% đầu (`split = int(0.8 * len(df))`) làm tập train, phần còn lại làm validation; (3) tính `mu, sd` (`ddof=0`, `sd == 0` thay bằng `1.0`) **chỉ trên train** rồi chuẩn hóa Z-score cho cả train và val; (4) khởi tạo `w = np.zeros(3)`, `b = 0.0`, chạy `steps` bước Gradient Descent cập nhật đồng thời `w, b`; (5) trả về dict `{\'w\': w, \'b\': float(b), \'train_mse\': ..., \'val_mse\': ...}`.',
          hint:
            'Dùng `split = int(0.8 * len(X))`; chuẩn hóa `Z_tr = (X_tr - mu) / sd`, `Z_va = (X_va - mu) / sd`. Trong mỗi bước lặp: `err = Z_tr @ w + b - y_tr`, `w = w - lr * (2.0 / len(y_tr)) * (Z_tr.T @ err)`, `b = b - lr * (2.0 / len(y_tr)) * np.sum(err)`.',
          setup:
            COMMON_SETUP +
            String.raw`
import pandas as pd
df_housing = pd.read_csv('data/real_estate.csv')
_df_orig = df_housing.copy(deep=True)
_split_idx = int(0.8 * len(df_housing))
_y_tr_mean = df_housing['price_10k_ntd_per_ping'].iloc[:_split_idx].mean()
_base_val_mse = float(np.mean((df_housing['price_10k_ntd_per_ping'].iloc[_split_idx:] - _y_tr_mean) ** 2))
`,
          starter: String.raw`def train_housing_gd(df, lr=0.1, steps=200):
    # X: ['age_years', 'mrt_distance_m', 'stores']
    # y: 'price_10k_ntd_per_ping'
    raise NotImplementedError
`,
          solution: String.raw`def train_housing_gd(df, lr=0.1, steps=200):
    cols = ['age_years', 'mrt_distance_m', 'stores']
    X = df[cols].to_numpy(dtype=float)
    y = df['price_10k_ntd_per_ping'].to_numpy(dtype=float)

    split = int(0.8 * len(X))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    mu = X_tr.mean(axis=0)
    sd = np.where(X_tr.std(axis=0) == 0, 1.0, X_tr.std(axis=0))
    Z_tr = (X_tr - mu) / sd
    Z_va = (X_va - mu) / sd

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    n_tr = len(y_tr)
    for _ in range(steps):
        err = (Z_tr @ w + b) - y_tr
        grad_w = (2.0 / n_tr) * (Z_tr.T @ err)
        grad_b = (2.0 / n_tr) * float(np.sum(err))
        w = w - lr * grad_w
        b = b - lr * grad_b

    train_mse = float(np.mean(((Z_tr @ w + b) - y_tr) ** 2))
    val_mse = float(np.mean(((Z_va @ w + b) - y_va) ** 2))
    return {'w': w, 'b': float(b), 'train_mse': train_mse, 'val_mse': val_mse}
`,
          test: String.raw`_cache = {}
def _run_base():
    if 'base' not in _cache:
        _cache['base'] = train_housing_gd(df_housing, lr=0.1, steps=200)
    return _cache['base']

def _run_leak():
    if 'leak' not in _cache:
        _df_leak = _df_orig.copy(deep=True)
        _df_leak.loc[_split_idx:, 'age_years'] = _df_leak.loc[_split_idx:, 'age_years'] + 1000.0
        _df_leak.loc[_split_idx:, 'mrt_distance_m'] = _df_leak.loc[_split_idx:, 'mrt_distance_m'] + 100000.0
        _cache['leak'] = train_housing_gd(_df_leak, lr=0.1, steps=200)
    return _cache['leak']

_report('m1-u1-housing-pipeline',
    _check('Trả về dict hợp lệ gồm w (shape 3), b, train_mse, val_mse', lambda: (lambda r: (
        isinstance(r, dict)
        and all(k in r for k in ('w', 'b', 'train_mse', 'val_mse'))
        and np.asarray(r['w'], dtype=float).shape == (3,)
        and np.isfinite(r['b'])
        and np.isfinite(r['train_mse'])
        and np.isfinite(r['val_mse']),
        f"train_mse={r.get('train_mse', 0):.2f}, val_mse={r.get('val_mse', 0):.2f}" if isinstance(r, dict) and 'train_mse' in r and 'val_mse' in r else 'Thiếu khóa kết quả.'
    ))(_run_base())),
    _check('Chuẩn hóa Z-score chỉ fit trên tập train (chống rò rỉ dữ liệu val)', lambda: (lambda r, rl: (
        np.allclose(r['w'], rl['w'], atol=1e-9) and np.isclose(r['b'], rl['b'], atol=1e-9),
        'Biến đổi tập validation không được làm thay đổi w và b đã học.'
    ))(_run_base(), _run_leak())),
    _check('MSE trên tập train & validation giảm mạnh hơn baseline đoán trung bình', lambda: (lambda r: (
        r['train_mse'] < 95.0 and r['val_mse'] < min(70.0, _base_val_mse * 0.5),
        f"Baseline val_mse={_base_val_mse:.2f} -> Mô hình val_mse={r['val_mse']:.2f}"
    ))(_run_base())),
    _check('Trọng số hợp lý với thực tế nhà đất và không sửa df_housing gốc', lambda: (lambda r: (
        r['w'][1] < 0 and r['w'][2] > 0 and abs(r['b'] - 37.9) < 1.0 and df_housing.equals(_df_orig),
        f"w={np.round(r['w'], 2)}, b={r['b']:.2f}"
    ))(_run_base())),
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
      bigPicture:
        'Tại sao gần như mọi mô hình học máy đều chọn tối ưu sai số bình phương (MSE) hay thêm hình phạt trọng số (Regularization $L_2$)? Câu trả lời không phải ngẫu nhiên: đằng sau chúng là **quy tắc xác suất**. Bài này giải thích **MLE** (chọn đáp án làm dữ liệu đang thấy hợp lý nhất) và **MAP** (biết kết hợp thêm kinh nghiệm trước khi mở dữ liệu để không kết luận vội vàng khi mẫu ít).',
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
          heading: '1. MLE: Tìm tham số làm dữ liệu hợp lý nhất',
          realWorld:
            'Bạn nhặt một đồng xu, tung thử **4 lần** và thấy ra **3 Ngửa, 1 Sấp**. Nếu bắt buộc phải đoán xem tỷ lệ ra Ngửa thật $\\theta$ của đồng xu đó là bao nhiêu (`0.2`, `0.5`, hay `0.75`?), chắc chắn bạn sẽ đặt cược cao nhất vào **`0.75`** (`3/4`), vì con số đó làm chuỗi kết quả `[Ngửa, Ngửa, Ngửa, Sấp]` có xác suất xuất hiện cao nhất! Đó chính xác là nguyên lý **MLE (Maximum Likelihood Estimation)**.\n\nCòn tại sao dân AI luôn lấy **`log(...)`** trước khi cộng? Bởi vì nếu có 1.000 mẫu dữ liệu, nhân 1.000 xác suất `0.75 * 0.75 * ... * 0.25` lại sẽ tạo ra con số `0.0000...00` quá nhỏ khiến máy tính làm tròn thành `0.0` (tràn số dưới)! Chuyển phép nhân thành phép **cộng các log** giúp máy tính tính an toàn tuyệt đối.',
          workedExample:
            String.raw`Với dữ liệu tung xu ` + '`3 Ngửa, 1 Sấp`' + String.raw`: xác suất sinh dữ liệu là $L(\theta) = \theta^3 (1 - \theta)^1$.
Thử so sánh giá trị tại 3 ứng viên:
- Nếu $\theta = 0.50$: $L = 0.5^3 \cdot 0.5 = 0.0625$ ($\log L \approx -2.77$).
- Nếu $\theta = 0.75$: $L = 0.75^3 \cdot 0.25 = \mathbf{0.1055}$ ($\log L \approx \mathbf{-2.25}$ — **Cao nhất!**).
- Nếu $\theta = 0.90$: $L = 0.9^3 \cdot 0.1 = 0.0729$ ($\log L \approx -2.62$).

Với dữ liệu số liên tục như chiều cao ` + '`[2.0, 4.0, 6.0]`' + String.raw` giả sử theo phân phối chuẩn hình chuông: nghiệm MLE cho ngay trung bình $\hat{\mu} = (2+4+6)/3 = 4.0$ và phương sai $\hat{\sigma}^2 = \frac{(-2)^2 + 0^2 + 2^2}{3} = \frac{8}{3} \approx 2.667$.`,
          workedExampleCode: `import math

# So sánh Likelihood cho 3 Ngửa, 1 Sấp ở các giá trị theta khác nhau
for theta in [0.50, 0.75, 0.90]:
    L = (theta ** 3) * (1.0 - theta)
    print(f"theta={theta:.2f} -> L={L:.4f}, log(L)={math.log(L):.2f}")

# Kiểm tra Gaussian MLE trên [2.0, 4.0, 6.0]
x = [2.0, 4.0, 6.0]
mu = sum(x) / len(x)
var = sum((v - mu) ** 2 for v in x) / len(x)
print(f"mu_hat = {mu}, var_hat = {var:.3f}")`,
          codeBridge: {
            title: 'Tính Log-Likelihood bằng vòng lặp Python vs NumPy',
            pythonLoop: `import math
n = len(x)
mu = sum(x) / n
var = sum((xi - mu) ** 2 for xi in x) / n
ll = 0.0
for xi in x:
    ll += -0.5 * math.log(2 * math.pi * var) - ((xi - mu) ** 2) / (2 * var)`,
            numpyWay: `mu = np.mean(x)
var = np.var(x, ddof=0)   # Chia n thay vì n-1
ll = np.sum(-0.5 * np.log(2 * np.pi * var) - ((x - mu) ** 2) / (2 * var))`,
            explanation:
              'Chú ý `ddof=0` trong `np.var(x, ddof=0)`: công thức MLE chia cho đúng số lượng mẫu `n`, khác với phương sai mẫu hiệu chỉnh `ddof=1` (chia `n - 1`) trong một số sách thống kê.',
          },
          body: String.raw`Cho tập dữ liệu $\mathcal{D} = \{x_1, \dots, x_n\}$ được sinh độc lập và đồng phân phối (i.i.d.) từ phân phối tham số $p(x \mid \theta)$. Nguyên lý MLE chọn bộ tham số $\theta$ làm cho dữ liệu thực tế quan sát được có khả năng xảy ra cao nhất. Vì hàm $\log$ đơn điệu tăng, việc chuyển tích xác suất sang tổng Log-Likelihood giúp tránh tràn số dưới (underflow) và thuận tiện lấy đạo hàm:`,
          math: String.raw`\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^n p(x_i \mid \theta) = \arg\max_{\theta} \sum_{i=1}^n \log p(x_i \mid \theta)`,
          intuition:
            'Với phân phối chuẩn 1 chiều $\\mathcal{N}(\\mu, \\sigma^2)$, đặt đạo hàm Log-Likelihood bằng $0$ cho kết quả giải tích: $\\hat{\\mu}_{\\text{MLE}} = \\frac{1}{n}\\sum x_i$ và $\\hat{\\sigma}^2_{\\text{MLE}} = \\frac{1}{n}\\sum (x_i - \\hat{\\mu})^2$.',
          pitfall:
            'Hàm hợp lý $L(\\theta; \\mathcal{D})$ **không phải là phân phối xác suất theo $\\theta$** (tích phân theo $\\theta$ không bằng $1$). Ngoài ra, khi cỡ mẫu $n$ nhỏ (ví dụ tung đồng xu $3$ lần đều ra ngửa), $\\hat{\\theta}_{\\text{MLE}} = 1.0$ dễ bị cực đoan hóa (overconfident).',
        },
        {
          heading: '2. MAP: Kết hợp Kinh nghiệm ban đầu để khỏi kết luận vội',
          realWorld:
            'Bạn lên sàn thương mại điện tử chọn mua tai nghe:\n- **Shop A** mới mở, mới có đúng **3 lượt mua** và cả 3 đều đánh giá 5 sao (`MLE = 3/3 = 100%` tuyệt đối!).\n- **Shop B** bán lâu năm, có **1.000 lượt mua** trong đó **950 đánh giá 5 sao** (`MLE = 95%`).\n\nBạn có tin Shop A xịn hơn Shop B 100% không? Không! Vì mẫu `n=3` quá nhỏ. Các hệ thống thông minh ngầm bỏ sẵn **vài phiếu đánh giá "trung bình" giả định** (gọi là **Tiên nghiệm Prior**) vào bình chọn trước khi cộng dữ liệu thật — đó chính là ước lượng **MAP (Maximum A Posteriori)**.',
          workedExample:
            String.raw`Dùng bộ đếm ` + '`Beta(alpha=4, beta=4)`' + String.raw` tương đương với việc giả định trước khi nhìn shop nào, nó giống một shop bình thường có ` + '`3`' + String.raw` phiếu tốt giả (` + '`alpha - 1 = 3`' + String.raw`) và ` + '`3`' + String.raw` phiếu xấu giả (` + '`beta - 1 = 3`' + String.raw`):

- Với **Shop A** (` + '`k=3`' + String.raw` tốt trong ` + '`n=3`' + String.raw` lượt):
  - **MLE (chỉ tin 3 mẫu)**: $3 / 3 = \mathbf{1.00}$ (100% — quá cực đoan!).
  - **MAP (thêm kinh nghiệm trước)**: $\frac{3 + 4 - 1}{3 + 4 + 4 - 2} = \frac{6}{9} = \mathbf{0.667}$ (kéo về mức an toàn!).
  - **Posterior Mean (kỳ vọng hậu nghiệm)**: $\frac{4 + 3}{4 + 4 + 3} = \frac{7}{11} \approx \mathbf{0.636}$.
- Với **Shop B** (` + '`k=950`' + String.raw` tốt trong ` + '`n=1000`' + String.raw` lượt): $\text{MAP} = \frac{950 + 3}{1000 + 6} \approx \mathbf{0.947}$ (dữ liệu nhiều thì kinh nghiệm giả ban đầu hầu như không ảnh hưởng nữa!).`,
          workedExampleCode: `def compare_mle_map(name, k, n, alpha=4, beta=4):
    mle = k / n
    map_est = (k + alpha - 1) / (n + alpha + beta - 2)
    post_mean = (k + alpha) / (n + alpha + beta)
    print(f"{name}: MLE={mle:.3f}, MAP={map_est:.3f}, PostMean={post_mean:.3f}")

compare_mle_map("Shop A (ít mẫu)", k=3, n=3)
compare_mle_map("Shop B (nhiều mẫu)", k=950, n=1000)`,
          codeBridge: {
            title: 'So sánh MLE, MAP và Posterior Mean trong Python',
            pythonLoop: `mle = k / n
# Cập nhật bộ đếm Beta hậu nghiệm (cộng thêm quan sát thật vào prior)
a_post = alpha + k
b_post = beta + (n - k)
map_est = (a_post - 1) / (a_post + b_post - 2) if (a_post + b_post > 2) else mle
post_mean = a_post / (a_post + b_post)`,
            numpyWay: `# Khi dữ liệu là vector nhiều đồng xu / nhiều sản phẩm cùng lúc:
mle = ks / ns
a_post = alpha + ks
b_post = beta + (ns - ks)
post_mean = a_post / (a_post + b_post)`,
            explanation:
              'Các số `alpha` và `beta` đóng vai trò hết sức tự nhiên như **"số lần thành công và thất bại giả định trước khi thí nghiệm"**.',
          },
          body: String.raw`Thay vì xem $\theta$ là hằng số chưa biết (như trường phái tần suất), suy luận Bayes gán cho $\theta$ một phân phối tiên nghiệm (Prior) $p(\theta)$ phản ánh tri thức trước khi thấy dữ liệu, rồi cập nhật thành phân phối hậu nghiệm (Posterior) theo Định lý Bayes: $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$.

Với dữ liệu Bernoulli có $k$ lần thành công trong $n$ phép thử và tiên nghiệm liên hợp $\theta \sim \text{Beta}(\alpha, \beta)$, phân phối hậu nghiệm vẫn là $\text{Beta}(\alpha + k, \beta + n - k)$:`,
          math: String.raw`\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right] = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} \quad (\alpha + k > 1,\; \beta + n - k > 1)`,
          intuition:
            'Các siêu tham số $\\alpha - 1$ và $\\beta - 1$ đóng vai trò như số "quan sát giả" (pseudo-counts) giúp làm mượt ước lượng khi dữ liệu thưa. Khi $n \\to \\infty$, đóng góp của prior mờ dần và $\\hat{\\theta}_{\\text{MAP}} \\to \\hat{\\theta}_{\\text{MLE}}$.',
        },
        {
          heading: '3. Cầu nối kỳ diệu: Từ Xác suất sinh ra MSE & Ridge ($L_2$)',
          realWorld:
            'Trong code ML bạn sẽ liên tục thấy dòng phạt `loss = mse + lam * sum(w**2)` (gọi là **Ridge Regression**). Thực chất công thức phạt bình phương trọng số đó **xuất phát từ xác suất**: giả sử rằng các trọng số đúng ngoài đời thường nằm quanh `0` theo hình chuông Gaussian (tức hiếm khi có trọng số to khổng lồ vô lý) — khi lấy `log` của đường cong chuông $e^{-w^2}$, hàm mũ rơi xuống biến đúng thành hình phạt $\|w\|_2^2$!',
          workedExample:
            String.raw`Giả sử hai mô hình cùng có sai số dự đoán MSE giống nhau (` + '`0.5`' + String.raw`), nhưng:
- Mô hình 1 dùng trọng số hiền hòa ` + '`w1 = [1.0, -1.0]`' + String.raw`: bình phương tổng trọng số $\|w_1\|_2^2 = 1^2 + (-1)^2 = 2$.
- Mô hình 2 nhảy vọt phức tạp ` + '`w2 = [10.0, -10.0]`' + String.raw`: $\|w_2\|_2^2 = 100 + 100 = 200$.
Với mức phạt tiên nghiệm $\lambda = 0.01$: Loss của mô hình 1 là $0.5 + 0.02 = \mathbf{0.52}$, còn mô hình 2 bị phạt thành $0.5 + 2.00 = \mathbf{2.50}$. Mô hình tự động ưu tiên trọng số gọn gàng hơn!`,
          workedExampleCode: `import numpy as np

mse = 0.5
lam = 0.01
for name, w in [("Hiền hòa", [1.0, -1.0]), ("Quá lớn", [10.0, -10.0])]:
    w = np.array(w)
    penalty = lam * np.sum(w ** 2)
    print(f"{name}: ||w||^2={np.sum(w**2):.0f}, phạt={penalty:.2f}, tổng loss={mse + penalty:.2f}")`,
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
      bigPicture:
        'Tại sao có những mô hình làm bài tập ở nhà **đúng 100%** nhưng lúc gặp dữ liệu thật ngoài đời lại đoán sai bét (**Overfitting** — học vẹt), còn mô hình khác quá đơn giản lại chẳng giải quyết được gì (**Underfitting**)? Bài này chia lỗi sai của mọi thuật toán ML thành đúng 3 mảnh ghép rõ ràng (**Bias**, **Variance**, và **Nhiễu gốc**), giúp bạn nhìn vào đồ thị là biết ngay nên thêm dữ liệu hay giảm độ phức tạp mô hình.',
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
          heading: '1. Rủi ro thật (True Risk) vs Rủi ro trên tập học (ERM)',
          realWorld:
            'Giống như đi thi đại học: bạn mong đạt điểm cao trên **toàn bộ đề thi thật tương lai** (*True Risk*), nhưng ở nhà bạn chỉ có một cuốn sách **100 đề thi thử** (*Empirical Risk*).\n\nViệc cố tối ưu thật thấp sai số trên 100 đề thử ở nhà gọi là **ERM (Empirical Risk Minimization)**. Nếu 100 đề thử đại diện tốt và bạn không học vẹt đáp án từng câu, điểm thi thử thấp mới kéo theo điểm thi thật tốt.',
          workedExample:
            String.raw`Xét 1 bài kiểm tra đơn giản trên 3 mẫu thử ` + '`(x, y)`' + String.raw` mà mô hình $h$ đoán sai từng điểm là ` + '`[0.2, -0.4, 0.2]`' + String.raw`:
- Sai số bình phương từng mẫu: ` + '`[0.04, 0.16, 0.04]`' + String.raw`.
- **Empirical Risk $\hat{R}_n(h)$**: lấy trung bình $\frac{0.04 + 0.16 + 0.04}{3} = 0.08$.
Mọi vòng lặp huấn luyện ` + '`fit`' + String.raw` trong ML thực chất đều chỉ tối thiểu hóa con số trung bình $\hat{R}_n(h)$ trên các mẫu đang nắm trong tay.`,
          workedExampleCode: `import numpy as np

errs = np.array([0.2, -0.4, 0.2])
sq_errs = errs ** 2
print("Bình phương từng mẫu:", sq_errs)
print("Empirical Risk R_hat =", np.mean(sq_errs))`,
          body: String.raw`Mục tiêu tối thượng của học có giám sát là tìm giả thuyết $h \in \mathcal{H}$ cực tiểu hóa **Rủi ro kỳ vọng thật (True Risk)** trên phân phối dữ liệu chưa biết $P(X, Y)$. Vì $P(X, Y)$ không quan sát được, thuật toán học thay thế bằng cực tiểu hóa **Rủi ro thực nghiệm (Empirical Risk)** trên $n$ mẫu huấn luyện:`,
          math: String.raw`R(h) = \mathbb{E}_{(x, y)\sim P}[\ell(h(x), y)], \qquad \hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)`,
          intuition:
            'Sai lệch giữa mô hình tối ưu trong tập giả thuyết $\\mathcal{H}$ và quy luật thật gọi là **Approximation Error**; sai lệch do chỉ có $n$ mẫu hữu hạn thay vì toàn bộ phân phối gọi là **Estimation Error**.',
        },
        {
          heading: '2. Phân rã Bias–Variance: Trò chơi bắn cung vào bia',
          realWorld:
            'Hãy tưởng tượng một cung thủ bắn 4 mũi tên vào tâm bia (tương ứng 4 lần huấn luyện mô hình trên 4 bộ dữ liệu ngẫu nhiên hơi khác nhau):\n- **Bias (Độ chệch)** = Tâm ngắm bị lệch khỏi hồng tâm (dù bắn tụ một chỗ nhưng toàn trúng góc trên bên phải — mô hình quá cứng nhắc, *Underfitting*).\n- **Variance (Phương sai)** = Tay cung thủ bị run (mũi tên bay tản mát mỗi phát một nơi theo nhiễu dữ liệu — mô hình quá ngoằn ngoèo, *Overfitting*).\n- **Nhiễu $\\sigma^2$** = Gió thổi ngẫu nhiên không thể tránh khỏi dù cung thủ giỏi đến đâu.',
          workedExample:
            String.raw`Giả sử giá trị đúng tại điểm kiểm tra $x$ là ` + '`f_true = 10.0`' + String.raw` (tạm bỏ qua gió $\sigma^2 = 0.25$). Ta train mô hình 3 lần trên 3 bộ dữ liệu khác nhau và thu được 3 dự đoán ` + '`[11, 13, 15]`' + String.raw`:

- **Bước 1 (Dự đoán trung bình $\bar{f}$)**: $(11 + 13 + 15) / 3 = \mathbf{13.0}$.
- **Bước 2 (Tính Bias²)**: Tâm bắn trung bình lệch khỏi đích ` + '`10.0`' + String.raw` bao nhiêu? $(13.0 - 10.0)^2 = 3^2 = \mathbf{9.0}$.
- **Bước 3 (Tính Variance)**: Các phát bắn tản mát quanh mức trung bình ` + '`13.0`' + String.raw` bao nhiêu?
  $\frac{(11 - 13)^2 + (13 - 13)^2 + (15 - 13)^2}{3} = \frac{4 + 0 + 4}{3} = \mathbf{2.667}$.
- **Bước 4 (Kiểm chứng đẳng thức tổng)**:
  - Tổng 3 mảnh: $\text{Bias}^2 + \text{Var} + \sigma^2 = 9.0 + 2.667 + 0.25 = \mathbf{11.917}$.
  - Tính sai số bình phương trực tiếp từng lần rồi trung bình + gió: $\frac{(11-10)^2 + (13-10)^2 + (15-10)^2}{3} + 0.25 = \frac{1 + 9 + 25}{3} + 0.25 = 11.667 + 0.25 = \mathbf{11.917}$! Khớp nhau chính xác tuyệt đối!`,
          workedExampleCode: `import numpy as np

f_true = 10.0
noise_var = 0.25
preds = np.array([11.0, 13.0, 15.0])  # 3 lần train khác nhau

mean_f = np.mean(preds)
bias_sq = (mean_f - f_true) ** 2
variance = np.var(preds)

print(f"Trung bình = {mean_f:.1f}, Bias^2 = {bias_sq:.3f}, Var = {variance:.3f}")
print(f"Bias^2 + Var + noise = {bias_sq + variance + noise_var:.3f}")
print(f"MSE tính trực tiếp   = {np.mean((preds - f_true) ** 2) + noise_var:.3f}")`,
          codeBridge: {
            title: 'Ước lượng Bias² và Variance từ ma trận M lần chạy',
            pythonLoop: `# preds là bảng M hàng (số lần train), N cột (số điểm test)
M, N = len(preds), len(f_true)
mean_p = [sum(preds[m][j] for m in range(M)) / M for j in range(N)]
bias_sq = sum((mean_p[j] - f_true[j]) ** 2 for j in range(N)) / N
variance = sum(
    sum((preds[m][j] - mean_p[j]) ** 2 for m in range(M)) / M
    for j in range(N)
) / N`,
            numpyWay: `mean_pred = np.mean(preds, axis=0)      # Trung bình dọc theo M lần train
bias_sq = np.mean((mean_pred - f_true) ** 2)
variance = np.mean(np.var(preds, axis=0))`,
            explanation:
              'Nhờ `axis=0` gộp qua toàn bộ `M` lần thử nghiệm, chỉ 3 dòng NumPy đã tách bạch xong thành phần sai số do công thức mô hình yếu (`bias_sq`) khỏi sai số do mô hình nhạy cảm với dữ liệu (`variance`).',
          },
          body: String.raw`Giả sử dữ liệu sinh bởi $y = f(x) + \varepsilon$ với $\mathbb{E}[\varepsilon] = 0$, $\text{Var}[\varepsilon] = \sigma^2$. Ký hiệu $\hat{f}(x) = \hat{f}_{\mathcal{D}}(x)$ là mô hình học được từ tập mẫu ngẫu nhiên $\mathcal{D}$ cỡ $n$, và $\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]$ là dự đoán trung bình qua vô số lần lấy mẫu $\mathcal{D}$ mới. Thêm bớt $\bar{f}(x)$ rồi khai triển kỳ vọng, các số hạng chéo triệt tiêu nhờ $\mathbb{E}[\varepsilon] = 0$ và $\mathbb{E}[\hat{f}(x) - \bar{f}(x)] = 0$:`,
          math: String.raw`\mathbb{E}_{\mathcal{D}, \varepsilon}\!\left[(y - \hat{f}(x))^2\right] = \underbrace{\big(f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\big)^2}_{\text{Bias}^2[\hat{f}(x)]} + \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\big(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\big)^2\right]}_{\text{Var}[\hat{f}(x)]} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}`,
          intuition:
            '**Bias²** đo độ lệch hệ thống do lớp giả thuyết quá đơn giản không diễn tả nổi $f(x)$. **Variance** đo độ nhạy của mô hình khi tập train bị nhiễu hoặc thay đổi đôi chút. **Nhiễu $\\sigma^2$** là cận dưới bất khả kháng của mọi mô hình.',
          pitfall:
            'Nhầm lẫn giữa độ nhạy của tập kiểm thử cố định với phương sai qua các tập train $\\mathcal{D}$. Trong thực nghiệm, muốn đo trực tiếp Bias và Variance cần mô phỏng lấy mẫu lại (bootstrap hoặc Monte Carlo nhiều tập $\\mathcal{D}_1, \\dots, \\mathcal{D}_M$).',
        },
        {
          heading: '3. Điều chuẩn (Regularization) & Ghi chú về Double Descent',
          realWorld:
            'Nếu tay cung thủ bị run quá mạnh (**Variance cao / Overfitting**), ta làm gì? Có 2 cách: (1) **Cho tập bắn nhiều hơn** (tăng cỡ dữ liệu train $n$), hoặc (2) **Gắn thanh giữ ổn định lên cánh cung** (**Regularization**) — chấp nhận kéo ngắm hơi lệch đi một chút cực nhỏ (tăng Bias nhẹ) đổi lại tay đỡ run đi rất nhiều (hạ mạnh Variance).',
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
      bigPicture:
        'Hai câu hỏi sống còn trước khi đem mô hình ra thực tế: **(1) Cần thu thập bao nhiêu mẫu dữ liệu** thì mới đáng tin cậy? và **(2) Làm sao biết chắc điểm kiểm thử cao là năng lực thật chứ không phải do ăn gian (Data Leakage)?** Bài này giải thích định lượng từ phép đếm mẫu tới kỷ luật chia tập Train/Validation/Test không bao giờ bị hớ.',
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
          heading: '1. Bất đẳng thức Hoeffding: Cần bao nhiêu dữ liệu là đủ?',
          realWorld:
            'Nếu bạn tung đồng xu `5` lần thấy ra `4` lần Ngửa (`80%`), chẳng ai tin đồng xu đó lệch `80%` vì số lần thử quá ít! Nhưng nếu tung `2.000` lần mà vẫn ra `80%`, gần như chắc chắn nó lệch thật.\n\n**Bất đẳng thức Hoeffding** trả lời chính xác câu hỏi: nếu bạn muốn sai lệch giữa điểm thi thử ở nhà và điểm thật ngoài đời **không vượt quá $\\epsilon = 5\\%$**, với độ chắc chắn **$\\ge 95\\%$**, thì bạn phải thu thập tối thiểu bao nhiêu mẫu $n$?',
          workedExample:
            String.raw`Thử thay số trực tiếp vào công thức khi phải thử ` + '`100`' + String.raw` cấu hình mô hình khác nhau ($|\mathcal{H}| = 100$), muốn sai lệch nhỏ hơn $\epsilon = 0.05$ (5%) với độ tin cậy $1 - \delta = 0.95$ ($\delta = 0.05$):

$$n \ge \frac{1}{2 \cdot 0.05^2} \ln\left(\frac{2 \times 100}{0.05}\right) = 200 \cdot \ln(4000) \approx 200 \times 8.294 \approx \mathbf{1{,}659\text{ mẫu}}$$

**Điều thú vị**: Nếu thử gấp **10 lần** số cấu hình ($1{,}000$ thay vì $100$), số mẫu cần thiết chỉ tăng nhẹ từ ` + '`1,659`' + String.raw` lên ` + '`2,119`' + String.raw` (tăng theo $\ln |\mathcal{H}|$ rất chậm, chứ không nhân 10!). Nhưng nếu muốn giảm sai số xuống một nửa ($\epsilon = 0.025$), cần dữ liệu tăng **gấp 4 lần** (` + '`1 / eps^2`' + String.raw`)!`,
          workedExampleCode: `import math

def min_samples(num_models, eps=0.05, delta=0.05):
    return math.ceil(math.log(2 * num_models / delta) / (2 * eps ** 2))

print("H=100,   eps=0.05  -> n >=", min_samples(100, eps=0.05))
print("H=1000,  eps=0.05  -> n >=", min_samples(1000, eps=0.05))
print("H=100,   eps=0.025 -> n >=", min_samples(100, eps=0.025))`,
          body: String.raw`Với hàm mất mát $0\text{--}1$, cho bất kỳ giả thuyết cố định $h \in \mathcal{H}$, bất đẳng thức Hoeffding giới hạn xác suất sai lệch giữa rủi ro thật $R(h)$ và rủi ro thực nghiệm $\hat{R}_n(h)$ trên $n$ mẫu i.i.d. Áp dụng Union Bound qua toàn bộ $k = |\mathcal{H}|$ giả thuyết, với xác suất ít nhất $1 - \delta$ (khung Probably Approximately Correct — PAC), mọi $h \in \mathcal{H}$ đều thỏa mãn hội tụ đều:`,
          math: String.raw`|R(h) - \hat{R}_n(h)| \le \sqrt{\frac{\ln(2|\mathcal{H}| / \delta)}{2n}} \quad \Longleftrightarrow \quad n \ge \frac{1}{2\epsilon^2} \ln\frac{2|\mathcal{H}|}{\delta}`,
          intuition:
            'Đáng chú ý, số lượng mẫu $n$ chỉ tăng theo $\\ln |\\mathcal{H}|$ (tốc độ logarit), nên việc xét hàng tỉ bộ tham số có thể được kiểm soát bằng lượng dữ liệu khả thi. Tuy nhiên nếu tập giả thuyết vô hạn ($\\mathcal{H}$ liên tục), chặn chứa $\\ln |\\mathcal{H}|$ trở nên vô nghĩa.',
        },
        {
          heading: '2. Shattering & Chiều VC (VC-Dimension)',
          realWorld:
            'Một đường thẳng vẽ trên tờ giấy 2D có **vô số** vị trí/độ dốc khác nhau, nên ta không thể đếm $|\\mathcal{H}|$. Vậy làm sao đo "độ linh hoạt" của đường thẳng?\n\nTa chơi trò **phá vỡ (Shatter)**: đặt `m` chấm tròn lên giấy, bạn có dám chắc **dù tôi tô màu Xanh/Đỏ tùy ý cho từng chấm**, bạn vẫn luôn kẻ được **1 đường thẳng duy nhất** tách sạch Đỏ ra khỏi Xanh?\n- Với `3` chấm bất kỳ (không thẳng hàng): **Luôn tách được** cả $2^3 = 8$ kiểu tô màu!\n- Với `4` chấm đặt 4 góc hình vuông: hễ tô chéo kiểu chữ X (`[Góc 1, 3 = Xanh]`, `[Góc 2, 4 = Đỏ]`, giống hàm XOR) là **chịu thua**!\n\nCon số tối đa tách được trong mọi kiểu tô màu chính là **Chiều VC** ($\\text{VC} = 3$ cho đường thẳng 2D, tổng quát $\\text{VC} = d + 1$).',
          body: String.raw`Để đo sức biểu diễn của một lớp giả thuyết vô hạn $\mathcal{H}$, Vapnik và Chervonenkis định nghĩa **Chiều VC ($\text{VC}(\mathcal{H})$)** là số lượng điểm tối đa mà $\mathcal{H}$ có thể **phá vỡ (shatter)** — tức chia cắt đúng cho mọi tổ hợp $2^m$ phép gán nhãn nhị phân trên $m$ điểm đó (tồn tại ít nhất một cách sắp xếp $m$ điểm để thực hiện điều này).

Kết quả kinh điển: Lớp siêu phẳng phân tách tuyến tính trên $\mathbb{R}^d$ (với bias) có $\text{VC}(\mathcal{H}) = d + 1$. Theo Định lý Vapnik–Chervonenkis, với xác suất $\ge 1 - \delta$, độ chênh tổng quát hóa được chặn bởi:`,
          math: String.raw`R(\hat{h}_{\text{ERM}}) \le R(h^*) + O\!\left(\sqrt{\frac{d_{\text{VC}}\ln(n/d_{\text{VC}}) + \ln(1/\delta)}{n}}\right)`,
          intuition:
            'Muốn giảm khoảng cách tổng quát hóa khi bài toán giữ nguyên kích thước mẫu $n$, ta cần kiểm soát chiều VC hiệu dụng của mô hình (ví dụ qua lựa chọn đặc trưng hoặc ràng buộc chuẩn tham số $\\|w\\|_2$).',
        },
        {
          heading: '3. Kỷ luật thực nghiệm & Chống rò rỉ dữ liệu (Data Leakage)',
          realWorld:
            'Sai lầm kinh điển khiến hàng loạt bài thực nghiệm hỏng hoàn toàn: chuẩn hóa (`StandardScaler`) hoặc chọn lọc đặc trưng **trên cả bảng dữ liệu gốc** rồi mới đem chia Train / Test!\n\nNếu bạn để giá trị cực lớn của tập Test tham gia vào tính trung bình $\\mu$ và độ lệch chuẩn $\\sigma$ lúc biến đổi tập Train, thì thực chất mô hình **đã lén nhìn thấy trước thông tin tương lai của đề thi Test**, tạo ra kết quả ảo trên giấy.',
          workedExample:
            String.raw`Giả sử tập Train chỉ có 3 số ` + '`[1, 2, 3]`' + String.raw` ($\mu_{\text{tr}} = 2$, $\sigma_{\text{tr}} \approx 0.82$), còn tập Validation có một giá trị ngoại lệ cực lớn ` + '`[98]`' + String.raw`:

- **Cách CHUẨN (Không rò rỉ)**: chỉ tính $\mu, \sigma$ từ Train ` + '`[1, 2, 3]`' + String.raw`, nên ` + '`Z_train = [-1.22, 0.0, +1.22]`' + String.raw` bất biến mãi mãi.
- **Cách SAI (Rò rỉ từ Val sang Train)**: gộp ` + '`[1, 2, 3, 98]`' + String.raw` tính chung $\mu = 26$, $\sigma = 41.6$ $\implies$ ` + '`Z_train`' + String.raw` biến thành ` + '`[-0.60, -0.58, -0.55]`' + String.raw`! Toàn bộ dữ liệu Train bị méo mó chỉ vì một số ở tập Test!`,
          workedExampleCode: `import numpy as np

X_tr = np.array([1.0, 2.0, 3.0])
X_va = np.array([98.0])

# ĐÚNG: Chỉ lấy mu, sd từ Train
Z_tr_clean = (X_tr - X_tr.mean()) / X_tr.std()

# SAI: Gộp cả Val rồi mới lấy mu, sd
X_all = np.concatenate([X_tr, X_va])
Z_tr_leaked = (X_tr - X_all.mean()) / X_all.std()

print("Z_train SẠCH:", np.round(Z_tr_clean, 2))
print("Z_train RÒ RỈ:", np.round(Z_tr_leaked, 2))`,
          codeBridge: {
            title: 'Luôn fit thống kê (mu, sd) DUY NHẤT trên X_train',
            pythonLoop: `# SAI: tính mean/std trên toàn bộ dữ liệu X trước khi chia!
# mu_all = X.mean(axis=0); sd_all = X.std(axis=0)  <-- RÒ RỈ!

# ĐÚNG:
mu_tr = X_train.mean(axis=0)
sd_tr = X_train.std(axis=0)`,
            numpyWay: `mu = X_train.mean(axis=0)
sd = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))
Z_train = (X_train - mu) / sd
Z_val   = (X_val   - mu) / sd   # Dùng lại (mu, sd) của Train!`,
            explanation:
              'Tập `X_val` và `X_test` **không bao giờ** được phép tự gọi `.mean()` hay `.std()` riêng hoặc tham gia tính toán trước khi biến đổi tập `X_train`.',
          },
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
