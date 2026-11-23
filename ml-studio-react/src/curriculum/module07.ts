import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_07: GradModule = {
  id: 7,
  title: 'Deep Learning & Thị giác Máy tính',
  englishTitle: 'Deep Neural Networks & Computer Vision',
  color: '#38bdf8',
  status: 'ready',
  overview:
    'Biểu diễn ảnh số và các kỹ thuật xử lý ảnh cổ điển (Filtering, Histogram Equalization, Edge Detection, OpenCV); MLP, Định lý Xấp xỉ Phổ quát và Reverse-mode Autodiff (Backpropagation); kỹ thuật huấn luyện mạng rất sâu (Xavier/He Init, BatchNorm/LayerNorm, Dropout & Residual Connections) và kiến trúc CNN đến Vision Transformer.',
  outcomes: [
    'Thực hành tiền xử lý ảnh số từ nền tảng đại số ma trận: tích chập không gian 2D, lọc Sobel/Gaussian và cân bằng Histogram qua CDF.',
    'Cài đặt cơ chế Reverse-mode Automatic Differentiation (tính toán đồ thị và lan truyền ngược) bằng NumPy và kiểm chứng bằng sai phân hữu hạn.',
    'Phân tích bảo toàn phương sai tín hiệu theo độ sâu (Xavier/He Initialization), Batch Normalization (`train` vs `eval`) và Skip Connection của ResNet.',
    'Tính toán Receptive Field hiệu dụng qua các lớp Conv/Strided/Dilated và hiểu bước chuyển từ Inductive Bias của CNN sang Vision Transformer (ViT).',
  ],
  sources: [
    'Thư mục 05_Thi_giac_May_tinh_Computer_Vision (C1–C3, cv2cheatsheet_Python.pdf)',
    'Stanford CS229 (Deep Learning Notes)',
    'Kevin P. Murphy — Probabilistic Machine Learning (Ch. 13–14)',
  ],
  units: [
    {
      id: 'm7-u1',
      moduleId: 7,
      order: 1,
      title: 'Biểu diễn ảnh & Xử lý ảnh cổ điển (Filtering, Histogram, OpenCV)',
      summary:
        'Cầu nối từ Thị giác Máy tính cổ điển đến Deep Vision: ảnh số như tensor $(H, W, C)$, không gian màu BGR/RGB/HSV, tăng cường độ tương phản bằng Histogram Equalization (CDF), và bộ lọc không gian 2D (Box, Gaussian, Median, Sobel/Laplacian).',
      outcomes: [
        'Phân biệt đúng quy ước thứ tự kênh màu (BGR của OpenCV vs RGB), tránh lỗi tràn số uint8 khi xử lý điểm ảnh.',
        'Cài đặt phép cân bằng Histogram dựa trên hàm phân phối tích lũy (CDF) chuẩn hóa về dải $[0, 255]$.',
        'Cài đặt toán tử tương quan chéo / tích chập 2D hợp lệ (valid 2D convolution) cho làm mượt và dò biên Sobel.',
      ],
      references: [
        {
          source: '05_Thi_giac_May_tinh_Computer_Vision/C1-CVApp_Introduction.pdf & C2-1CVApp_Fundamental.pdf',
          section: 'Image Representation, Color Spaces & Pixel Coordinates',
          note: 'Định nghĩa lưới ảnh rời rạc I(y, x), kênh màu RGB/HSV/Grayscale và phép biến đổi điểm.',
        },
        {
          source: '05_Thi_giac_May_tinh_Computer_Vision/C3-Image Enhancement.pdf',
          section: 'Histogram Equalization, Spatial Filtering & Edge Detection',
          note: 'Biến đổi điểm dựa trên CDF, lọc tuyến tính (Box, Gaussian, Sobel) và lọc phi tuyến (Median).',
        },
        {
          source: '05_Thi_giac_May_tinh_Computer_Vision/cv2cheatsheet_Python.pdf',
          section: 'OpenCV Python Conventions (BGR Order, uint8 Casting)',
          note: 'Chuẩn thao tác tensor NumPy tương ứng với hàm thư viện cv2.',
        },
      ],
      sections: [
        {
          heading: '1. Tensor ảnh số & Cân bằng Histogram qua CDF',
          body: String.raw`Ảnh số xám kích thước $H \times W$ là ma trận cường độ sáng $I \in \{0,\dots,255\}^{H \times W}$ (trong khi ảnh màu có thêm trục kênh $C = 3$; lưu ý ` + '`cv2.imread`' + String.raw` mặc định nạp thứ tự **BGR** thay vì RGB). Để trải đều dải động cho ảnh thiếu/quá sáng, **Histogram Equalization** dùng hàm phân phối tích lũy $\text{CDF}(v) = \sum_{k=0}^v h(k)$ ánh xạ tuyến tính về $[0, 255]$:`,
          math: String.raw`s_v = \text{round}\!\left(\frac{\text{CDF}(v) - \text{CDF}_{\min}}{H W - \text{CDF}_{\min}} \cdot 255\right) \quad (\text{nếu } H W > \text{CDF}_{\min})`,
          intuition:
            'Trong trường hợp biến liên tục, biến đổi tích lũy $Y = F_X(X)$ luôn cho biến $Y \\sim \\text{Uniform}(0, 1)$. Trên lưới rời rạc 8-bit, công thức trên xấp xỉ biến đổi đó để các mức xám có nhiều điểm ảnh được giãn rộng khoảng cách hơn.',
          pitfall:
            'Phép cộng trừ trên mảng `np.uint8` dùng số học modulo 256 (ví dụ `np.uint8(200) + np.uint8(100) == 44`). Trước khi lọc không gian hay chuẩn hóa, luôn ép kiểu sang `np.float32` hoặc `np.float64`.',
        },
        {
          heading: '2. Lọc không gian 2D: Cross-Correlation, Gaussian & Sobel',
          body: String.raw`Lọc tuyến tính trượt một kernel $K \in \mathbb{R}^{k_h \times k_w}$ qua mọi vùng lân cận ảnh. Trong xử lý ảnh cổ điển và Deep Learning (` + '`Conv2d`' + String.raw`), toán tử thực tế thường là **tương quan chéo (Cross-Correlation)** không lật ngược kernel:`,
          math: String.raw`(I \star K)[i, j] = \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} I[i + u,\, j + v]\, K[u, v]`,
          intuition:
            'Kernel Gaussian làm mượt đẳng hướng và có tính tách biến hạng 1 ($K_{2D} = g g^\\top$), giảm chi phí từ $O(k^2)$ xuống $O(2k)$. Kernel đạo hàm bậc nhất như **Sobel** ($[-1,0,1]$ kết hợp làm mượt $[1,2,1]^\\top$) khuếch đại biên cường độ theo phương $x, y$, làm tiền đề cho Canny Edge Detector.',
        },
        {
          heading: '3. Từ đặc trưng thiết kế tay (Hand-crafted) sang học biểu diễn',
          body: String.raw`Trong thị giác cổ điển, kỹ sư phải tự chọn tuần tự bộ lọc Gaussian khử nhiễu, Sobel lấy gradient ảnh, HOG hoặc SIFT rồi đưa vào SVM. Mạng CNN hiện đại giữ nguyên cấu trúc phép tính trượt $I \star K$, nhưng biến chính các trọng số kernel $K$ thành tham số khả vi được học trực tiếp từ dữ liệu qua lan truyền ngược.`,
          pitfall:
            'Lọc tuyến tính thông thấp (Gaussian/Box) làm nhòe cạnh và kém hiệu quả với nhiễu xung đột biến (Salt-and-Pepper noise); khi đó bắt buộc phải dùng bộ lọc phi tuyến **Median Filter**.',
        },
      ],
      simulationId: 'm7-u1',
      exercises: [
        {
          id: 'm7-u1-conv2d-hist',
          title: '2D Valid Cross-Correlation & Cân bằng Histogram (CDF)',
          prompt:
            'Cài đặt hai hàm thuần NumPy:\n1. `conv2d_valid(image, kernel)`: thực hiện tương quan chéo 2D hợp lệ (không padding) trả về mảng float64 kích thước `(H - kh + 1, W - kw + 1)`.\n2. `hist_equalize_uint8(gray)`: nhận mảng ảnh xám 2D giá trị nguyên trong `[0, 255]`, tính histogram 256 mức và CDF (`np.cumsum`). Nếu ảnh đồng nhất (`cdf_min == H * W`), giữ nguyên ảnh `uint8`; ngược lại áp dụng công thức Lookup Table chuẩn hóa làm tròn về `[0, 255]` kiểu `np.uint8`.',
          hint: 'Trong `hist_equalize_uint8`: `arr = np.asarray(gray, dtype=np.uint8)`, `hist = np.bincount(arr.ravel(), minlength=256)`, `cdf = np.cumsum(hist)`, `cdf_min = cdf[cdf > 0][0]`, `denom = arr.size - cdf_min`. Nếu `denom == 0` trả về `arr.copy()`. Ngược lại `lut = np.round(np.maximum(0, cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)` rồi trả `lut[arr]`.',
          setup:
            COMMON_SETUP +
            String.raw`
_img_cv = np.array([
    [10., 10., 50., 50.],
    [10., 10., 50., 50.],
    [10., 10., 50., 50.],
    [10., 10., 50., 50.],
])
_sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_low_contrast = np.array([[50, 51, 52], [52, 53, 55], [50, 54, 55]], dtype=np.uint8)
_const_img = np.full((3, 3), 120, dtype=np.uint8)
`,
          starter: String.raw`def conv2d_valid(image, kernel):
    raise NotImplementedError

def hist_equalize_uint8(gray):
    raise NotImplementedError
`,
          solution: String.raw`def conv2d_valid(image, kernel):
    img = np.asarray(image, dtype=float)
    ker = np.asarray(kernel, dtype=float)
    H, W = img.shape
    kh, kw = ker.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    out = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * ker)
    return out

def hist_equalize_uint8(gray):
    arr = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    pos = cdf[cdf > 0]
    if pos.size == 0:
        return arr.copy()
    cdf_min = int(pos[0])
    denom = arr.size - cdf_min
    if denom <= 0:
        return arr.copy()
    lut = np.round(np.maximum(0, cdf - cdf_min) / float(denom) * 255.0).clip(0, 255).astype(np.uint8)
    return lut[arr]
`,
          test: String.raw`def _run_checks():
    _ref_edge = np.array([[160., 160.], [160., 160.]])
    c1 = _check('conv2d_valid Sobel phat hien dung bien dung', lambda: (
        lambda out: (
            out.shape == (2, 2) and np.allclose(out, _ref_edge, atol=1e-8),
            f'got={out.tolist()}'
        )
    )(conv2d_valid(_img_cv, _sobel_x)))
    c2 = _check('hist_equalize_uint8 gian dai dong tu [50..55] ra [0..255]', lambda: (
        lambda eq: (
            eq.shape == (3, 3) and eq.dtype == np.uint8 and int(eq.min()) == 0 and int(eq.max()) == 255,
            f'dtype={eq.dtype}, min={eq.min()}, max={eq.max()}'
        )
    )(hist_equalize_uint8(_low_contrast)))
    c3 = _check('hist_equalize_uint8 an toan tren anh dong nhat khong chia cho 0', lambda: (
        lambda eq: (
            eq.shape == (3, 3) and np.array_equal(eq, _const_img),
            f'got={eq.tolist()}'
        )
    )(hist_equalize_uint8(_const_img)))
    _report('m7-u1-conv2d-hist', c1, c2, c3)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Về mặt toán học, làm thế nào dùng ma trận toeplitz / thủ thuật `im2col` để biến phép `conv2d_valid` nhiều kênh từ 4 vòng lặp thành một phép nhân ma trận BLAS duy nhất?',
          guidance:
            'Trải mỗi cửa sổ trượt $C_{in} \\times k_h \\times k_w$ thành một cột của ma trận $X_{\\text{col}}$ kích thước $(C_{in} k_h k_w) \\times (H_{\\text{out}} W_{\\text{out}})$, rồi nhân với ma trận trọng số duỗi phẳng $W_{\\text{row}} \\in \\mathbb{R}^{C_{\\text{out}} \\times (C_{\\text{in}} k_h k_w)}$. Thuật toán này cho phép GPU dùng nhân ma trận GEMM tốc độ tối đa, đánh đổi tăng bộ nhớ tạm.',
        },
        {
          question:
            'Tại sao các phương pháp phát hiện cạnh hiện đại hoặc tiền xử lý ảnh y khoa dùng CLAHE (Contrast Limited Adaptive Histogram Equalization) thay vì Global Histogram Equalization?',
          guidance:
            'Global Histogram Equalization chỉ tính 1 CDF toàn cục nên vùng nền tối lớn có thể khuếch đại nhiễu nền quá mức hoặc làm cháy chi tiết cục bộ. CLAHE chia ảnh thành lưới ô cục bộ, cắt ngưỡng (clip limit) trên histogram trước khi tính CDF và nội suy song tuyến tính qua biên ô.',
        },
      ],
    },
    {
      id: 'm7-u2',
      moduleId: 7,
      order: 2,
      title: 'MLP, Universal Approximation & Reverse-mode Autodiff (Backprop)',
      summary:
        'Kiến trúc mạng nơ-ron truyền thẳng nhiều lớp (MLP), bản chất hình học của Định lý Xấp xỉ Phổ quát với ReLU piecewise linear, và động cơ cốt lõi của Deep Learning: vi phân tự động chế độ ngược (Reverse-mode Autodiff / Backpropagation) trên đồ thị tính toán.',
      outcomes: [
        'Giải thích tại sao kích hoạt phi tuyến là điều kiện cần để MLP không suy biến thành ánh xạ tuyến tính duy nhất.',
        'Phân tích ưu thế phức tạp tính toán $O(1)$ lượt của Reverse-mode Autodiff so với $O(P)$ lượt của Forward-mode khi hàm mất mát vô hướng.',
        'Cài đặt lan truyền ngược ma trận cho MLP 2 tầng với Softmax Cross-Entropy và một mini Autograd engine (`Value`) bằng NumPy/Python.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Deep Learning Notes (Backpropagation, Vectorized Gradients & Softmax Loss)',
          note: 'Dẫn xuất quy tắc chuỗi ma trận và cách kiểm tra gradient bằng sai phân hữu hạn.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§13.1–13.4 (Feedforward Neural Networks, Universal Approximation & Autodiff)',
          note: 'Định lý Cybenko/Hornik, đồ thị tính toán DAG và Reverse-mode VJP.',
        },
      ],
      sections: [
        {
          heading: '1. MLP & Định lý Xấp xỉ Phổ quát',
          body: String.raw`Một tầng tuyến tính đơn lẻ chỉ thực hiện biến đổi afin $x \mapsto Wx + b$. Nếu chồng nhiều tầng không có kích hoạt phi tuyến, tích của chúng vẫn chỉ là một phép biến đổi tuyến tính duy nhất ($W_2 W_1 x$). Với hàm kích hoạt phi tuyến bị chặn hoặc không đa thức (như $\text{ReLU}(z) = \max(0, z)$ hay GELU), mạng MLP 1 tầng ẩn có thể xấp xỉ bất kỳ hàm liên tục nào trên tập compact với sai số $\varepsilon > 0$ tùy ý (Cybenko / Hornik):`,
          math: String.raw`f(x) = W_2\,\sigma(W_1 x + b_1) + b_2, \qquad \sigma_{\text{ReLU}}(z) = \max(0, z)`,
          intuition:
            'Mỗi nơ-ron ReLU bẻ không gian đầu vào dọc theo siêu phẳng $w_j^\\top x + b_j = 0$, ghép lại thành hàm tuyến tính từng khúc (piecewise linear). Mạng sâu xếp chồng các phép gấp nếp giúp số vùng tuyến tính tăng theo hàm mũ độ sâu thay vì chỉ tuyến tính theo độ rộng.',
        },
        {
          heading: '2. Đồ thị tính toán & Reverse-mode Autodiff (Backpropagation)',
          body:
            String.raw`Xét đồ thị tính toán có hướng không chu trình (DAG) đi từ biến đầu vào và tham số tới hàm mất mát vô hướng $\mathcal{L} \in \mathbb{R}$. **Reverse-mode Autodiff** gồm hai pha:
- **Forward pass**: tính giá trị các nút theo thứ tự topo và lưu kích hoạt trung gian.
- **Backward pass**: duyệt ngược thứ tự topo, lan truyền gradient $\bar{v} = \frac{\partial \mathcal{L}}{\partial v}$ qua tích Jacobian–vector (VJP). Khi nút $u$ tỏa ra nhiều nhánh (fan-out), gradient theo quy tắc chuỗi đa biến **phải cộng dồn (` +
            '`+=`' +
            String.raw`)**:`,
          math: String.raw`\frac{\partial \mathcal{L}}{\partial u} = \sum_{v \in \text{children}(u)} \frac{\partial \mathcal{L}}{\partial v} \frac{\partial v}{\partial u}`,
          pitfall:
            'Lỗi phổ biến nhất khi cài đặt Autodiff từ đầu là ghi đè `=` thay vì cộng dồn `+=` cho `.grad` tại nút dùng chung (ví dụ biểu thức `z = a * b + a * a`), khiến mất hoàn toàn đóng góp đạo hàm từ một hoặc nhiều nhánh đồ thị.',
        },
        {
          heading: '3. Dẫn xuất ma trận MLP 2 tầng & Softmax Cross-Entropy',
          body: String.raw`Với mini-batch $X \in \mathbb{R}^{N \times D}$, nhãn one-hot $Y \in \{0,1\}^{N \times C}$, kích hoạt ẩn $Z_1 = XW_1 + b_1,\; A_1 = \text{ReLU}(Z_1)$, logits $O = A_1 W_2 + b_2$ và mất mát trung bình $\mathcal{L} = -\frac{1}{N}\sum_{i,c} Y_{ic}\log p_{ic}$:`,
          math: String.raw`\nabla_O \mathcal{L} = \frac{1}{N}(P - Y), \qquad \nabla_{W_2}\mathcal{L} = A_1^\top \nabla_O\mathcal{L}, \qquad \nabla_{Z_1}\mathcal{L} = \big((\nabla_O\mathcal{L}) W_2^\top\big) \odot \mathbb{I}[Z_1 > 0]`,
        },
      ],
      simulationId: 'm7-u2',
      exercises: [
        {
          id: 'm7-u2-mlp-backprop',
          title: 'Vectorized 2-Layer ReLU MLP & Finite-Difference Check',
          prompt:
            'Cài đặt hàm `mlp_forward_backward(X, y_onehot, W1, b1, W2, b2)`:\n- Kiến trúc: `Z1 = X @ W1 + b1` $\\to$ `A1 = np.maximum(0.0, Z1)` $\\to$ `logits = A1 @ W2 + b2` $\\to$ Softmax Cross-Entropy ổn định bằng dịch chuyển cực đại (lấy trung bình trên $N$ mẫu).\n- Trả về `(loss, grads)` với `loss` là `float` và `grads` là dict `{"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}` có cùng shape với từng tham số.',
          hint: 'Tính `shifted = logits - logits.max(axis=1, keepdims=True)`, `log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))`, `log_probs = shifted - log_sum_exp`, `probs = np.exp(log_probs)`. Gradient theo `logits` là `dlogits = (probs - y_onehot) / len(X)`. Nhớ nhân mặt nạ `(Z1 > 0)` khi truyền ngược qua ReLU.',
          setup:
            COMMON_SETUP +
            String.raw`
_rng_mlp = np.random.default_rng(42)
_X_mlp = np.array([[1.2, -0.7, 0.5], [-0.9, 1.3, 0.8], [0.4, 0.9, -1.1], [-1.3, -0.6, 1.0]])
_y_mlp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])
_W1_mlp = np.array([[0.4, -0.3, 0.2, 0.5], [-0.5, 0.6, -0.2, 0.3], [0.3, 0.2, 0.7, -0.4]])
_b1_mlp = np.array([0.1, -0.1, 0.2, 0.05])
_W2_mlp = np.array([[0.5, -0.2, 0.1], [-0.3, 0.6, 0.2], [0.2, -0.4, 0.5], [0.1, 0.3, -0.6]])
_b2_mlp = np.array([0.0, 0.1, -0.1])
`,
          starter: String.raw`def mlp_forward_backward(X, y_onehot, W1, b1, W2, b2):
    raise NotImplementedError
`,
          solution: String.raw`def mlp_forward_backward(X, y_onehot, W1, b1, W2, b2):
    X = np.asarray(X, dtype=float)
    y_onehot = np.asarray(y_onehot, dtype=float)
    N = X.shape[0]
    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    logits = A1 @ W2 + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    sum_exp = np.sum(exp_s, axis=1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)
    probs = exp_s / sum_exp
    loss = float(-np.sum(y_onehot * log_probs) / N)

    dlogits = (probs - y_onehot) / N
    dW2 = A1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    dA1 = dlogits @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return loss, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
`,
          test: String.raw`def _run_checks():
    def _shape_chk():
        loss, grads = mlp_forward_backward(_X_mlp, _y_mlp, _W1_mlp, _b1_mlp, _W2_mlp, _b2_mlp)
        return (
            grads['W1'].shape == _W1_mlp.shape and grads['b1'].shape == _b1_mlp.shape
            and grads['W2'].shape == _W2_mlp.shape and grads['b2'].shape == _b2_mlp.shape,
            f'loss={loss:.4f}'
        )
    c1 = _check('Shape cac gradient khop tham so', _shape_chk)
    def _num_check():
        _, grads = mlp_forward_backward(_X_mlp, _y_mlp, _W1_mlp, _b1_mlp, _W2_mlp, _b2_mlp)
        eps = 1e-6
        params = {'W1': _W1_mlp, 'b1': _b1_mlp, 'W2': _W2_mlp, 'b2': _b2_mlp}
        max_err = 0.0
        for name, arr in params.items():
            num_g = np.zeros_like(arr)
            for idx in np.ndindex(*arr.shape):
                old = arr[idx]
                arr[idx] = old + eps
                lp, _ = mlp_forward_backward(_X_mlp, _y_mlp, _W1_mlp, _b1_mlp, _W2_mlp, _b2_mlp)
                arr[idx] = old - eps
                lm, _ = mlp_forward_backward(_X_mlp, _y_mlp, _W1_mlp, _b1_mlp, _W2_mlp, _b2_mlp)
                arr[idx] = old
                num_g[idx] = (lp - lm) / (2.0 * eps)
            max_err = max(max_err, float(np.max(np.abs(num_g - grads[name]))))
        return max_err < 1e-7, f'max_grad_diff={max_err:.2e}'
    c2 = _check('Kiem chung sai phan huu han < 1e-7 tren moi tham so', _num_check)
    _report('m7-u2-mlp-backprop', c1, c2)
_run_checks()
`,
        },
        {
          id: 'm7-u2-micro-autodiff',
          title: 'Xây dựng Scalar Autograd Engine (DAG + Topological Sort)',
          prompt:
            'Hoàn thiện lớp vô hướng `Value` hỗ trợ xây dựng đồ thị tính toán và lan truyền ngược:\n- Thuộc tính `.data` (float), `.grad` (khởi tạo `0.0`), `._backward` (hàm đóng), `._prev` (set các nút cha).\n- Hỗ trợ phép toán `__add__`, `__radd__`, `__mul__`, `__rmul__` (với `Value` hoặc `int`/`float`) và phương thức `.relu()`.\n- Phương thức `backward(self)`: sắp xếp topo DFS từ nút hiện tại, gán `self.grad = 1.0`, rồi gọi `node._backward()` theo thứ tự ngược topo. Lưu ý cộng dồn `+=` trong mọi `_backward`!',
          hint: 'Trong `_mul`: `out.grad` đóng góp `other.data * out.grad` vào `self.grad` và `self.data * out.grad` vào `other.grad`. Trong `relu()`: `out = Value(max(0.0, self.data), (self,), "ReLU")`, và `_backward`: `self.grad += (out.data > 0) * out.grad`. Trong `backward()`: dùng DFS hậu thứ tự (`visited.add(v)` trước khi đệ quy con rồi `topo.append(v)`).',
          setup: COMMON_SETUP,
          starter: String.raw`class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def relu(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
`,
          solution: String.raw`class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), 'ReLU')
        def _backward():
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
`,
          test: String.raw`def _run_checks():
    def _chk_fanout():
        a = Value(3.0)
        b = Value(-2.0)
        z = (a * b + a * a + 1.5).relu()
        z.backward()
        # z = ab + a^2 + 1.5 = -6 + 9 + 1.5 = 4.5 > 0 -> dz/da = b + 2a = 4.0, dz/db = a = 3.0
        ok = abs(z.data - 4.5) < 1e-9 and abs(a.grad - 4.0) < 1e-9 and abs(b.grad - 3.0) < 1e-9
        return ok, f'z={z.data}, da={a.grad}, db={b.grad}'
    c1 = _check('Fan-out cong don gradient dung (a*b + a*a)', _chk_fanout)
    def _chk_relu_neg():
        x = Value(-4.0)
        y = (2.0 * x + 1.0).relu()
        y.backward()
        return abs(y.data) < 1e-9 and abs(x.grad) < 1e-9, f'y={y.data}, dx={x.grad}'
    c2 = _check('ReLU chan gradient ve 0 o nhanh am', _chk_relu_neg)
    _report('m7-u2-micro-autodiff', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao mạng Deep Learning hiện đại dùng Reverse-mode Autodiff thay vì Forward-mode hay Sai phân hữu hạn?',
          guidance:
            'Hàm mất mát $f: \\mathbb{R}^P \\to \\mathbb{R}$ có $P \\approx 10^8$ tham số đầu vào và chỉ $1$ đầu ra vô hướng. Sai phân hữu hạn và Forward-mode cần $\\Omega(P)$ lượt đánh giá hàm, còn Reverse-mode tính trọn vẹn toàn bộ $P$ đạo hàm riêng chỉ sau 1 lượt Forward và 1 lượt Backward (chi phí $\\approx 2\\times$ đến $3\\times$ lượt Forward).',
        },
        {
          question:
            'Hạn chế trí nhớ lớn nhất của Reverse-mode Autodiff là gì và kỹ thuật Gradient Checkpointing giải quyết ra sao?',
          guidance:
            'Để tính VJP ở lượt ngược, đồ thị phải lưu toàn bộ tensor kích hoạt trung gian từ lượt xuôi ($O(L)$ bộ nhớ theo độ sâu). Gradient Checkpointing chỉ lưu mốc kích hoạt tại vài lớp cách quãng và tính lại lượt xuôi cục bộ trong lúc Backward, đánh đổi $O(\\sqrt{L})$ bộ nhớ lấy thêm khoảng 33% thời gian tính toán.',
        },
      ],
    },
    {
      id: 'm7-u3',
      moduleId: 7,
      order: 3,
      title: 'Huấn luyện mạng sâu: Xavier/He Init, BatchNorm, Dropout & Residual',
      summary:
        'Các đột phá ổn định quá trình tối ưu hóa mạng nơ-ron sâu: bảo toàn phương sai tín hiệu qua khởi tạo Xavier/Kaiming He, chuẩn hóa BatchNorm (và LayerNorm) cùng khác biệt chế độ `train`/`eval`, Inverted Dropout và Kết nối phần dư Residual Network.',
      outcomes: [
        'Dẫn xuất phương sai khởi tạo Kaiming He $\\text{Var}(W) = 2/n_{\\text{in}}$ cho tầng tuyến tính theo sau bởi ReLU.',
        'Phân biệt hành vi và cập nhật thống kê trượt EMA của Batch Normalization giữa chế độ huấn luyện (`train`) và suy luận (`eval`).',
        'Giải thích cơ chế truyền gradient không bị triệt tiêu nhờ đường tắt đơn vị $I + \\frac{\\partial F}{\\partial x}$ trong khối Residual.',
      ],
      references: [
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Deep Learning Optimization, Weight Initialization, BatchNorm & Residual Blocks',
          note: 'Phân tích độ ổn định số học khi tăng độ sâu mạng và quy trình chuẩn hóa theo batch.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§13.5 & §14.2 (Initialization, Normalization Layers & Skip Connections)',
          note: 'So sánh Xavier vs He Init, BatchNorm vs LayerNorm và giải thích hiệu ứng ResNet.',
        },
      ],
      sections: [
        {
          heading: '1. Bảo toàn phương sai: Khởi tạo Xavier/Glorot & Kaiming He',
          body: String.raw`Xét tầng $z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} a_i$ với các trọng số và đầu vào độc lập trung bình 0. Để tín hiệu không bùng nổ ra vô cực hay triệt tiêu về 0 sau 50 lớp nhân liên tiếp, ta cần $\text{Var}(z_j) \approx \text{Var}(a_i)$:`,
          math: String.raw`\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \;\;(\text{Xavier/Tanh}), \qquad \text{Var}(w) = \frac{2}{n_{\text{in}}} \;\;(\text{Kaiming He cho ReLU})`,
          intuition:
            'ReLU triệt tiêu phân nửa phân phối đối xứng về $0$, làm $\\mathbb{E}[a^2] = \\frac{1}{2}\\text{Var}(z)$ thay vì $1$. Nhân thêm thừa số $2$ trong công thức Kaiming He bù đắp chính xác phần năng lượng bị cắt đó.',
        },
        {
          heading: '2. Batch Normalization, LayerNorm & Inverted Dropout',
          body: String.raw`**BatchNorm** chuẩn hóa từng kênh đặc trưng qua thống kê của mini-batch (phương sai tổng thể $\frac{1}{N}\sum (x_i - \mu_B)^2$) rồi khôi phục khả năng biểu diễn nhờ tham số học được $\gamma, \beta$:`,
          math: String.raw`\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta, \qquad \mu_{\text{run}} \leftarrow (1 - \alpha)\mu_{\text{run}} + \alpha \mu_B`,
          pitfall:
            'Quên chuyển mô hình sang chế độ `eval` khi suy luận hoặc đánh giá sẽ làm đầu ra của một ảnh bị phụ thuộc ngẫu nhiên vào các ảnh khác chung batch (hoặc gây nhiễu nghiêm trọng khi batch size = 1). Trong khi đó, **LayerNorm** (chuẩn hóa dọc theo chiều đặc trưng của từng mẫu) và **Inverted Dropout** ($\tilde{a} = \frac{m \odot a}{1-p}$ khi train) hoàn toàn độc lập với kích thước batch.',
        },
        {
          heading: '3. Kết nối phần dư (Residual Connections)',
          body: String.raw`Khi chồng mạng sâu thông thường, sai số ngay cả trên tập huấn luyện cũng tăng (hiện tượng thoái hóa — Degradation, không phải overfit). ResNet học ánh xạ phần dư $x_{l+1} = x_l + \mathcal{F}(x_l)$, tạo đường cao tốc trực tiếp cho gradient ngược từ lớp cuối $L$ về lớp sớm $l$:`,
          math: String.raw`\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(I + \frac{\partial}{\partial x_l}\sum_{k=l}^{L-1}\mathcal{F}(x_k)\right)`,
        },
      ],
      simulationId: 'm7-u3',
      exercises: [
        {
          id: 'm7-u3-batchnorm-forward',
          title: 'Batch Normalization: Chế độ Train (EMA) vs Eval',
          prompt:
            'Cài đặt `batchnorm_forward(X, gamma, beta, eps=1e-5, mode=\'train\', running_mean=None, running_var=None, momentum=0.1)` cho mảng `X` shape `(N, D)`:\n- Nếu `running_mean` hoặc `running_var` là `None`, khởi tạo lần lượt bằng `np.zeros(D)` và `np.ones(D)` (không chỉnh sửa tại chỗ mảng đầu vào).\n- Khi `mode == \'train\'`: tính trung bình batch `mean = X.mean(axis=0)` và phương sai batch `var = X.var(axis=0)` (chia $N$), chuẩn hóa `x_hat = (X - mean) / np.sqrt(var + eps)`, trả về `out = gamma * x_hat + beta` cùng bản cập nhật `new_rm = (1 - momentum) * rm + momentum * mean`, `new_rv = (1 - momentum) * rv + momentum * var`.\n- Khi `mode == \'eval\'`: chuẩn hóa bằng `rm` và `rv`, không cập nhật chúng (trả bản sao không đổi).',
          hint: 'Nhớ sao chép `rm = np.zeros(D, dtype=float) if running_mean is None else np.array(running_mean, dtype=float, copy=True)` để không làm biến đổi mảng `running_mean` gốc bên ngoài hàm (tránh side-effect).',
          setup:
            COMMON_SETUP +
            String.raw`
_X_bn1 = np.array([[1.0, 2.0, -1.0], [3.0, 6.0, 1.0], [5.0, 10.0, 3.0], [-1.0, 2.0, 1.0]])
_gamma_bn = np.array([2.0, 0.5, 1.0])
_beta_bn = np.array([-1.0, 3.0, 0.25])
`,
          starter: String.raw`def batchnorm_forward(X, gamma, beta, eps=1e-5, mode='train', running_mean=None, running_var=None, momentum=0.1):
    raise NotImplementedError
`,
          solution: String.raw`def batchnorm_forward(X, gamma, beta, eps=1e-5, mode='train', running_mean=None, running_var=None, momentum=0.1):
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    _, D = X.shape
    rm = np.zeros(D, dtype=float) if running_mean is None else np.array(running_mean, dtype=float, copy=True)
    rv = np.ones(D, dtype=float) if running_var is None else np.array(running_var, dtype=float, copy=True)

    if mode == 'train':
        mean = X.mean(axis=0)
        var = X.var(axis=0)
        x_hat = (X - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta
        new_rm = (1.0 - momentum) * rm + momentum * mean
        new_rv = (1.0 - momentum) * rv + momentum * var
        return out, new_rm, new_rv
    elif mode == 'eval':
        x_hat = (X - rm) / np.sqrt(rv + eps)
        out = gamma * x_hat + beta
        return out, rm, rv
    else:
        raise ValueError(f'Unknown mode: {mode}')
`,
          test: String.raw`def _run_checks():
    _X_single = np.array([[4.0, 8.0, 2.0]])
    c1 = _check('Che do train chuan hoa trung binh beta & cap nhat running stats EMA', lambda: (
        lambda out, rm, rv: (
            out.shape == (4, 3)
            and np.allclose(out.mean(axis=0), _beta_bn, atol=1e-7)
            and np.allclose(rm, 0.1 * _X_bn1.mean(axis=0), atol=1e-7)
            and np.allclose(rv, 0.9 * np.ones(3) + 0.1 * _X_bn1.var(axis=0), atol=1e-7),
            f'out_mean={out.mean(axis=0).round(3)}, rm={rm.round(3)}'
        )
    )(*batchnorm_forward(_X_bn1, _gamma_bn, _beta_bn, mode='train')))

    _rm_init = np.array([2.0, 5.0, 1.0])
    _rv_init = np.array([4.0, 9.0, 1.0])
    _ref_eval = _gamma_bn * ((_X_single - _rm_init) / np.sqrt(_rv_init + 1e-5)) + _beta_bn
    c2 = _check('Che do eval hoat dong voi batch size = 1 & khong thay doi running stats', lambda: (
        lambda out, rm, rv: (
            np.allclose(out, _ref_eval, atol=1e-7)
            and np.allclose(rm, _rm_init, atol=1e-9)
            and np.allclose(rv, _rv_init, atol=1e-9),
            f'eval_out={out.round(4)}, rm_unchanged={np.allclose(rm, _rm_init)}'
        )
    )(*batchnorm_forward(_X_single, _gamma_bn, _beta_bn, mode='eval', running_mean=_rm_init, running_var=_rv_init)))
    _report('m7-u3-batchnorm-forward', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao mạng Transformer trong NLP (như BERT, GPT) ưu tiên LayerNorm thay cho BatchNorm trong khi CNN cổ điển ưu tiên BatchNorm?',
          guidance:
            'Dữ liệu chuỗi có độ dài biến thiên (nhiều padding) và kích thước lô hiệu dụng trên từng token dao động mạnh giữa các mini-batch; thống kê BatchNorm qua phương batch bị nhiễu nặng. LayerNorm chuẩn hóa độc lập theo chiều ẩn $d_{\\text{model}}$ cho từng token ở từng câu, nên hành vi giữa train và test đồng nhất tuyệt đối.',
        },
        {
          question:
            'BatchNorm có phải là hàm trơn tuyệt đối về mặt gradient khi đang đặt sau một lớp Linear không có bias không? Vì sao các lớp Linear/Conv ngay trước BatchNorm thường tắt `bias=False`?',
          guidance:
            'Mọi hằng số cộng thêm $b$ vào tiền kích hoạt $z$ đều cộng đồng thời vào mẫu lẫn trung bình $\\mu_B$, nên $(z + b) - (\\mu_B + b) = z - \\mu_B$ triệt tiêu hoàn toàn! Bias cũ bị dư thừa vì tham số dịch chuyển $\\beta$ ngay sau BatchNorm đã đảm nhiệm vai trò đó.',
        },
      ],
    },
    {
      id: 'm7-u4',
      moduleId: 7,
      order: 4,
      title: 'Kiến trúc CNN (Conv, Pooling, Receptive Field, ResNet → ViT)',
      summary:
        'Thiên kiến quy nạp không gian của mạng tích chập (CNN): chia sẻ trọng số và đẳng biến dịch chuyển, công thức kích thước ngõ ra và trường tiếp nhận (Receptive Field) qua Conv $3\\times 3$, Stride và Dilated Conv, tiến hóa từ VGG/ResNet Bottleneck tới Vision Transformer (ViT).',
      outcomes: [
        'Giải thích hai thiên kiến quy nạp cốt lõi của CNN (Local Connectivity và Translation Equivariance) và tác dụng bất biến cục bộ của Max Pooling.',
        'Tính chính xác kích thước không gian ngõ ra và Receptive Field tích lũy qua chuỗi tầng Conv có kernel size $k$, stride $s$, dilation $d$.',
        'Phân tích lý do tiết kiệm tham số của khối Bottleneck $1\\times 1 \\to 3\\times 3 \\to 1\\times 1$ và đánh đổi Inductive Bias của CNN vs Vision Transformer (ViT).',
      ],
      references: [
        {
          source: '05_Thi_giac_May_tinh_Computer_Vision/',
          section: 'CNN Foundations: Conv2D, Pooling, Receptive Field & Modern Vision Backbones',
          note: 'Từ trích xuất đặc trưng cạnh địa phương tới kiến trúc phân cấp sâu và Patch Embeddings.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§14.1–14.5 (Convolutional Neural Networks, ResNets & Vision Transformers)',
          note: 'Đẳng biến dịch chuyển, hiệu quả xếp tầng 3x3 và Self-Attention trên ảnh chia patch.',
        },
      ],
      sections: [
        {
          heading: '1. Thiên kiến quy nạp của CNN & Kích thước không gian',
          body: String.raw`Thay vì nối đặc khổng lồ mỗi pixel với mọi nơ-ron (khiến ảnh $224\times 224\times 3$ tốn hàng chục triệu tham số chỉ ở lớp đầu), tầng Conv áp dụng hai ràng buộc: **kết nối cục bộ** (chỉ nhìn vùng $k \times k$) và **chia sẻ trọng số** (cùng một bộ lọc trượt khắp ảnh), tạo ra tính **đẳng biến dịch chuyển (Translation Equivariance)** ($T(\text{Conv}(X)) = \text{Conv}(T(X))$). Kết hợp với Max Pooling giúp thu được tính bất biến với dịch chuyển nhỏ. Kích thước không gian ngõ ra khi có padding $p$, dilation $d$, stride $s$:`,
          math: String.raw`H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(k - 1) - 1}{s} \right\rfloor + 1`,
        },
        {
          heading: '2. Receptive Field tích lũy & Hiệu quả của tầng 3×3 / Bottleneck',
          body: String.raw`Gọi $r$ là trường tiếp nhận hiệu dụng (vùng pixel gốc tác động lên một nơ-ron) và $j$ là bước nhảy (jump) tích lũy từ đầu vào (khởi tạo $r_0 = 1,\; j_0 = 1$). Qua mỗi tầng $(k_l, s_l, d_l)$:`,
          math: String.raw`r_l = r_{l-1} + d_l (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1} \cdot s_l`,
          intuition:
            'Xếp chồng **3 tầng Conv $3\\times 3$** ($s=1, d=1$) cho trường tiếp nhận $r_3 = 1 + 2 + 2 + 2 = 7\\times 7$, tương đương 1 tầng $7\\times 7$, nhưng chỉ tốn $3 \\times (3^2 C^2) = 27 C^2$ tham số (giảm 45% so với $49 C^2$) và chèn thêm 3 hàm phi tuyến ReLU xen giữa (triết lý thiết kế VGG/ResNet). Ngoài ra, **Dilated Conv ($d > 1$)** mở rộng RF theo cấp số nhân mà không làm giảm phân giải không gian.',
          pitfall:
            'Kích thước Receptive Field **lý thuyết** có dạng ô vuông lớn, nhưng Receptive Field **hiệu dụng (ERF)** thực tế theo phân tích gradient lại tuân theo phân phối Gaussian tập trung ở vùng trung tâm nếu không dùng kết nối phần dư hay Global Attention.',
        },
        {
          heading: '3. Từ ResNet Bottleneck đến Vision Transformer (ViT)',
          body: String.raw`- **Bottleneck ResNet**: dùng $1\times 1$ Conv chiếu số kênh từ $256 \to 64$, chạy $3\times 3$ trên không gian thu gọn, rồi mở rộng lại $64 \to 256$ nhằm cắt giảm mạnh FLOPs khi tăng chiều sâu lên 50–152 tầng.
- **Vision Transformer (ViT)**: chia ảnh $H \times W$ thành $N = HW / P^2$ miếng vá (patch) $P \times P$, chiếu tuyến tính mỗi patch thành token $d$ chiều rồi áp dụng Self-Attention toàn cục ngay từ lớp 1 (Receptive Field = toàn ảnh). Vì thiếu thiên kiến quy nạp địa phương như CNN, ViT dễ overfit trên tập nhỏ và cần dữ liệu tiền huấn luyện lớn hoặc tăng cường dữ liệu mạnh.`,
        },
      ],
      simulationId: 'm7-u4',
      exercises: [
        {
          id: 'm7-u4-conv-rf-im2col',
          title: 'Tính Receptive Field (Conv/Stride/Dilation) & MaxPool2D',
          prompt:
            'Cài đặt hai hàm thuần NumPy:\n1. `receptive_field(layers)`: nhận danh sách các bộ `(kernel_size, stride, dilation)` theo thứ tự từ đầu vào, khởi tạo `rf = 1, jump = 1`, cập nhật tuần tự và trả về bộ số nguyên `(rf, jump)` sau tầng cuối cùng.\n2. `maxpool2d(X, pool_size, stride)`: thực hiện Max Pooling 2D trên mảng 2D `X` shape `(H, W)` trả về mảng float64 kích thước `((H - pool_size) // stride + 1, (W - pool_size) // stride + 1)`.',
          hint: 'Với mỗi `(k, s, d)` trong `layers`: cập nhật `rf += d * (k - 1) * jump` trước, rồi mới cập nhật `jump *= s`! Với `maxpool2d`: duyệt `i in range(out_h)` và `j in range(out_w)`, lấy `np.max(X[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size])`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X_pool = np.array([
    [1.0, 3.0, 2.0, 4.0],
    [5.0, 2.0, 8.0, 1.0],
    [3.0, 6.0, 0.0, 2.0],
    [7.0, 4.0, 9.0, 5.0],
])
`,
          starter: String.raw`def receptive_field(layers):
    raise NotImplementedError

def maxpool2d(X, pool_size, stride):
    raise NotImplementedError
`,
          solution: String.raw`def receptive_field(layers):
    rf, jump = 1, 1
    for k, s, d in layers:
        rf = rf + int(d) * (int(k) - 1) * jump
        jump = jump * int(s)
    return int(rf), int(jump)

def maxpool2d(X, pool_size, stride):
    X = np.asarray(X, dtype=float)
    H, W = X.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    out = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        r0 = i * stride
        for j in range(out_w):
            c0 = j * stride
            out[i, j] = np.max(X[r0:r0 + pool_size, c0:c0 + pool_size])
    return out
`,
          test: String.raw`def _run_checks():
    c1 = _check('3 lop Conv 3x3 cho RF = 7, jump = 1 (VGG theorem)', lambda: (
        lambda res: (res == (7, 1), f'got={res}')
    )(receptive_field([(3, 1, 1), (3, 1, 1), (3, 1, 1)])))
    c2 = _check('Ket hop Stride=2 va Dilated Conv d=2 tinh dung RF = 15, jump = 2', lambda: (
        lambda res: (res == (15, 2), f'got={res}')
    )(receptive_field([(3, 2, 1), (3, 1, 2), (3, 1, 1)])))
    _ref_pool = np.array([[5.0, 8.0], [7.0, 9.0]])
    c3 = _check('maxpool2d kich thuoc 2x2 stride 2 dung gia tri', lambda: (
        lambda out: (
            out.shape == (2, 2) and np.allclose(out, _ref_pool),
            f'got={out.tolist()}'
        )
    )(maxpool2d(_X_pool, 2, 2)))
    _report('m7-u4-conv-rf-im2col', c1, c2, c3)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong Depthwise Separable Convolution (dùng trong MobileNet/Xception), chi phí tính toán giảm bao nhiêu lần so với Standard Conv2D khi dùng kernel $k\\times k$ và $C_{\\text{out}}$ kênh đầu ra?',
          guidance:
            'Standard Conv tốn $H W \\cdot C_{\\text{in}} \\cdot C_{\\text{out}} \\cdot k^2$ phép nhân cộng. Depthwise ($k\\times k$ độc lập từng kênh) + Pointwise ($1\\times 1$ trộn kênh) chỉ tốn $H W \\cdot (C_{\\text{in}} k^2 + C_{\\text{in}} C_{\\text{out}})$, giảm tỷ lệ xuống còn $\\frac{1}{C_{\\text{out}}} + \\frac{1}{k^2}$ (với $k=3$, tiết kiệm gần 8–9 lần!).',
        },
        {
          question:
            'Khi áp dụng ResNet hay ViT trên bài toán Phân loại Y tế với chỉ 2,000 ảnh X-ray nhưng cần train từ đầu (from scratch), lựa chọn kiến trúc nào an toàn hơn về mặt thiên kiến quy nạp và vì sao?',
          guidance:
            'CNN/ResNet có sẵn thiên kiến địa phương (Local Connectivity) và đẳng biến dịch chuyển ngay ở cấu trúc trọng số, giúp học các đặc trưng cạnh/kết cấu hiệu quả trên tập dữ liệu nhỏ. ViT không có thiên kiến dịch chuyển nội tại mà phải tự học lại cấu trúc không gian 2D qua Self-Attention, nên dễ overfit trừ phi được tiền huấn luyện (pre-trained) hoặc chưng cất (distillation) từ một CNN teacher.',
        },
      ],
    },
  ],
};
