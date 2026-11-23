import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_09: GradModule = {
  id: 9,
  title: 'Học trên Dữ liệu Đồ thị (Graph Neural Networks)',
  englishTitle: 'Graph Representation Learning & GNNs',
  color: '#2dd4bf',
  status: 'ready',
  overview:
    'Lý thuyết phổ đồ thị (Graph Laplacian & Spectral Convolutions), khung truyền thông điệp tổng quát (Message Passing — MPNN), phân tích toán học các kiến trúc chuẩn GCN (Kipf & Welling), GraphSAGE, Graph Attention Networks (GAT), bài toán phân loại Nút/Cạnh/Đồ thị và đồ thị động (Dynamic Graphs).',
  outcomes: [
    'Phân tích phổ trị riêng của ma trận Graph Laplacian chuẩn hóa, năng lượng Dirichlet và phép xấp xỉ bậc nhất Chebyshev trong GCN.',
    'Cài đặt khung Message Passing tổng quát cùng lớp GCN và Masked Softmax Attention (GAT) bảo toàn tính đẳng biến hoán vị nút.',
    'Thiết kế quy trình chia tập hợp lệ (Transductive vs Inductive, Message/Supervision Split, Temporal Split cho đồ thị động) tránh rò rỉ dữ liệu.',
    'Cài đặt Graph Readout phân mảnh (batch_ids) và chấm điểm dự đoán liên kết (Link Prediction ROC-AUC).',
  ],
  sources: [
    'Zhiyuan Liu & Jie Zhou — Introduction to Graph Neural Networks (§2.3, Ch. 4–5, §5.1.3, §7.1, §11.1)',
    'Phòng lab GNN tích hợp sẵn trên ML Studio (#gnn)',
  ],
  units: [
    {
      id: 'm9-u1',
      moduleId: 9,
      order: 1,
      title: 'Graph Laplacian & Spectral Graph Convolutions',
      summary:
        'Chuyển đổi phép tích chập từ lưới ảnh đều sang cấu trúc đồ thị phi Euclid bất kỳ: ma trận Laplacian $L = D - A$, năng lượng Dirichlet đo độ trơn tín hiệu nút, phép biến đổi Fourier đồ thị và định lý xấp xỉ đa thức Chebyshev bậc nhất của GCN (Kipf & Welling, 2017).',
      outcomes: [
        'Tính năng lượng Dirichlet $\\text{Tr}(X^\\top L X)$ để định lượng độ trơn của đặc trưng nút trên đồ thị.',
        'Giải thích phép nhân chuẩn hóa đối xứng $\\tilde{D}^{-1/2} (A + I) \\tilde{D}^{-1/2} X \\Theta$ như bộ lọc thông thấp trên phổ trị riêng.',
        'Chẩn đoán hiện tượng Over-smoothing khi xếp quá nhiều lớp truyền thông điệp làm triệt tiêu khác biệt giữa các nút.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
          section: 'Zhiyuan Liu & Jie Zhou (§2.3 & Ch. 4: Spectral Graph Neural Networks)',
          note: 'Định nghĩa Graph Laplacian, phổ trị riêng, ChebNet và phép rút gọn GCN.',
        },
        {
          source: 'Phòng lab GNN ML Studio (#gnn)',
          section: 'GCN — Normalization by Both Endpoint Degrees',
          note: 'Thực hành trực tiếp đối chiếu phép chuẩn hóa bậc hai đầu cạnh.',
        },
      ],
      sections: [
        {
          heading: '1. Graph Laplacian & Năng lượng Dirichlet (Dirichlet Energy)',
          body: String.raw`Xét đồ thị vô hướng không tự khuyên $G = (V, E)$ với ma trận kề đối xứng $A \in \{0,1\}^{N \times N}$, ma trận bậc $D_{ii} = \sum_j A_{ij}$ và tín hiệu đặc trưng nút $X \in \mathbb{R}^{N \times d}$. Ma trận Laplacian tổ hợp $L = D - A$ đo độ biến thiên tín hiệu giữa các cặp nút kề nhau thông qua **Năng lượng Dirichlet**:`,
          math: String.raw`\mathcal{E}_{\text{Dir}}(X) = \text{Tr}(X^\top L X) = \frac{1}{2} \sum_{i,j=1}^N A_{ij} \|x_i - x_j\|_2^2`,
          intuition:
            'Năng lượng Dirichlet càng nhỏ đồng nghĩa các nút nối cạnh với nhau mang vector đặc trưng càng giống nhau (độ trơn cao / tần số đồ thị thấp).',
        },
        {
          heading: '2. Spectral Convolution & Xấp xỉ GCN (Kipf & Welling)',
          body: String.raw`Vì $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2} = U \Lambda U^\top$ đối xứng bán xác định dương (với $0 \le \lambda_k \le 2$), phép tích chập phổ $g_\theta \star x = U g_\theta(\Lambda) U^\top x$ đòi hỏi phân rã trị riêng $O(N^3)$. ChebNet và **GCN** xấp xỉ $g_\theta(\Lambda)$ bằng đa thức Chebyshev bậc 1 quanh $\lambda_{\max} \approx 2$, kết hợp thủ thuật tái chuẩn hóa **Renormalization Trick** (cộng ma trận đơn vị $I_N$ để giữ tự khuyên):`,
          math: String.raw`\tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}, \qquad H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} \Theta^{(l)}\right)`,
          pitfall:
            'Do $\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}$ hoạt động như bộ lọc thông thấp, chồng quá nhiều tầng GCN ($L \\ge 6$–$8$) gây **Over-smoothing**: $\\mathcal{E}_{\\text{Dir}}(H^{(L)}) \\to 0$ khiến mọi nút trong cùng thành phần liên thông suy biến về cùng một biểu diễn tuyến tính tỷ lệ với $\\sqrt{\\tilde{d}_i}$!',
        },
      ],
      simulationId: 'm9-u1',
      exercises: [
        {
          id: 'm9-u1-graph-laplacian',
          title: 'Dirichlet Energy & Tầng truyền GCN Tái chuẩn hóa',
          prompt:
            'Cài đặt `dirichlet_energy_and_cheb1(A, X, theta)` cho đồ thị vô hướng không tự khuyên `A` `(N, N)`, đặc trưng `X` `(N, d)`, và ma trận trọng số `theta` `(d, d_out)`:\n1. Tính `L = np.diag(A.sum(axis=1)) - A` và năng lượng Dirichlet vô hướng `energy = float(np.trace(X.T @ L @ X))`.\n2. Cộng tự khuyên `A_tilde = A + np.eye(N)`, tính `d_inv_sqrt = np.where(deg_t > 0, 1.0 / np.sqrt(deg_t), 0.0)`, rồi tính phép truyền chuẩn hóa đối xứng `H_out = (A_hat @ X) @ theta`.\nTrả về `(energy, H_out)`.',
          hint: 'Chú ý không cộng tự khuyên vào ma trận dùng tính `L` cho năng lượng Dirichlet (vì tự khuyên không đổi $x_i - x_i = 0$ nhưng `np.diag(A.sum(axis=1)) - A` luôn chuẩn), chỉ dùng `A_tilde = A + np.eye(N)` ở bước lan truyền GCN.',
          setup:
            COMMON_SETUP +
            String.raw`
# 4-node cycle graph: 0-1-2-3-0 with high-frequency alternating signal [-1, +1, -1, +1]
_A_cycle = np.array([
    [0., 1., 0., 1.],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 0., 1., 0.],
])
_X_hf = np.array([[-1.0, 0.5], [1.0, 0.5], [-1.0, -0.5], [1.0, -0.5]])
_theta_id = np.eye(2)
`,
          starter: String.raw`def dirichlet_energy_and_cheb1(A, X, theta):
    raise NotImplementedError
`,
          solution: String.raw`def dirichlet_energy_and_cheb1(A, X, theta):
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n = A.shape[0]
    L = np.diag(A.sum(axis=1)) - A
    energy = float(np.trace(X.T @ L @ X))

    A_tilde = A + np.eye(n)
    deg_t = A_tilde.sum(axis=1)
    d_inv_sqrt = np.where(deg_t > 0.0, 1.0 / np.sqrt(deg_t), 0.0)
    A_hat = (d_inv_sqrt[:, None] * A_tilde) * d_inv_sqrt[None, :]
    H_out = (A_hat @ X) @ theta
    return energy, H_out
`,
          test: String.raw`def _run_checks():
    def _chk_energy():
        e, H1 = dirichlet_energy_and_cheb1(_A_cycle, _X_hf, _theta_id)
        # Edge diffs on col0: 4 edges * (2^2) = 16; col1: 2 edges * (1^2) = 2 -> sum = 18
        return (
            H1.shape == (4, 2) and abs(e - 18.0) < 1e-8,
            f'energy={e:.6f} (expected 18.0)'
        )
    c1 = _check('Nang luong Dirichlet khop cong thuc Tr(X^T L X)', _chk_energy)
    def _chk_smooth():
        e0, H1 = dirichlet_energy_and_cheb1(_A_cycle, _X_hf, _theta_id)
        e1, _ = dirichlet_energy_and_cheb1(_A_cycle, H1, _theta_id)
        return e1 < e0 * 0.25, f'E_before={e0:.3f} -> E_after={e1:.3f}'
    c2 = _check('Lan truyen GCN hoat dong nhu bo loc thong thap (giam Dirichlet energy)', _chk_smooth)
    _report('m9-u1-graph-laplacian', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao các nghiên cứu sâu trên GNN dùng Residual / JK-Net hoặc chuẩn hóa **PairNorm** để giảm Over-smoothing?',
          guidance:
            'Khi năng lượng Dirichlet tiến về $0$, mọi nút tiến về một điểm chung nên Gradient cũng mất phân biệt lớp. Kết nối nhảy tầng (Skip / JK-Net) giữ lại tín hiệu tần số cao từ các tầng sớm, còn PairNorm trực tiếp bảo toàn tổng khoảng cách bình phương giữa các nút ở mức hằng số bất biến theo độ sâu.',
        },
        {
          question:
            'Trong biểu thức GCN $\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} X$, chuyện gì xảy ra trên đồ thị có sự chênh lệch bậc cực lớn (như mạng xã hội theo phân phối luật lũy thừa — power law)?',
          guidance:
            'Nút trung tâm (hub) có bậc $d_j = 10{,}000$ truyền thông điệp bị nhân hệ số $1/\\sqrt{d_i d_j}$ rất nhỏ tới mỗi láng giềng lá, nhưng vẫn nhận tổng cộng $10{,}000$ tín hiệu đổ về từ các láng giềng khiến chuẩn biểu diễn của chính hub có nguy cơ phình to hoặc mất cân đối nếu không dùng chuẩn hóa nút/chú ý.',
        },
      ],
    },
    {
      id: 'm9-u2',
      moduleId: 9,
      order: 2,
      title: 'Khung Message Passing (MPNN) & Tính đẳng biến hoán vị',
      summary:
        'Chuyển từ miền phổ phụ thuộc cấu trúc cố định sang miền không gian tổng quát **Message Passing Neural Networks (MPNN)**: giải phẫu ba hàm `Message`–`Aggregate`–`Update`, tính chất bất biến/đẳng biến hoán vị bắt buộc trên đồ thị và giới hạn phân biệt đẳng cấu 1-WL.',
      outcomes: [
        'Phân biệt **Permutation Invariance** (bài toán cấp đồ thị) và **Permutation Equivariance** (bài toán cấp nút).',
        'Giải thích lý do bộ tổng hợp (Aggregator) bắt buộc phải đối xứng trên tập (multiset): `SUM` phân biệt được số lượng láng giềng tốt hơn `MEAN`/`MAX` theo kiểm định 1-WL.',
        'Cài đặt tầng MPNN hai nhánh (tự thân + tổng hợp trung bình láng giềng) an toàn trước nút cô lập bậc 0 và kiểm chứng tính đẳng biến hoán vị.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
          section: 'Zhiyuan Liu & Jie Zhou (Ch. 5, §5.1.3 & Spatial GNNs)',
          note: 'Tổng hợp không gian trên lân cận $N(v)$ và khung MPNN của Gilmer et al.',
        },
        {
          source: 'Phòng lab GNN ML Studio (#gnn)',
          section: 'Mean Aggregation & Isolated Node Safety',
          note: 'So sánh 1-hop và 2-hop message passing đồng bộ trạng thái.',
        },
      ],
      sections: [
        {
          heading: '1. Tính Bất biến vs Đẳng biến Hoán vị (Permutation Symmetry)',
          body: String.raw`Một đồ thị $N$ nút không có thứ tự nút cố hữu: đánh số lại tập nút bởi ma trận hoán vị bất kỳ $P \in \{0,1\}^{N \times N}$ ($P^\top P = I$) cho ma trận kề $P A P^\top$ và đặc trưng $P H$ vẫn mô tả cùng một đồ thị. Do đó, mọi mạng GNN hợp lệ phải thỏa:
- **Permutation Equivariance (mức nút)**: đánh số lại nút thì thứ tự hàng đầu ra phải hoán vị đúng theo $P$.
- **Permutation Invariance (mức toàn đồ thị)**: hàm dự đoán nhãn đồ thị không thay đổi dưới bất kỳ hoán vị $P$ nào.`,
          math: String.raw`f(P A P^\top,\, P H) = P\, f(A, H), \qquad g(P A P^\top,\, P H) = g(A, H)`,
        },
        {
          heading: '2. Khuôn mẫu MPNN & Sức biểu diễn Weisfeiler–Lehman (1-WL)',
          body: String.raw`Tại tầng $l$, mỗi nút $u$ tổng hợp thông điệp từ tập láng giềng $\mathcal{N}(u)$ qua một bộ toán tử đối xứng bất biến hoán vị $\bigoplus$ rồi kết hợp với trạng thái tự thân $h_u^{(l)}$:`,
          math: String.raw`m_{\mathcal{N}(u)}^{(l)} = \bigoplus_{v \in \mathcal{N}(u)} \text{MSG}\big(h_u^{(l)}, h_v^{(l)}, e_{uv}\big), \qquad h_u^{(l+1)} = \sigma\!\left(W_{\text{self}} h_u^{(l)} + W_{\text{neigh}}\, m_{\mathcal{N}(u)}^{(l)}\right)`,
          intuition:
            'Xu et al. (GIN, 2019) chứng minh không có mô hình Message Passing chuẩn nào mạnh hơn phép kiểm tra đẳng cấu đồ thị **1-Weisfeiler-Lehman (1-WL)**. Trong ba bộ gộp `SUM`, `MEAN`, `MAX`: `SUM` bảo toàn toàn bộ lực lượng đa tập (multiset cardinality) nên phân biệt được nút có 2 láng giềng giống hệt nhau với nút có 4 láng giềng giống hệt nhau, điều mà `MEAN` và `MAX` bị mù.',
          pitfall:
            'Nếu một nút bị cô lập ($\\text{deg}(u) = 0$) khi dùng chuẩn hóa `MEAN`, chia trực tiếp cho `deg` sẽ tạo ra `NaN`/`inf` làm hỏng toàn bộ lan truyền xuôi và ngược. Luôn dùng phép chia có điều kiện (`where=deg > 0`) để gán thông điệp bằng vector $0$.',
        },
      ],
      simulationId: 'm9-u2',
      exercises: [
        {
          id: 'm9-u2-mpnn-equiv',
          title: 'MPNN Tách nhánh Tự thân & Kiểm chứng Đẳng biến Hoán vị',
          prompt:
            'Cài đặt tầng `mpnn_mean_layer(A, H, W_self, W_neigh)`:\n1. Tính vector tổng đặc trưng láng giềng `raw = A @ H` `(N, d_in)` và bậc nút `deg = A.sum(axis=1, keepdims=True)`.\n2. Chia an toàn ra trung bình láng giềng `agg`: tại nút bậc 0 (`deg == 0`), gán `agg[u] = 0` không phát sinh `NaN`.\n3. Trả về `np.maximum(0.0, H @ W_self + agg @ W_neigh)` `(N, d_out)`.',
          hint: 'Dùng `np.divide(raw, deg, out=np.zeros_like(raw, dtype=float), where=deg > 0)`. Vì nhân ma trận trái `A @ H` và kích hoạt theo phần tử ReLU đều giao hoán với ma trận hoán vị `P`, hàm tự động đẳng biến $f(PAP^\\top, PH) = Pf(A, H)$.',
          setup:
            COMMON_SETUP +
            String.raw`
# Node 4 is completely isolated (degree 0)
_A_mp = np.array([
    [0., 1., 1., 0., 0.],
    [1., 0., 0., 1., 0.],
    [1., 0., 0., 1., 0.],
    [0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 0.],
])
_H_mp = np.array([
    [1.0, -0.5, 0.2],
    [0.4, 1.2, -0.3],
    [-0.2, 0.8, 1.1],
    [0.9, -0.1, 0.5],
    [1.5, 0.6, -0.4],
])
_Ws_mp = np.array([[0.5, -0.2], [0.1, 0.7], [-0.4, 0.3]])
_Wn_mp = np.array([[-0.3, 0.6], [0.4, 0.2], [0.5, -0.1]])
_perm_idx = [3, 0, 4, 1, 2]
_P_mp = np.eye(5)[_perm_idx]
`,
          starter: String.raw`def mpnn_mean_layer(A, H, W_self, W_neigh):
    raise NotImplementedError
`,
          solution: String.raw`def mpnn_mean_layer(A, H, W_self, W_neigh):
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    W_self = np.asarray(W_self, dtype=float)
    W_neigh = np.asarray(W_neigh, dtype=float)
    deg = A.sum(axis=1, keepdims=True)
    raw = A @ H
    agg = np.divide(raw, deg, out=np.zeros_like(raw, dtype=float), where=deg > 0)
    return np.maximum(0.0, H @ W_self + agg @ W_neigh)
`,
          test: String.raw`def _run_checks():
    def _chk_iso():
        out = mpnn_mean_layer(_A_mp, _H_mp, _Ws_mp, _Wn_mp)
        iso_expected = np.maximum(0.0, _H_mp[4] @ _Ws_mp)
        return (
            out.shape == (5, 2) and bool(np.all(np.isfinite(out)))
            and np.allclose(out[4], iso_expected, atol=1e-8),
            f'iso_node_out={out[4].round(4)} vs expected={iso_expected.round(4)}'
        )
    c1 = _check('Nut co lap bac 0 huu han & chi nhan nhanh self-connection', _chk_iso)
    def _chk_eq():
        out1 = mpnn_mean_layer(_A_mp, _H_mp, _Ws_mp, _Wn_mp)
        out2 = mpnn_mean_layer(_P_mp @ _A_mp @ _P_mp.T, _P_mp @ _H_mp, _Ws_mp, _Wn_mp)
        return np.allclose(out2, _P_mp @ out1, atol=1e-8), f'max_equiv_err={np.max(np.abs(out2 - _P_mp @ out1)):.2e}'
    c2 = _check('Dang bien hoan vi chinh xac f(P A P^T, P H) = P f(A, H)', _chk_eq)
    _report('m9-u2-mpnn-equiv', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Làm sao dựng một cặp đồ thị 6 nút không đẳng cấu mà mạng GCN/GraphSAGE dùng Mean hoặc Max Pooling cho đầu ra hoàn toàn giống nhau?',
          guidance:
            'Ví dụ cổ điển của Xu et al. (GIN, 2019): xét một nút kết nối với 2 nút màu Xanh-Đỏ (Mean = 0.5 Xanh + 0.5 Đỏ) so với một nút kết nối với 4 nút gồm 2 Xanh - 2 Đỏ! Vì trung bình cộng của multiset nhân đôi vẫn giữ nguyên tỷ lệ (và Max cũng y hệt), chỉ bộ tổng hợp `SUM` mới nhận ra bậc 2 khác bậc 4.',
        },
        {
          question:
            'Mở rộng khung MPNN vượt rào cản 1-WL bằng cách nào trong nghiên cứu hiện đại (Graph Transformers / Subgraph GNN)?',
          guidance:
            'Có ba hướng chính: (1) Bơm mã hóa cấu trúc **Positional / Structural Encodings** (trị riêng Laplacian hoặc khoảng cách bước đi ngẫu nhiên RWPE) trực tiếp vào đặc trưng nút ban đầu; (2) Graph Transformer cho mọi cặp nút chú ý trực tiếp qua khoảng cách đường đi ngắn nhất (Graphormer); (3) k-WL / Subgraph GNNs.',
        },
      ],
    },
    {
      id: 'm9-u3',
      moduleId: 9,
      order: 3,
      title: 'Kiến trúc GCN, GraphSAGE & Graph Attention Networks (GAT)',
      summary:
        'Đối chiếu ba kiến trúc không gian trụ cột của GNN: GCN (trọng số gộp cố định theo cấu trúc bậc), GraphSAGE (lấy mẫu láng giềng kích thước cố định cho suy luận quy nạp quy mô lớn), và Graph Attention Networks (GAT — học trọng số cạnh thích nghi phụ thuộc nội dung đặc trưng nút).',
      outcomes: [
        'Phân biệt mô hình Transductive (GCN trên ma trận cố định) với quy trình Neighbor Sampling quy nạp (Inductive) của GraphSAGE.',
        'Viết phương trình cơ chế Masked Attention trong GAT: chiếu đặc trưng $Z = H W$, điểm chú ý LeakyReLU tách $a_{\\text{src}}^\\top z_i + a_{\\text{dst}}^\\top z_j$ và mặt nạ cạnh.',
        'Cài đặt hoàn chỉnh một tầng GAT 1 đầu (Single-Head Attention) bảo toàn Masked Softmax và đẳng biến hoán vị.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
          section: 'Zhiyuan Liu & Jie Zhou (Ch. 5 & §7.1: GCN, GraphSAGE & GAT)',
          note: 'Cơ chế lấy mẫu láng giềng quy nạp và Graph Attention với mặt nạ đồ thị.',
        },
        {
          source: 'Phòng lab GNN ML Studio (#gnn)',
          section: 'GAT — Masked Softmax Attention & Equivariance',
          note: 'Thực hành trực tiếp trọng số chú ý phụ thuộc đặc trưng và kiểm tra đẳng biến.',
        },
      ],
      sections: [
        {
          heading: '1. GCN vs GraphSAGE: Khả năng Tổng quát Quy nạp (Inductive Learning)',
          body:
            String.raw`Trong GCN chuẩn, phép nhân ma trận $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ đòi hỏi toàn bộ ma trận kề $N \times N$ trong lúc tối ưu và tính lan truyền toàn phần ($O(|E| d)$ mỗi bước, dễ bùng nổ bộ nhớ lân cận *Neighbor Explosion* khi số tầng $L$ tăng). **GraphSAGE** giải quyết bằng cách:
1. Lấy mẫu cố định kích thước nhỏ $S_1, S_2$ láng giềng mỗi tầng để tạo mini-batch kích thước chặn.
2. Nối ghép (` +
            '`concat`' +
            String.raw`) tự thân và vector gộp láng giềng kèm chuẩn hóa L2, giúp áp dụng mượt mà cho các nút mới xuất hiện hoàn toàn ở tập Test (**Inductive Inference**).`,
        },
        {
          heading: '2. Graph Attention Networks (GAT) & Masked Softmax',
          body: String.raw`GCN gán trọng số cạnh cố định $1/\sqrt{\tilde{d}_i \tilde{d}_j}$ chỉ dựa trên bậc, bất chấp láng giềng $j$ có thông tin đồng nhất hay gây nhiễu với nút $i$. **GAT (Veličković et al., 2018)** cho phép từng cạnh tự học hệ số chú ý $\alpha_{ij}$ theo nội dung đặc trưng sau khi thêm tự khuyên $\tilde{\mathcal{N}}(i) = \mathcal{N}(i) \cup \{i\}$:`,
          math: String.raw`e_{ij} = \text{LeakyReLU}\!\left(a_{\text{src}}^\top (W^\top h_i) + a_{\text{dst}}^\top (W^\top h_j)\right), \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}`,
          intuition:
            'Thủ thuật tách vector chú ý $a = [a_{\\text{src}} \\parallel a_{\\text{dst}}]$ giúp tính $(Z a_{\\text{src}})$ và $(Z a_{\\text{dst}})$ trong $O(N F\')$ rồi cộng phát sóng cho các cạnh, tránh phải tạo tensor $H_i \\parallel H_j$ tốn bộ nhớ gấp bội!',
          pitfall:
            'Trong đồ thị có nhiều cạnh nhiễu hoặc **Heterophilic Graph** (hai nút khác lớp nối cạnh nhiều hơn cùng lớp, như mạng hẹn hò hay lưới hóa học), GAT giúp hạ trọng số cạnh nhiễu; tuy nhiên lưu ý cơ chế GAT gốc tính $e_{ij}$ là phép chú ý "tĩnh" (rank-1 theo khóa truy vấn bất kể đặc trưng đích — sau được khắc phục ở GATv2 bằng cách đặt `LeakyReLU` *trước* `a`).',
        },
      ],
      simulationId: 'm9-u3',
      exercises: [
        {
          id: 'm9-u3-gat-layer',
          title: 'Single-Head Graph Attention (GAT) & Masked Softmax',
          prompt:
            'Cài đặt `gat_single_head(A, H, W, a_src, a_dst, negative_slope=0.2)`:\n1. Chiếu tuyến tính `Z = H @ W` `(N, d_out)`.\n2. Tính điểm chú ý cặp `e = (Z @ a_src)[:, None] + (Z @ a_dst)[None, :]` và áp `LeakyReLU(e, negative_slope)`.\n3. Xác định mặt nạ kề có tự khuyên `mask = ((A + np.eye(N)) > 0)`, gán `-1e9` tại vị trí `~mask`, rồi tính hàng Softmax ổn định ra ma trận `alpha` `(N, N)` (đảm bảo `alpha[~mask] == 0.0`).\n4. Trả về `(H_out, alpha)` với `H_out = alpha @ Z`.',
          hint: 'Dùng `e_leaky = np.where(e >= 0.0, e, negative_slope * e)`. Sau bước `exp / sum` trên hàng `axis=1`, nhân thêm mặt nạ `alpha = np.where(mask, alpha, 0.0)` để triệt tiêu tuyệt đối phần dư cạnh không tồn tại.',
          setup:
            COMMON_SETUP +
            String.raw`
_A_gat = np.array([
    [0., 1., 1., 0.],
    [1., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
])
_H_gat = np.array([
    [1.2, -0.4, 0.6],
    [1.1, -0.3, 0.7],   # similar to node 0
    [-1.5, 2.0, -0.8],  # dissimilar to node 0
    [0.2, 0.5, 1.3],
])
_W_gat = np.array([[0.6, -0.2], [0.1, 0.8], [-0.3, 0.5]])
_a_src = np.array([0.7, -0.4])
_a_dst = np.array([-0.5, 0.9])
_P_gat = np.eye(4)[[2, 0, 3, 1]]
`,
          starter: String.raw`def gat_single_head(A, H, W, a_src, a_dst, negative_slope=0.2):
    raise NotImplementedError
`,
          solution: String.raw`def gat_single_head(A, H, W, a_src, a_dst, negative_slope=0.2):
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    W = np.asarray(W, dtype=float)
    a_src = np.asarray(a_src, dtype=float)
    a_dst = np.asarray(a_dst, dtype=float)
    n = A.shape[0]

    Z = H @ W
    e = (Z @ a_src)[:, None] + (Z @ a_dst)[None, :]
    e_leaky = np.where(e >= 0.0, e, negative_slope * e)
    mask = (A + np.eye(n)) > 0
    logits = np.where(mask, e_leaky, -1e9)
    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(logits_shift)
    alpha = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    alpha = np.where(mask, alpha, 0.0)
    H_out = alpha @ Z
    return H_out, alpha
`,
          test: String.raw`def _run_checks():
    def _chk_mask():
        H_out, alpha = gat_single_head(_A_gat, _H_gat, _W_gat, _a_src, _a_dst)
        mask = (_A_gat + np.eye(4)) > 0
        ok = (
            H_out.shape == (4, 2) and alpha.shape == (4, 4)
            and np.allclose(alpha.sum(axis=1), 1.0, atol=1e-7)
            and np.all(alpha[~mask] == 0.0)
            and abs(float(alpha[0, 1]) - float(alpha[0, 2])) > 0.01
        )
        return ok, f'alpha0={alpha[0].round(4).tolist()}'
    c1 = _check('Masked Softmax tong hang = 1, khong canh = 0 & trong so thich nghi khac nhau', _chk_mask)
    def _chk_eq():
        H1, a1 = gat_single_head(_A_gat, _H_gat, _W_gat, _a_src, _a_dst)
        H2, a2 = gat_single_head(_P_gat @ _A_gat @ _P_gat.T, _P_gat @ _H_gat, _W_gat, _a_src, _a_dst)
        ok = np.allclose(H2, _P_gat @ H1, atol=1e-7) and np.allclose(a2, _P_gat @ a1 @ _P_gat.T, atol=1e-7)
        return ok, f'H_err={np.max(np.abs(H2 - _P_gat @ H1)):.2e}'
    c2 = _check('Bao toan tinh dang bien hoan vi cho ca H_out va ma tran alpha', _chk_eq)
    _report('m9-u3-gat-layer', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'So sánh độ phức tạp suy luận và ưu thế biểu diễn giữa **GAT** (tự học trọng số chú ý trên từng cạnh $\\alpha_{ij}$) với **Transformer** tiêu chuẩn?',
          guidance:
            'Transformer tiêu chuẩn chạy Dense Self-Attention $O(N^2)$ xem mọi token nối với nhau hoàn toàn (chỉ cộng mã hóa vị trí), trong khi GAT áp dụng Sparse Attention trực tiếp theo cấu trúc cạnh thực có ($O(|E| + N)$), kế thừa thiên kiến quy nạp thưa của đồ thị mà vẫn phân biệt mức độ quan trọng của từng láng giềng.',
        },
        {
          question:
            'Tại sao các bài báo gốc (Kipf & Welling, Veličković) khuyến nghị dùng Multi-Head Attention cho các tầng GAT ẩn nhưng lại lấy **trung bình cộng** (`mean`) các đầu ở tầng dự đoán cuối cùng thay vì nối ghép (`concat`)?',
          guidance:
            'Ở các tầng ẩn, `concat` $K$ đầu chiều $F\'$ giữ lại đủ $K F\'$ biểu diễn đa dạng song song. Nhưng ở tầng cuối có số chiều bắt buộc bằng đúng số lớp phân loại $C$, lấy trung bình cộng $K$ đầu giúp giảm phương sai ước lượng logits mà không làm phình kích thước ma trận phân loại cuối.',
        },
      ],
    },
    {
      id: 'm9-u4',
      moduleId: 9,
      order: 4,
      title: 'Phân loại Nút/Cạnh/Đồ thị & Đánh giá Đồ thị động',
      summary:
        'Quy chuẩn đánh giá thực nghiệm chuẩn cao học cho mọi tác vụ trên đồ thị: Phân loại Nút (Transductive vs Inductive), Dự đoán Liên kết (Message vs Supervision Edge Split + Negative Sampling), Phân loại Đồ thị (Graph Readout `batch_ids`), và Đồ thị động (Temporal Split chống rò rỉ tương lai).',
      outcomes: [
        'Thiết lập cách phân tách cạnh chuẩn mực cho Link Prediction (tách biệt Message Passing Edges khỏi Supervision Edges).',
        'Cài đặt hàm tổng hợp phân mảnh **Graph Readout** (`mean` / `sum`) trên nhiều đồ thị đóng gói theo `batch_ids`.',
        'Cài đặt bộ chấm điểm Link Prediction bằng tích vô hướng và tính chỉ số ROC-AUC chính xác qua thống kê Mann–Whitney, tránh rò rỉ dữ liệu thời gian ở đồ thị động.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
          section: 'Zhiyuan Liu & Jie Zhou (§11.1 & Graph Readout / Dynamic Graphs / Link Prediction)',
          note: 'Bài toán đa cấp độ (Node, Edge, Graph) và mô hình hóa đồ thị biến thiên theo thời gian.',
        },
        {
          source: 'Phòng lab GNN ML Studio (#gnn)',
          section: 'Graph Splits, Transductive Evaluation & Leakage Guards',
          note: 'Kiểm soát rò rỉ nhãn/cạnh trong thực nghiệm đồ thị.',
        },
      ],
      sections: [
        {
          heading: '1. Ba cấp độ tác vụ & Graph Readout',
          body: String.raw`- **Node Level**: dùng trực tiếp $h_u^{(L)}$ sau $L$ lớp truyền thông điệp. Lưu ý: chia tập **Transductive** giữ chung 1 đồ thị khi chạy message passing (chỉ che nhãn ` + '`val/test`' + String.raw`), còn **Inductive** hoàn toàn tách đồ thị mới chưa từng thấy.
- **Edge / Link Prediction**: biểu diễn cạnh tiềm năng qua tích vô hướng $s_{uv} = z_u^\top z_v$ hoặc MLP($[z_u \odot z_v]$) để phân biệt cạnh dương (tồn tại) với cạnh âm (Negative Sampling).
- **Graph Level Readout**: gộp biểu diễn mọi nút thuộc cùng một đồ thị con $g$ thành vector cố định:`,
          math: String.raw`h_G^{(\text{sum})} = \sum_{u \in V_G} h_u^{(L)}, \qquad h_G^{(\text{mean})} = \frac{1}{|V_G|} \sum_{u \in V_G} h_u^{(L)}`,
        },
        {
          heading: '2. Message Edges vs Supervision Edges & Đồ thị động (Dynamic Graphs)',
          body: String.raw`Hai lỗi rò rỉ nghiêm trọng phổ biến nhất khi đánh giá GNN thực nghiệm:
1. **Rò rỉ cạnh giám sát (Supervision Leakage)**: trong Link Prediction trên đồ thị train, nếu để cạnh cần dự đoán nằm ngay trong ma trận kề dùng để truyền thông điệp, mô hình sẽ học đường tắt tầm thường ("hai nút kề nhau $\implies$ có cạnh") và sụp đổ trên ` + '`val/test`' + String.raw`! Bắt buộc tách cạnh Train thành **Message Edges** (nằm trong $A$) và **Supervision Edges** (chỉ dùng tính loss).
2. **Đồ thị động & Chia ngẫu nhiên (Temporal Leakage)**: với mạng giao dịch, tương tác người–sản phẩm có nhãn thời gian $t$, chia ngẫu nhiên $80/20$ khiến mô hình dùng các cạnh tương lai (tương tác xảy ra ngày mai) để dự đoán giao dịch hôm qua. Mọi đồ thị động **bắt buộc dùng Temporal Split** ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$).`,
          math: String.raw`\text{ROC-AUC} = \frac{1}{|E^+||E^-|} \sum_{e^+ \in E^+}\sum_{e^- \in E^-} \Big(\mathbb{I}[s_{e^+} > s_{e^-}] + \tfrac{1}{2}\mathbb{I}[s_{e^+} = s_{e^-}]\Big)`,
        },
      ],
      simulationId: 'm9-u4',
      exercises: [
        {
          id: 'm9-u4-link-readout',
          title: 'Batched Graph Readout & Link Prediction ROC-AUC',
          prompt:
            'Cài đặt hai hàm lõi cho tầng đọc ra và dự đoán cạnh:\n1. `graph_readout(H, batch_ids, mode=\'mean\')`: với ma trận nhúng nút `H` shape `(N, d)` và vector chỉ số đồ thị `batch_ids` shape `(N,)` mang giá trị nguyên từ `0` đến `G - 1`, tính tổng (`mode == \'sum\'`) hoặc trung bình (`mode == \'mean\'`) theo từng đồ thị trả về `(G, d)`.\n2. `link_pred_auc_scores(Z, pos_edges, neg_edges)`: với nhúng `Z` `(N, d)` và danh sách cạnh `pos_edges` `(Ep, 2)`, `neg_edges` `(En, 2)`, tính điểm tích vô hướng `z_u . z_v` cho mỗi cạnh dương và âm, rồi tính `auc` theo công thức Mann–Whitney U. Trả về `(pos_scores, neg_scores, auc)`.',
          hint: 'Với `graph_readout`: `G = int(np.max(batch_ids)) + 1`, với mỗi `g in range(G)` lấy `mask = batch_ids == g`, dùng `.mean(axis=0)` hoặc `.sum(axis=0)`. Với `auc`: `diff = pos_scores[:, None] - neg_scores[None, :]`, sau đó `float(np.mean((diff > 0) + 0.5 * (diff == 0)))`.',
          setup:
            COMMON_SETUP +
            String.raw`
_H_ro = np.array([
    [1.0, 2.0],
    [3.0, 4.0],   # graph 0 -> mean [2, 3], sum [4, 6]
    [0.0, 1.0],
    [2.0, 5.0],
    [4.0, 6.0],   # graph 1 -> mean [2, 4], sum [6, 12]
])
_batch_ro = np.array([0, 0, 1, 1, 1])
_Z_lp = np.array([
    [2.0, 0.2],
    [1.8, 0.3],
    [-1.5, 1.2],
    [-1.6, 1.0],
])
_pos_lp = np.array([[0, 1], [2, 3]])
_neg_lp = np.array([[0, 2], [1, 3]])
`,
          starter: String.raw`def graph_readout(H, batch_ids, mode='mean'):
    raise NotImplementedError

def link_pred_auc_scores(Z, pos_edges, neg_edges):
    raise NotImplementedError
`,
          solution: String.raw`def graph_readout(H, batch_ids, mode='mean'):
    H = np.asarray(H, dtype=float)
    batch_ids = np.asarray(batch_ids, dtype=int)
    G = int(np.max(batch_ids)) + 1
    out = np.zeros((G, H.shape[1]), dtype=float)
    for g in range(G):
        sub = H[batch_ids == g]
        if sub.shape[0] > 0:
            if mode == 'mean':
                out[g] = sub.mean(axis=0)
            elif mode == 'sum':
                out[g] = sub.sum(axis=0)
            else:
                raise ValueError(f'Unknown readout mode: {mode}')
    return out

def link_pred_auc_scores(Z, pos_edges, neg_edges):
    Z = np.asarray(Z, dtype=float)
    pos_edges = np.asarray(pos_edges, dtype=int)
    neg_edges = np.asarray(neg_edges, dtype=int)
    pos_scores = np.sum(Z[pos_edges[:, 0]] * Z[pos_edges[:, 1]], axis=1)
    neg_scores = np.sum(Z[neg_edges[:, 0]] * Z[neg_edges[:, 1]], axis=1)
    diff = pos_scores[:, None] - neg_scores[None, :]
    auc = float(np.mean((diff > 0.0) + 0.5 * (diff == 0.0)))
    return pos_scores, neg_scores, auc
`,
          test: String.raw`def _run_checks():
    def _chk_ro():
        m_out = graph_readout(_H_ro, _batch_ro, mode='mean')
        s_out = graph_readout(_H_ro, _batch_ro, mode='sum')
        return (
            m_out.shape == (2, 2) and s_out.shape == (2, 2)
            and np.allclose(m_out, [[2.0, 3.0], [2.0, 4.0]])
            and np.allclose(s_out, [[4.0, 6.0], [6.0, 12.0]]),
            f'mean={m_out.tolist()}, sum={s_out.tolist()}'
        )
    c1 = _check('Batched Graph Readout mean & sum dung tung do thi con', _chk_ro)
    def _chk_auc():
        ps, ns, auc = link_pred_auc_scores(_Z_lp, _pos_lp, _neg_lp)
        _, _, auc_tie = link_pred_auc_scores(np.ones((4, 2)), _pos_lp, _neg_lp)
        return (
            ps.shape == (2,) and ns.shape == (2,) and abs(auc - 1.0) < 1e-9 and abs(auc_tie - 0.5) < 1e-9,
            f'auc={auc:.4f}, auc_tie={auc_tie:.4f}'
        )
    c2 = _check('Link Prediction ROC-AUC dat 1.0 khi phan tach & 0.5 khi hoa hoan toan', _chk_auc)
    _report('m9-u4-link-readout', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong Link Prediction trên mạng trích dẫn khoa học (Citation Network) hoặc thương mại điện tử, tại sao chỉ lấy mẫu âm hoàn toàn ngẫu nhiên dễ tạo ảo giác điểm ROC-AUC ~ 0.99 quá dễ?',
          guidance:
            'Trong đồ thị thưa, hai nút chọn ngẫu nhiên gần như chắc chắn khác cụm nên cách nhau rất xa — bất kỳ phương pháp cơ sở nào (như đếm bạn chung hoặc degree) cũng phân biệt dễ dàng. Đánh giá nghiêm túc cao học dùng **Hard Negative Sampling** (chọn cặp cùng lân cận 2-hop hoặc cùng cộng đồng nhưng không nối cạnh) và thước đo xếp hạng **Hits@K / MRR**.',
        },
        {
          question:
            'Trong bài toán Dynamic Graph (Temporal Graph Networks — TGN / DyRep), khi suy luận trực tuyến trên luồng sự kiện mới, làm sao tránh "độ trễ tương lai" khi xây dựng bộ nhớ nút?',
          guidance:
            'Bộ nhớ của nút tại thời điểm sự kiện $t$ chỉ được phép cập nhật *sau khi* sự kiện tại thời điểm $t$ đã được dự đoán mất mát xong (**Memory Update after Prediction**), đồng thời hàm lấy mẫu láng giềng thời gian bắt buộc chỉ truy xuất các cạnh tương tác có nhãn thời gian $\\tau < t$.',
        },
      ],
    },
  ],
};
