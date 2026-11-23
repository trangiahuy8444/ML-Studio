import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_08: GradModule = {
  id: 8,
  title: 'Xử lý Ngôn ngữ Tự nhiên, Transformer & LLMs',
  englishTitle: 'NLP, Transformers & Large Language Models',
  color: '#c084fc',
  status: 'ready',
  overview:
    'Từ biểu diễn từ tĩnh Word2Vec (Skip-gram, CBOW, Negative Sampling) tới mạng hồi tiếp RNN/LSTM giải quyết triệt tiêu gradient; cơ chế Bahdanau Attention trong Seq2Seq; kiến trúc Transformer (Scaled Dot-Product, Multi-Head Attention, Positional Encoding); và các mô hình tiền huấn luyện (BERT, PhoBERT tiếng Việt, GPT, Fine-tuning & RAG).',
  outcomes: [
    'Dẫn xuất hàm mục tiêu Negative Sampling của Word2Vec và cài đặt đầy đủ loss + gradient kiểm chứng sai phân hữu hạn.',
    'Phân tích cơ chế cổng giữ luồng gradient bất biến qua Constant Error Carousel trong LSTM và cài đặt tế bào LSTM.',
    'Cài đặt Bahdanau Additive Attention, Scaled Dot-Product Self-Attention có Causal Mask, Positional Encoding và LayerNorm.',
    'Đánh giá quy trình tiền huấn luyện và tinh chỉnh mô hình ngôn ngữ (bao gồm PhoBERT cho tiếng Việt), truy xuất RAG cosine và giải mã Temperature / Top-k.',
  ],
  sources: [
    'Mikolov et al. (2013) — Word2Vec & Negative Sampling',
    'Hochreiter & Schmidhuber (1997) — LSTM; Goldberg (2016)',
    'Bahdanau et al. (2015); Vaswani et al. (2017) — Attention Is All You Need',
    'Nguyen & Nguyen (2020) — PhoBERT; Jurafsky & Martin (Ch. 5, 14)',
  ],
  units: [
    {
      id: 'm8-u1',
      moduleId: 8,
      order: 1,
      title: 'Word2Vec (Skip-gram, CBOW, Negative Sampling) & Dense Embeddings',
      summary:
        'Giả thuyết phân phối (Distributional Hypothesis) và biểu diễn phân tán từ vựng: kiến trúc CBOW vs Skip-gram, nút thắt cổ chai tính toán Softmax trên toàn bộ từ điển $|V|$ và lời giải xấp xỉ đối chiếu nhiễu Skip-gram with Negative Sampling (SGNS).',
      outcomes: [
        'Phân biệt biểu diễn thưa one-hot trực giao với không gian vector dày nắm bắt quan hệ ngữ nghĩa (tương đồng tuyến tính king - man + woman).',
        'Giải thích tại sao Softmax đầy đủ tốn $O(|V|)$ mỗi cập nhật và cách Negative Sampling quy về bài toán phân loại nhị phân $O(K)$.',
        'Dẫn xuất và cài đặt hàm mất mát SGNS cùng gradient giải tích cho vector trung tâm và các vector ngữ cảnh.',
      ],
      references: [
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/02_Mikolov_Word2Vec_Vector_Space_2013.pdf',
          section: 'Mikolov et al. (2013a) — CBOW & Skip-gram',
          note: 'Cấu trúc hai kiến trúc nền tảng học nhúng từ hiệu quả từ ngữ liệu lớn.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/03_Mikolov_Word2Vec_Negative_Sampling_NeurIPS_2013.pdf',
          section: 'Mikolov et al. (2013b) — Negative Sampling & Subsampling',
          note: 'Phân phối Unigram^(3/4) chọn mẫu âm và loại bỏ từ dừng tần suất cao.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/06_Jurafsky_Martin_Ch05_Embeddings.pdf',
          section: 'Jurafsky & Martin (Ch. 5) — Vector Semantics & Embeddings',
          note: 'Liên hệ giữa SGNS với phân rã ma trận PMI dịch chuyển (Levy & Goldberg).',
        },
      ],
      sections: [
        {
          heading: '1. Giới hạn của One-Hot & Mô hình Skip-gram',
          body: String.raw`Mã hóa one-hot biến mỗi từ thành vector chiều $|V| \sim 10^5$ trực giao đôi một ($e_i^\top e_j = 0$ kể cả giữa "xe máy" và "mô tô"). Word2Vec học hai ma trận nhúng dày chiều $d \ll |V|$ ($W_{\text{center}}, W_{\text{context}} \in \mathbb{R}^{|V|\times d}$) để dự đoán từ ngữ cảnh $o$ xung quanh từ trung tâm $c$ trong cửa sổ bán kính $m$:`,
          math: String.raw`P(w_{t+j} = o \mid w_t = c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in V} \exp(u_w^\top v_c)}`,
          intuition:
            'Levy & Goldberg (2014) chứng minh Skip-gram với Negative Sampling thực chất đang ngầm phân rã ma trận Pointwise Mutual Information dịch chuyển: $u_w^\\top v_c \\approx \\text{PMI}(w, c) - \\log K$.',
        },
        {
          heading: '2. Skip-gram with Negative Sampling (SGNS) & Dẫn xuất Gradient',
          body: String.raw`Thay vì tính mẫu số Softmax qua $|V|$ từ ở mỗi bước SGD, **SGNS** tối ưu xác suất phân biệt cặp thật $(c, o)$ khỏi $K$ từ nhiễu $\{n_k\}_{k=1}^K$ bốc theo phân phối $P_n(w) \propto U(w)^{3/4}$ (nâng luỹ thừa $3/4$ để tăng cơ hội chọn các từ ít phổ biến hơn):`,
          math: String.raw`\mathcal{L}_{\text{SGNS}} = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^K \log \sigma(-u_{n_k}^\top v_c)`,
          pitfall:
            'Word2Vec học **một vector tĩnh duy nhất** cho mỗi mục từ, dẫn đến bất lực trước hiện tượng đa nghĩa (Polysemy: "đường" giao thông vs "đường" ăn không phân biệt được theo ngữ cảnh câu) — động lực trực tiếp cho mô hình chuỗi và tiền huấn luyện theo ngữ cảnh ở các bài sau.',
        },
      ],
      simulationId: 'm8-u1',
      exercises: [
        {
          id: 'm8-u1-sgns-loss',
          title: 'Hàm mục tiêu Skip-gram Negative Sampling (SGNS) & Gradients',
          prompt:
            'Cài đặt `sgns_loss_and_grads(v_c, u_o, U_neg)` tính hàm mất mát SGNS $\\mathcal{L}$ và 3 gradient $\\nabla_{v_c}\\mathcal{L}$, $\\nabla_{u_o}\\mathcal{L}$, $\\nabla_{U_{\\text{neg}}}\\mathcal{L}$:\n- `v_c` shape `(d,)`, `u_o` shape `(d,)`, `U_neg` shape `(K, d)`.\n- Dùng dạng ổn định số học của `-log(sigmoid(z)) = log(1 + exp(-z))` (`np.logaddexp(0.0, -z)`).\n- Trả về `(loss, grad_vc, grad_uo, grad_Uneg)`.',
          hint: 'Với $s_{\\text{pos}} = \\sigma(u_o^\\top v_c)$ và $s_{\\text{neg}} = \\sigma(-U_{\\text{neg}} v_c)$ shape `(K,)`: đạo hàm $\\frac{\\partial}{\\partial z}(-\\log\\sigma(z)) = \\sigma(z) - 1$. Do dấu âm trước $u_{n_k}^\\top v_c$, đóng góp âm vào $v_c$ là `(1 - s_neg)[:, None] * U_neg` tổng theo trục 0, và `grad_Uneg = (1 - s_neg)[:, None] * v_c[None, :]`.',
          setup:
            COMMON_SETUP +
            String.raw`
_vc_w2v = np.array([0.6, -0.4, 0.8, 0.2])
_uo_w2v = np.array([0.5, 0.3, -0.2, 0.9])
_Uneg_w2v = np.array([[0.3, -0.7, 0.4, 0.1], [-0.5, 0.2, 0.6, -0.3], [0.1, 0.8, -0.5, 0.4]])
`,
          starter: String.raw`def sgns_loss_and_grads(v_c, u_o, U_neg):
    raise NotImplementedError
`,
          solution: String.raw`def sgns_loss_and_grads(v_c, u_o, U_neg):
    v_c = np.asarray(v_c, dtype=float)
    u_o = np.asarray(u_o, dtype=float)
    U_neg = np.asarray(U_neg, dtype=float)

    def _sig(z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    z_pos = float(u_o @ v_c)
    z_neg = -(U_neg @ v_c)
    loss = float(np.logaddexp(0.0, -z_pos) + np.sum(np.logaddexp(0.0, -z_neg)))

    s_pos = float(_sig(z_pos))
    s_neg = _sig(z_neg)
    coeff_neg = 1.0 - s_neg  # sigma(+U_neg @ v_c)

    grad_uo = (s_pos - 1.0) * v_c
    grad_Uneg = coeff_neg[:, None] * v_c[None, :]
    grad_vc = (s_pos - 1.0) * u_o + np.sum(coeff_neg[:, None] * U_neg, axis=0)
    return loss, grad_vc, grad_uo, grad_Uneg
`,
          test: String.raw`def _run_checks():
    def _shape_chk():
        loss, gvc, guo, gun = sgns_loss_and_grads(_vc_w2v, _uo_w2v, _Uneg_w2v)
        return (
            np.isfinite(loss) and gvc.shape == _vc_w2v.shape and guo.shape == _uo_w2v.shape and gun.shape == _Uneg_w2v.shape,
            f'loss={loss:.4f}'
        )
    c1 = _check('Shape loss & 3 vector gradient hop le', _shape_chk)
    def _fd_chk():
        loss, gvc, guo, gun = sgns_loss_and_grads(_vc_w2v, _uo_w2v, _Uneg_w2v)
        eps = 1e-6
        pairs = [(_vc_w2v, gvc), (_uo_w2v, guo), (_Uneg_w2v, gun)]
        max_err = 0.0
        for arr, ana in pairs:
            num = np.zeros_like(arr)
            for idx in np.ndindex(*arr.shape):
                old = arr[idx]
                arr[idx] = old + eps
                lp, _, _, _ = sgns_loss_and_grads(_vc_w2v, _uo_w2v, _Uneg_w2v)
                arr[idx] = old - eps
                lm, _, _, _ = sgns_loss_and_grads(_vc_w2v, _uo_w2v, _Uneg_w2v)
                arr[idx] = old
                num[idx] = (lp - lm) / (2.0 * eps)
            max_err = max(max_err, float(np.max(np.abs(num - ana))))
        return max_err < 1e-7, f'max_grad_diff={max_err:.2e}'
    c2 = _check('Kiem chung sai phan huu han < 1e-7 cho vc, uo va U_neg', _fd_chk)
    _report('m8-u1-sgns-loss', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Đối với tiếng Việt (như trong PhoBERT hay kho văn bản Việt ngữ), tại sao tách từ theo dấu cách đơn thuần (whitespace tokenization) dễ làm méo không gian nhúng Word2Vec?',
          guidance:
            'Tiếng Việt là ngôn ngữ đơn lập với từ ghép đa âm tiết chiếm đa số (ví dụ "máy tính", "học máy", "sinh viên"). Nếu không phân tách từ (`VnCoreNLP` / `underthesea` hoặc Subword BPE), từng âm tiết đơn bị gán ngữ cảnh hỗn tạp; trong thực tế tiếng Việt dùng kết hợp Word Segmentation hoặc BPE.',
        },
        {
          question:
            'Vì sao Mikolov et al. (2013) chọn lũy thừa $P_n(w) \\propto U(w)^{3/4}$ thay vì phân phối đồng đều hoặc nguyên bản unigram $U(w)$?',
          guidance:
            'Phân phối đồng đều $1/|V|$ chọn quá nhiều từ cực hiếm không mang tín hiệu đối chiếu mạnh, còn $U(w)$ thuần túy khiến từ dừng tần suất cao chiếm hết mẫu âm. Lũy thừa $0.75$ làm mượt phần đuôi phân phối, tăng vừa phải xác suất lấy mẫu các từ tần suất trung bình/thấp.',
        },
      ],
    },
    {
      id: 'm8-u2',
      moduleId: 8,
      order: 2,
      title: 'RNN, BPTT & LSTM/GRU (Giải quyết Gradient triệt tiêu)',
      summary:
        'Mô hình hóa chuỗi có thứ tự bằng Mạng Nơ-ron Hồi tiếp (RNN), phân tích toán học hiện tượng bùng nổ/triệt tiêu gradient qua thời gian (BPTT) theo giá trị kỳ dị ma trận $W_{hh}$, và kiến trúc LSTM với đường cao tốc ô nhớ (Constant Error Carousel).',
      outcomes: [
        'Giải thích tích lũy lũy thừa $\\prod_{k=t_1}^{t_2} W_{hh}^\\top \\text{diag}(1 - \\tanh^2(z_k))$ gây triệt tiêu hoặc bùng nổ gradient trong Vanilla RNN.',
        'Chứng minh cấu trúc cập nhật cộng tính $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$ trong LSTM giữ được luồng đạo hàm qua hàng trăm bước.',
        'Cài đặt trọn vẹn phép chuyển tiếp 1 bước của tế bào LSTM với 4 cổng xếp chồng `[i, f, g, o]`.',
      ],
      references: [
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/04_Hochreiter_Schmidhuber_LSTM_1997.pdf',
          section: 'Hochreiter & Schmidhuber (1997) — Long Short-Term Memory',
          note: 'Bài báo kinh điển đề xuất ô nhớ cộng tính giải quyết bài toán phụ thuộc xa.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/06b_Jurafsky_Martin_Ch14_RNNs_and_LSTMs.pdf',
          section: 'Jurafsky & Martin (Ch. 14) — RNNs, BPTT, Gradient Clipping & LSTM/GRU',
          note: 'Phương trình cổng quên, cổng vào, cổng ra và tính đối ngẫu thuận-nghịch.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/05_Yoav_Goldberg_NN_Primer_NLP_2016.pdf',
          section: 'Yoav Goldberg (2016) — RNNs, LSTM & Highway Connections in NLP',
          note: 'Tổng hợp chuẩn thực hành NLP với LSTM đa tầng và BiLSTM.',
        },
      ],
      sections: [
        {
          heading: '1. Vanilla RNN & Vấn đề BPTT (Vanishing / Exploding Gradient)',
          body: String.raw`Mạng RNN cập nhật trạng thái ẩn $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ dùng chung tham số qua mọi bước thời gian $t$. Khi lan truyền ngược qua thời gian (BPTT) từ bước $T$ về bước $k \ll T$, gradient nhân liên tiếp cùng ma trận $W_{hh}^\top$:`,
          math: String.raw`\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^T \text{diag}\big(1 - h_t^2\big)\, W_{hh}^\top`,
          pitfall:
            'Nếu giá trị kỳ dị lớn nhất của $W_{hh}$ nhỏ hơn $1$ (hoặc $\\tanh$ bão hòa khiến $1-h_t^2 \\to 0$), chuẩn gradient phân rã theo hàm mũ $(T-k)$ làm mạng "quên" phụ thuộc xa; ngược lại nếu lớn hơn $1$, gradient bùng nổ thành `NaN` — chỉ ngăn được triệu chứng bằng **Gradient Clipping** $\\|g\\| \\le \\tau$, không chữa được hiện tượng triệt tiêu.',
        },
        {
          heading: '2. LSTM: Đường cao tốc Ô nhớ (Cell State) & Các Cổng điều tiết',
          body: String.raw`LSTM tách biệt bộ nhớ dài hạn (cell state $c_t$) khỏi trạng thái làm việc ($h_t$). Thay vì nhân ma trận lặp lại trực tiếp lên $c_{t-1}$, $c_t$ được cập nhật **cộng tính** thông qua cổng quên $f_t$ và cổng vào $i_t$:`,
          math: String.raw`[i_t, f_t, g_t, o_t] = \text{split}\big(x_t W_x + h_{t-1} W_h + b\big), \qquad c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)`,
          intuition:
            'Khi đạo hàm theo $c_{t-1}$: $\\frac{\\partial c_t}{\\partial c_{t-1}} \\approx \\text{diag}(f_t)$. Nếu cổng quên $f_t \\approx 1$ (thường khởi tạo bias cổng quên $b_f = +1$), luồng thông tin và gradient đi xuyên qua ô nhớ mà không bị nhân ma trận $W_{hh}$ làm suy giảm! Biến thể **GRU** gộp $c_t$ và $h_t$ thành 2 cổng (Update & Reset) giúp giảm ~25% tham số.',
        },
      ],
      simulationId: 'm8-u2',
      exercises: [
        {
          id: 'm8-u2-lstm-cell',
          title: 'Cài đặt 1 bước LSTM Cell & Kiểm chứng Constant Error Carousel',
          prompt:
            'Cài đặt `lstm_step(x_t, h_prev, c_prev, W_x, W_h, b)` cho một mẫu duy nhất:\n- `x_t` `(d_x,)`, `h_prev` `(H,)`, `c_prev` `(H,)`, `W_x` `(d_x, 4*H)`, `W_h` `(H, 4*H)`, `b` `(4*H,)`.\n- Chia pre-activation thành 4 phần bằng nhau theo đúng thứ tự `[i, f, g, o]`.\n- Áp dụng `sigmoid` cho `i, f, o` và `np.tanh` cho `g`.\n- Trả về `(h_t, c_t)` đều có shape `(H,)`.',
          hint: 'Tính `gates = x_t @ W_x + h_prev @ W_h + b`, rồi cắt `H = h_prev.shape[0]`: `ai, af, ag, ao = gates[:H], gates[H:2*H], gates[2*H:3*H], gates[3*H:]`. Luôn kẹp input của `exp(-z)` vào `[-60, 60]` trong sigmoid.',
          setup:
            COMMON_SETUP +
            String.raw`
_H_lstm, _dx_lstm = 3, 2
_xt_lstm = np.array([0.8, -0.5])
_hp_lstm = np.array([0.2, -0.1, 0.4])
_cp_lstm = np.array([1.5, -2.0, 0.7])
_Wx_lstm = np.linspace(-0.4, 0.4, _dx_lstm * 4 * _H_lstm).reshape(_dx_lstm, 4 * _H_lstm)
_Wh_lstm = np.linspace(0.3, -0.3, _H_lstm * 4 * _H_lstm).reshape(_H_lstm, 4 * _H_lstm)
_b_lstm = np.zeros(4 * _H_lstm)
`,
          starter: String.raw`def lstm_step(x_t, h_prev, c_prev, W_x, W_h, b):
    raise NotImplementedError
`,
          solution: String.raw`def lstm_step(x_t, h_prev, c_prev, W_x, W_h, b):
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    c_prev = np.asarray(c_prev, dtype=float)
    H = h_prev.shape[0]

    def _sig(z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    gates = x_t @ W_x + h_prev @ W_h + b
    i_t = _sig(gates[:H])
    f_t = _sig(gates[H:2 * H])
    g_t = np.tanh(gates[2 * H:3 * H])
    o_t = _sig(gates[3 * H:])
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t
`,
          test: String.raw`def _run_checks():
    def _basic_chk():
        ht, ct = lstm_step(_xt_lstm, _hp_lstm, _cp_lstm, _Wx_lstm, _Wh_lstm, _b_lstm)
        return (
            ht.shape == (3,) and ct.shape == (3,) and np.all(np.abs(ht) <= 1.0),
            f'h_t={ht.round(4)}, c_t={ct.round(4)}'
        )
    c1 = _check('Shape & mien gia tri |h_t| <= 1', _basic_chk)
    def _cec_chk():
        b_hw = np.zeros(4 * _H_lstm)
        b_hw[:_H_lstm] = -30.0          # input gate closed (i ~ 0)
        b_hw[_H_lstm:2*_H_lstm] = 30.0  # forget gate open (f ~ 1)
        _, ct = lstm_step(_xt_lstm, _hp_lstm, _cp_lstm, np.zeros_like(_Wx_lstm), np.zeros_like(_Wh_lstm), b_hw)
        return np.allclose(ct, _cp_lstm, atol=1e-7), f'max_err={np.max(np.abs(ct - _cp_lstm)):.2e}'
    c2 = _check('Constant Error Carousel: f~1, i~0 bao toan nguyen ven c_{t-1}', _cec_chk)
    _report('m8-u2-lstm-cell', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao mặc dù có LSTM với Constant Error Carousel, mạng hồi tiếp vẫn gặp giới hạn khi xử lý ngữ cảnh rất dài (hàng nghìn token) và phần cứng GPU hiện đại?',
          guidance:
            'LSTM bắt buộc tính tuần tự $h_1 \\to h_2 \\to \\dots \\to h_T$, không thể song song hóa qua chiều thời gian $T$ trong lúc huấn luyện. Đồng thời mọi thông tin vẫn phải nén qua vector cố định $c_t \\in \\mathbb{R}^H$, tạo nút thắt thông tin khi $T$ dài.',
        },
        {
          question:
            'Trong Backpropagation Through Time cho bài toán phân loại chuỗi hay dịch máy dài, người ta dùng kỹ thuật Truncated BPTT như thế nào và đánh đổi điều gì?',
          guidance:
            'Forward vẫn truyền $h_t$ xuyên suốt chuỗi, nhưng Backward chỉ cắt ngược $K_{\\text{bptt}}$ bước (ví dụ 35 hoặc 64 token) rồi chặn gradient qua `detach()`, giảm bộ nhớ kích hoạt từ $O(T)$ xuống $O(K_{\\text{bptt}})$ nhưng mất khả năng học phụ thuộc vượt quá khoảng cắt.',
        },
      ],
    },
    {
      id: 'm8-u3',
      moduleId: 8,
      order: 3,
      title: 'Seq2Seq & Cơ chế Bahdanau Attention',
      summary:
        'Khắc phục nút thắt nén toàn bộ câu nguồn vào một vector ngữ cảnh duy nhất trong kiến trúc Encoder–Decoder Seq2Seq cổ điển bằng cơ chế Additive Attention (Bahdanau et al., 2015) — bước đệm trực tiếp khai sinh ra Transformer.',
      outcomes: [
        'Phân tích nguyên nhân chất lượng Seq2Seq không có Attention suy giảm nhanh khi độ dài câu $T > 20$ từ.',
        'Viết phương trình căn chỉnh cộng tính (Additive Alignment), trọng số Softmax và vector ngữ cảnh động $c_t$.',
        'Cài đặt cơ chế Bahdanau Attention bằng NumPy cho phép Decoder nhìn trực tiếp vào mọi trạng thái Encoder.',
      ],
      references: [
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/10_Bahdanau_Neural_Machine_Translation_Attention_2015.pdf',
          section: 'Bahdanau, Cho & Bengio (2015) — Jointly Learning to Align and Translate',
          note: 'Công thức Additive Alignment $e_{ti} = v_a^\\top \\tanh(W_a s_{t-1} + U_a h_i)$ và trọng số $\\alpha_{ti}$.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/06b_Jurafsky_Martin_Ch14_RNNs_and_LSTMs.pdf',
          section: 'Jurafsky & Martin (Ch. 14) — Encoder-Decoder & Attention',
          note: 'So sánh Bottleneck Encoder-Decoder với Attention có kết nối tắt trực tiếp.',
        },
      ],
      sections: [
        {
          heading: '1. Nút thắt thông tin (Information Bottleneck) trong Seq2Seq',
          body: String.raw`Kiến trúc Encoder–Decoder nguyên bản (Sutskever et al., 2014) bắt buộc toàn bộ ngữ nghĩa của câu nguồn bất kể dài 5 từ hay 80 từ phải đi qua một vector trạng thái ẩn cuối cùng duy nhất $h_T^{\text{enc}} \in \mathbb{R}^d$. Khi câu càng dài, các token đầu câu bị pha loãng và mất thông tin trong phép đệ quy.`,
        },
        {
          heading: '2. Cơ chế Bahdanau Additive Attention',
          body: String.raw`Thay vì dùng một vector ngữ cảnh tĩnh duy nhất, ở mỗi bước giải mã $t$, **Bahdanau Attention** tính vector ngữ cảnh động $c_t$ là tổng có trọng số lồi của **toàn bộ** các trạng thái ẩn Encoder $\{h_1, \dots, h_T\}$ (thường từ BiLSTM):`,
          math: String.raw`e_{ti} = v_a^\top \tanh\big(W_a^\top s_{t-1} + U_a^\top h_i\big), \qquad \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^T \exp(e_{tj})}, \qquad c_t = \sum_{i=1}^T \alpha_{ti} h_i`,
          intuition:
            'Attention đóng vai trò như bộ nhớ truy cập mềm (soft content-addressable memory): mỗi bước sinh một từ đích mới, Decoder chủ động "truy vấn" xem mình cần dồn tỷ trọng chú ý $\\alpha_{ti}$ vào vị trí từ nguồn nào (kể cả khi trật tự ngữ pháp đảo ngược như Anh–Việt).',
          pitfall:
            'Trong Attention cộng tính, thành phần chiếu $U_a^\\top h_i$ chỉ phụ thuộc vào phía Encoder nên **có thể tính trước một lần duy nhất** cho cả câu nguồn trước vòng lặp giải mã; tính lại $H_{\\text{enc}} U_a$ ở mỗi bước $t$ sẽ làm chậm suy luận tuyến tính theo $T_{\\text{dec}}$.',
        },
      ],
      simulationId: 'm8-u3',
      exercises: [
        {
          id: 'm8-u3-bahdanau-attn',
          title: 'Cài đặt Bahdanau Additive Attention',
          prompt:
            'Cài đặt `bahdanau_attention(s_prev, H_enc, W_a, U_a, v_a)`:\n- `s_prev` shape `(d_dec,)` (trạng thái decoder bước trước).\n- `H_enc` shape `(T, d_enc)` (các trạng thái ẩn encoder).\n- `W_a` shape `(d_dec, d_attn)`, `U_a` shape `(d_enc, d_attn)`, `v_a` shape `(d_attn,)`.\n- Tính điểm `scores = np.tanh(s_prev @ W_a + H_enc @ U_a) @ v_a`, áp dụng softmax ổn định ra `weights` shape `(T,)`, rồi `context = weights @ H_enc` shape `(d_enc,)`.\n- Trả về `(context, weights)`.',
          hint: 'Chú ý broadcast `(d_attn,)` + `(T, d_attn)` cho kết quả `(T, d_attn)`. Dùng `s_shift = scores - np.max(scores)` trước khi lấy `exp` để không bị overflow.',
          setup:
            COMMON_SETUP +
            String.raw`
_s_prev_bah = np.array([0.7, -0.4, 0.2])
_H_enc_bah = np.array([
    [1.0, 0.0, 0.2, -0.1],
    [-0.5, 1.2, 0.4, 0.3],
    [0.2, -0.3, 1.5, 0.8],
    [0.0, 0.5, -0.6, 1.1],
])
_Wa_bah = np.array([[0.5, -0.2], [0.3, 0.6], [-0.4, 0.7]])
_Ua_bah = np.array([[0.4, 0.1], [-0.3, 0.5], [0.6, -0.2], [0.2, 0.4]])
_va_bah = np.array([0.8, -0.5])
`,
          starter: String.raw`def bahdanau_attention(s_prev, H_enc, W_a, U_a, v_a):
    raise NotImplementedError
`,
          solution: String.raw`def bahdanau_attention(s_prev, H_enc, W_a, U_a, v_a):
    s_prev = np.asarray(s_prev, dtype=float)
    H_enc = np.asarray(H_enc, dtype=float)
    W_a = np.asarray(W_a, dtype=float)
    U_a = np.asarray(U_a, dtype=float)
    v_a = np.asarray(v_a, dtype=float)

    scores = np.tanh(s_prev @ W_a + H_enc @ U_a) @ v_a
    shifted = scores - np.max(scores)
    exp_s = np.exp(shifted)
    weights = exp_s / np.sum(exp_s)
    context = weights @ H_enc
    return context, weights
`,
          test: String.raw`def _run_checks():
    def _chk_val():
        ctx, w = bahdanau_attention(_s_prev_bah, _H_enc_bah, _Wa_bah, _Ua_bah, _va_bah)
        ref_scores = np.tanh(_s_prev_bah @ _Wa_bah + _H_enc_bah @ _Ua_bah) @ _va_bah
        e = np.exp(ref_scores - np.max(ref_scores))
        ref_w = e / np.sum(e)
        ref_c = ref_w @ _H_enc_bah
        ok = ctx.shape == (4,) and w.shape == (4,) and abs(float(w.sum()) - 1.0) < 1e-8 and np.allclose(w, ref_w) and np.allclose(ctx, ref_c)
        return ok, f'sum_w={w.sum():.6f}, max_c_err={np.max(np.abs(ctx - ref_c)):.2e}'
    c1 = _check('Trong so Attention co tong bang 1 & Context vector dung gia tri', _chk_val)
    def _chk_shift():
        # Translation invariance of softmax scores
        _, w1 = bahdanau_attention(_s_prev_bah, _H_enc_bah, _Wa_bah, _Ua_bah, _va_bah)
        _, w2 = bahdanau_attention(_s_prev_bah, _H_enc_bah, _Wa_bah, _Ua_bah, 2.0 * _va_bah)
        return float(np.max(w2)) > float(np.max(w1)), f'peak(v)={w1.max():.3f}, peak(2v)={w2.max():.3f}'
    c2 = _check('Tang bien do v_a lam phan phoi chu y nhon hon (low entropy)', _chk_shift)
    _report('m8-u3-bahdanau-attn', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'So sánh độ phức tạp tính toán và số tham số giữa Additive Attention (Bahdanau) và Multiplicative / Dot-Product Attention (Luong / Vaswani)?',
          guidance:
            'Bahdanau đòi hỏi ma trận trung gian cộng $W_a s_{t-1} + U_a h_i$ rồi áp $\\tanh$, tạo tensor $(T_{\\text{dec}}, T_{\\text{enc}}, d_{\\text{attn}})$ tốn băng thông bộ nhớ và khó tận dụng thuần phép nhân ma trận $QK^\\top$. Multiplicative/Dot-Product Attention chỉ là một phép GEMM $Q K^\\top$ cực nhanh trên GPU.',
        },
        {
          question:
            'Tại sao bản đồ Attention trọng số $\\alpha_{ti}$ giải thích được sự căn chỉnh từ (word alignment) nhưng không phải lúc nào cũng là lời giải thích nhân quả đáng tin cậy hoàn toàn cho dự đoán của mô hình?',
          guidance:
            'Đầu ra mô hình phụ thuộc vào nhiều tầng biến đổi phi tuyến sau khi nhân $\\alpha_{ti} h_i$ (và đặc biệt trong các kiến trúc nhiều đầu có kết nối phần dư). Hai bản phân bố $\\alpha$ khác nhau đôi khi cho cùng dự đoán (Jain & Wallace, 2019), nên chỉ nên xem Attention là tín hiệu gợi ý, cần kết hợp kiểm chứng gradient/occlusion.',
        },
      ],
    },
    {
      id: 'm8-u4',
      moduleId: 8,
      order: 4,
      title: 'Transformer (Multi-Head Attention, Positional Encoding, LayerNorm)',
      summary:
        'Kiến trúc Transformer (Vaswani et al., 2017) loại bỏ hoàn toàn phép đệ quy tuần tự: Scaled Dot-Product Attention (cùng giải thích hệ số $1/\\sqrt{d_k}$), Causal Masking cho sinh tự hồi quy, Multi-Head Attention, mã hóa vị trí Sinusoidal/RoPE và Layer Normalization.',
      outcomes: [
        'Chứng minh thống kê tại sao phương sai tích vô hướng $q^\\top k$ tăng tỷ lệ thuận với $d_k$, đòi hỏi chia $\\sqrt{d_k}$ để tránh bão hòa Softmax.',
        'Cài đặt Scaled Dot-Product Attention hỗ trợ Causal Mask tam giác trên nghiêm ngặt cho Decoder tự hồi quy.',
        'Cài đặt Sinusoidal Positional Encoding và Pre-/Post-LayerNorm chuẩn hóa theo trục đặc trưng cuối cùng.',
      ],
      references: [
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/08_Vaswani_Attention_Is_All_You_Need_2017.pdf',
          section: 'Vaswani et al. (2017) — Attention Is All You Need (§3.1–3.5)',
          note: 'Scaled Dot-Product Attention, Multi-Head Projections, FFN và Positional Encoding.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§15.4 (Transformers, Self-Attention & Autoregressive Masking)',
          note: 'Phân tích độ phức tạp $O(T^2 d)$ và vai trò kết nối tắt Pre-LN.',
        },
      ],
      sections: [
        {
          heading: '1. Scaled Dot-Product Attention & Hệ số chia √d_k',
          body: String.raw`Với các ma trận Truy vấn $Q \in \mathbb{R}^{T_q \times d_k}$, Khóa $K \in \mathbb{R}^{T_k \times d_k}$ và Giá trị $V \in \mathbb{R}^{T_k \times d_v}$, tích vô hướng giữa hai vector ngẫu nhiên chuẩn độc lập phương sai $1$ có trung bình $0$ nhưng **phương sai bằng $d_k$**. Nếu không chia $\sqrt{d_k}$, khi $d_k = 64$ hay $128$, một vài giá trị cực đại sẽ đẩy Softmax về trạng thái gần one-hot khiến gradient triệt tiêu:`,
          math: String.raw`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V`,
          intuition:
            'Để ngăn token ở vị trí $i$ nhìn trộm token tương lai $j > i$ trong mô hình ngôn ngữ tự hồi quy (như GPT), mặt nạ **Causal Mask** đặt $M_{ij} = -\\infty$ với mọi $j > i$, biến trọng số $\\exp(-\\infty) = 0$ chính xác.',
        },
        {
          heading: '2. Multi-Head Attention, Positional Encoding & LayerNorm',
          body:
            String.raw`- **Multi-Head Attention**: chiếu $Q, K, V$ sang $h$ không gian con chiều $d_k = d_{\text{model}}/h$ song song để từng head chuyên trách quan hệ khác nhau (cú pháp, đồng tham chiếu, vị trí gần), rồi nối ghép (` +
            '`concat`' +
            String.raw`) và chiếu qua $W^O$.
- **Positional Encoding**: vì Self-Attention thuần túy là phép **đẳng biến hoán vị** (không phân biệt thứ tự từ!), Transformer cộng thêm vector mã hóa vị trí $\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}),\; \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$ (các LLM hiện đại chuyển sang **RoPE** — xoay trực tiếp $Q, K$ theo góc vị trí tương đối).
- **Pre-LN vs Post-LN**: chuẩn hóa **LayerNorm** $y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$ đặt *trước* các khối Attention/FFN (**Pre-LN**) giữ luồng gradient qua Residual đường thẳng trơn tru, loại bỏ nhu cầu warm-up tinh vi ở Post-LN gốc 2017.`,
          pitfall:
            'Khác biệt chi phí cốt lõi khi chuỗi dài $T$ tăng gấp $10\\times$: Self-Attention tiêu tốn $O(T^2 d)$ thời gian và bộ nhớ ma trận chú ý — lý do xuất hiện FlashAttention (tối ưu IO SRAM không lưu tường minh $T\\times T$ trên HBM) và KV-Cache khi sinh token.',
        },
      ],
      simulationId: 'm8-u4',
      exercises: [
        {
          id: 'm8-u4-scaled-attn',
          title: 'Scaled Dot-Product Attention & Causal Mask',
          prompt:
            'Cài đặt `scaled_dot_product_attention(Q, K, V, causal_mask=False)` với `Q` `(Tq, dk)`, `K` `(Tk, dk)`, `V` `(Tk, dv)`:\n1. Tính `scores = (Q @ K.T) / np.sqrt(dk)`.\n2. Nếu `causal_mask=True`, gán `-1e9` vào phần tam giác trên nghiêm ngặt (`j > i`, dùng `np.triu(..., k=1)`).\n3. Tính hàng Softmax ổn định ra `attn_weights` `(Tq, Tk)` và đầu ra `out = attn_weights @ V` `(Tq, dv)`.\nTrả về `(out, attn_weights)`.',
          hint: 'Tạo mặt nạ `mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1)` rồi `scores = np.where(mask, -1e9, scores)`. Softmax ổn định theo hàng `axis=-1`: trừ `scores.max(axis=-1, keepdims=True)` trước khi lấy `np.exp`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Q_tx = np.array([[1.0, 0.5, -0.2, 0.8], [0.2, 1.2, 0.4, -0.5], [-0.4, 0.3, 1.1, 0.6], [0.9, -0.2, 0.3, 1.0]])
_K_tx = np.array([[0.8, 0.2, 0.1, 0.5], [0.1, 1.0, 0.6, -0.3], [-0.5, 0.4, 1.2, 0.7], [0.6, -0.1, 0.2, 1.3]])
_V_tx = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0], [-2.0, 8.0]])
`,
          starter: String.raw`def scaled_dot_product_attention(Q, K, V, causal_mask=False):
    raise NotImplementedError
`,
          solution: String.raw`def scaled_dot_product_attention(Q, K, V, causal_mask=False):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)
    if causal_mask:
        mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(shifted)
    attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    out = attn_weights @ V
    return out, attn_weights
`,
          test: String.raw`def _run_checks():
    def _chk_bi():
        out, w = scaled_dot_product_attention(_Q_tx, _K_tx, _V_tx, causal_mask=False)
        return (
            out.shape == (4, 2) and w.shape == (4, 4) and np.allclose(w.sum(axis=1), 1.0, atol=1e-7)
            and np.allclose(out, w @ _V_tx, atol=1e-8),
            f'row_sums={w.sum(axis=1).round(6)}'
        )
    c1 = _check('Bidirectional Attention tong hang = 1 & out = W @ V', _chk_bi)
    def _chk_causal():
        out, w = scaled_dot_product_attention(_Q_tx, _K_tx, _V_tx, causal_mask=True)
        upper = np.triu(w, k=1)
        return (
            np.all(np.abs(upper) < 1e-8) and np.allclose(out[0], _V_tx[0], atol=1e-8),
            f'max_future_weight={np.max(np.abs(upper)):.2e}, out0={out[0].round(3)}'
        )
    c2 = _check('Causal Mask triet tieu toan bo tuong lai (=0) & hang dau bang V[0]', _chk_causal)
    _report('m8-u4-scaled-attn', c1, c2)
_run_checks()
`,
        },
        {
          id: 'm8-u4-pos-layernorm',
          title: 'Sinusoidal Positional Encoding & Layer Normalization',
          prompt:
            'Cài đặt hai thành phần lõi của Transformer:\n1. `sinusoidal_positional_encoding(max_len, d_model)` (giả sử `d_model` chẵn): trả về mảng `(max_len, d_model)` với `PE[pos, 2i] = sin(pos / 10000^(2i/d_model))` và `PE[pos, 2i+1] = cos(...)`.\n2. `layer_norm(X, gamma, beta, eps=1e-5)`: chuẩn hóa dọc theo trục cuối cùng (`axis=-1`) dùng phương sai tổng thể (`ddof=0`) và trả về `gamma * x_hat + beta`.',
          hint: 'Với `PE`: `pos = np.arange(max_len, dtype=float)[:, None]`, `div = np.exp(np.arange(0, d_model, 2, dtype=float) * (-np.log(10000.0) / d_model))[None, :]`, gán `pe[:, 0::2] = np.sin(pos * div)` và `pe[:, 1::2] = np.cos(pos * div)`. Với `layer_norm`: `mean = X.mean(axis=-1, keepdims=True)`, `var = X.var(axis=-1, keepdims=True)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_X_ln = np.array([
    [[1.0, 2.0, 3.0, 6.0], [-2.0, 0.0, 2.0, 4.0]],
    [[5.0, 5.0, 7.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
])
_gamma_ln = np.array([1.5, 0.5, 2.0, 1.0])
_beta_ln = np.array([0.0, 1.0, -0.5, 0.25])
`,
          starter: String.raw`def sinusoidal_positional_encoding(max_len, d_model):
    raise NotImplementedError

def layer_norm(X, gamma, beta, eps=1e-5):
    raise NotImplementedError
`,
          solution: String.raw`def sinusoidal_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=float)
    pos = np.arange(max_len, dtype=float)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=float) * (-np.log(10000.0) / d_model))[None, :]
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe

def layer_norm(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    x_hat = (X - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta
`,
          test: String.raw`def _run_checks():
    def _chk_pe():
        pe = sinusoidal_positional_encoding(10, 8)
        unit_pairs = pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2
        return (
            pe.shape == (10, 8) and np.allclose(pe[0, 0::2], 0.0) and np.allclose(pe[0, 1::2], 1.0)
            and np.allclose(unit_pairs, 1.0, atol=1e-7),
            f'pe0={pe[0].round(3)}'
        )
    c1 = _check('Positional Encoding thoa sin^2 + cos^2 = 1 va PE[0] = [0,1,0,1,...]', _chk_pe)
    def _chk_ln():
        out = layer_norm(_X_ln, _gamma_ln, _beta_ln)
        raw_norm = layer_norm(_X_ln, np.ones(4), np.zeros(4))
        return (
            out.shape == _X_ln.shape
            and np.allclose(raw_norm.mean(axis=-1), 0.0, atol=1e-7)
            and np.allclose(raw_norm.var(axis=-1), 1.0, atol=1e-4),
            f'raw_mean_max={np.max(np.abs(raw_norm.mean(axis=-1))):.2e}'
        )
    c2 = _check('LayerNorm chuan hoa tung vector token co mean=0 va var=1', _chk_ln)
    _report('m8-u4-pos-layernorm', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao kích thước thực tế khi sinh tự hồi quy một câu 1.000 token lại không phải $O(T^2)$ mỗi bước nhờ **KV Cache**, nhưng KV Cache lại gây nghẽn ở đâu trên GPU?',
          guidance:
            'Khi sinh token thứ $t$, $K_{1:t-1}$ và $V_{1:t-1}$ từ các bước trước không đổi nên được lưu cache, khiến mỗi bước giải mã chỉ tính 1 truy vấn $q_t K_{1:t}^\\top$ ($O(t d)$ thay vì $O(t^2 d)$). Đổi lại KV Cache tăng tuyến tính theo batch size $\\times$ độ dài chuỗi $\\times$ số tầng, biến suy luận LLM thành bài toán nghẽn băng thông bộ nhớ (memory-bandwidth bound), thúc đẩy Multi-Query / Grouped-Query Attention (GQA) và PagedAttention.',
        },
        {
          question:
            'Phân biệt mã hóa vị trí tuyệt đối (Sinusoidal 2017) với **RoPE (Rotary Position Embedding)** dùng trong LLaMA/Qwen về khả năng ngoại suy vị trí tương đối?',
          guidance:
            'Sinusoidal cộng vào embedding đầu vào nên tín hiệu vị trí bị mờ dần qua các tầng sâu và cặp tích vô hướng $q_m^\\top k_n$ chứa các thành phần chéo hỗn hợp. RoPE xoay trực tiếp vector $Q_m$ và $K_n$ theo góc $m\\theta, n\\theta$ nên tích vô hướng $Q_m^\\top K_n$ chỉ phụ thuộc chính xác vào khoảng cách tương đối $(m - n)\\theta$.',
        },
      ],
    },
    {
      id: 'm8-u5',
      moduleId: 8,
      order: 5,
      title: 'Pre-trained LMs (BERT, PhoBERT tiếng Việt, GPT), Fine-tuning & RAG',
      summary:
        'Kỷ nguyên Tiền huấn luyện tự giám sát (Self-Supervised Pre-training): Encoder-only (Masked LM trong BERT & **PhoBERT** chuyên biệt tiếng Việt có Word Segmentation), Decoder-only (Causal LM trong GPT), tinh chỉnh hiệu quả tham số (LoRA), giải mã Temperature/Top-k và Retrieval-Augmented Generation (RAG).',
      outcomes: [
        'Phân biệt mục tiêu tiền huấn luyện Masked Language Modeling (hai chiều — Encoder) với Next-Token Prediction (tự hồi quy — Decoder).',
        'Hiểu kiến trúc và bước xử lý từ ghép đặc thù của **PhoBERT** (Nguyen & Nguyen, 2020) đối với các tác vụ NLP tiếng Việt.',
        'Cài đặt hàm truy xuất tương đồng Cosine Top-k cho RAG và biến đổi phân phối xác suất sinh từ với Temperature & Top-k Filtering.',
      ],
      references: [
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/09_Nguyen_PhoBERT_Pretrained_Language_Model_2020.pdf',
          section: 'Nguyen & Nguyen (2020) — PhoBERT: Pre-trained Language Models for Vietnamese',
          note: 'Vai trò bắt buộc của phân tách từ tiếng Việt (RDRSegmenter) + RoBERTa MLM trên 20GB văn bản tiếng Việt.',
        },
        {
          source: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/08_Vaswani_Attention_Is_All_You_Need_2017.pdf',
          section: 'Encoder-only vs Decoder-only Variants & Decoding Strategies',
          note: 'Bước chuyển từ kiến trúc Encoder–Decoder sang các mô hình nền tảng quy mô lớn.',
        },
      ],
      sections: [
        {
          heading: '1. BERT / PhoBERT (Encoder) vs GPT (Decoder) & Tinh chỉnh LoRA',
          body: String.raw`- **BERT & RoBERTa**: huấn luyện bằng **Masked Language Modeling (MLM)** — che ngẫu nhiên 15% token và dùng Self-Attention hai chiều dự đoán lại, cực mạnh cho phân loại văn bản, gán nhãn thực thể (NER) và trích xuất đặc trưng dày.
- **PhoBERT (Nguyen & Nguyen, 2020)**: mô hình tiền huấn luyện đột phá cho tiếng Việt. Thay vì đưa trực tiếp từng âm tiết rời rạc vào BPE như XLM-R, PhoBERT thực hiện **Word Segmentation** nối các âm tiết thành từ ghép (ví dụ ` + '`sinh_viên`' + String.raw`, ` + '`học_máy`' + String.raw`) trước khi tách subword, vượt trội trên POS, Parsing, NER và NLI tiếng Việt.
- **GPT (Decoder-only)**: dùng Causal Mask huấn luyện dự đoán token kế tiếp (CLM) quy mô lớn. Khi tinh chỉnh cho miền mới mà không đủ VRAM cập nhật toàn bộ tỷ tham số, **LoRA** đóng băng trọng số gốc $W_0$ và chỉ học cập nhật hạng thấp $W = W_0 + \frac{\alpha}{r} B A$ ($r \ll d$).`,
          math: String.raw`\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}), \qquad W_{\text{LoRA}} = W_0 + \frac{\alpha}{r} B A \;\;(B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k})`,
        },
        {
          heading: '2. Retrieval-Augmented Generation (RAG) & Giải mã xác suất',
          body: String.raw`LLM chỉ lưu tri thức tĩnh trong trọng số nên dễ "ảo giác" (Hallucination) với dữ liệu nội bộ hoặc thông tin mới cập nhật. **RAG** kết nối LLM với kho vector: nhúng đoạn văn bản thành $D \in \mathbb{R}^{M \times d}$, tính độ tương đồng Cosine với truy vấn $q$, lấy top-$k$ ngữ cảnh chèn vào prompt kèm trích dẫn nguồn, rồi điều chỉnh độ đa dạng đầu ra qua nhiệt độ $T$ và cắt lọc **Top-$k$ / Top-$p$**:`,
          math: String.raw`\text{sim}_{\cos}(q, d_i) = \frac{q^\top d_i}{\|q\|_2 \|d_i\|_2 + \varepsilon}, \qquad p_i = \frac{\exp(z_i / T)}{\sum_{j \in \mathcal{V}_{\text{top-}k}} \exp(z_j / T)} \;\;\text{với } i \in \mathcal{V}_{\text{top-}k}`,
          pitfall:
            'Đánh giá hệ thống RAG chỉ bằng điểm LLM cuối cùng dễ che lấp lỗi gốc: bắt buộc tách biệt đánh giá **Retriever** (Recall@k, MRR trên tài liệu chuẩn) khỏi đánh giá **Generator** (Faithfulness — bám sát ngữ cảnh được cấp, không bịa đặt ngoài nguồn).',
        },
      ],
      simulationId: 'm8-u5',
      exercises: [
        {
          id: 'm8-u5-rag-retrieval',
          title: 'RAG Cosine Top-k Retrieval & Temperature Top-k Decoding',
          prompt:
            'Cài đặt hai hàm lõi cho đường ống LLM + RAG:\n1. `cosine_topk_retrieve(query_vec, doc_matrix, k=3)`: tính độ tương đồng Cosine giữa `query_vec` `(d,)` và từng hàng trong `doc_matrix` `(M, d)`, trả về `(indices, scores)` chứa top `k` kết quả sắp xếp giảm dần theo điểm tương đồng.\n2. `softmax_temperature_sample_probs(logits, temperature=1.0, top_k=None)`: chia `logits` cho `temperature > 0`; nếu `top_k` khác `None`, giữ lại duy nhất `top_k` logit lớn nhất (còn lại gán `-1e12`), sau đó áp dụng Softmax ổn định và trả về vector xác suất.',
          hint: 'Chuẩn hóa vector bằng `np.maximum(np.linalg.norm(...), 1e-12)`. Sắp xếp giảm dần bằng `np.argsort(-sims, kind="mergesort")[:k]`. Trong `top_k`: tìm ngưỡng `thresh = np.partition(scaled, -top_k)[-top_k]` hoặc lấy chỉ số `idx = np.argsort(-scaled)[:top_k]` và khởi tạo mảng mask `-1e12`.',
          setup:
            COMMON_SETUP +
            String.raw`
_docs_rag = np.array([
    [1.0, 0.0, 0.2, 0.0],
    [0.0, 1.0, 0.1, 0.0],
    [0.9, 0.1, 0.3, 0.0],
    [-1.0, 0.0, 0.0, 0.5],
    [0.4, 0.4, 0.9, 0.1],
])
_q_rag = np.array([2.0, 0.1, 0.5, 0.0])  # scaled amplitude shouldn't skew cosine vs dot
_logits_dec = np.array([2.5, 1.2, 0.4, -0.8, 3.0])
`,
          starter: String.raw`def cosine_topk_retrieve(query_vec, doc_matrix, k=3):
    raise NotImplementedError

def softmax_temperature_sample_probs(logits, temperature=1.0, top_k=None):
    raise NotImplementedError
`,
          solution: String.raw`def cosine_topk_retrieve(query_vec, doc_matrix, k=3):
    q = np.asarray(query_vec, dtype=float)
    D = np.asarray(doc_matrix, dtype=float)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-12)
    d_norms = np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    sims = (D / d_norms) @ q_norm
    k_eff = min(int(k), D.shape[0])
    indices = np.argsort(-sims, kind='mergesort')[:k_eff]
    return indices.astype(int), sims[indices]

def softmax_temperature_sample_probs(logits, temperature=1.0, top_k=None):
    z = np.asarray(logits, dtype=float) / float(temperature)
    if top_k is not None and 0 < int(top_k) < z.shape[0]:
        k = int(top_k)
        keep_idx = np.argsort(-z, kind='mergesort')[:k]
        masked = np.full_like(z, -1e12)
        masked[keep_idx] = z[keep_idx]
        z = masked
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z)
`,
          test: String.raw`def _run_checks():
    def _chk_ret():
        idx, sc = cosine_topk_retrieve(_q_rag, _docs_rag, k=3)
        return (
            idx.shape == (3,) and sc.shape == (3,) and idx[0] in (0, 2)
            and bool(np.all(sc[:-1] >= sc[1:])) and sc[0] > 0.95 and sc[2] > 0.5,
            f'top_idx={idx.tolist()}, scores={sc.round(4).tolist()}'
        )
    c1 = _check('Cosine Top-k tra dung thu tu giam dan va bat bien theo chuan vector', _chk_ret)
    def _chk_dec():
        p_t1 = softmax_temperature_sample_probs(_logits_dec, temperature=1.0, top_k=None)
        p_cold = softmax_temperature_sample_probs(_logits_dec, temperature=0.3, top_k=2)
        non_top2 = np.delete(p_cold, [0, 4])
        return (
            abs(float(p_t1.sum()) - 1.0) < 1e-8 and abs(float(p_cold.sum()) - 1.0) < 1e-8
            and np.all(non_top2 < 1e-9) and p_cold[4] > p_t1[4],
            f'p_cold={p_cold.round(4).tolist()}'
        )
    c2 = _check('Temperature thap lam nhon phan phoi & Top-k=2 triet tieu xac suat con lai', _chk_dec)
    _report('m8-u5-rag-retrieval', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Tại sao cùng mô hình tiếng Việt có từ điển ~64,000 subword, **PhoBERT** yêu cầu đầu vào đã tách từ ghép bằng `VnCoreNLP` còn **XLM-RoBERTa** hay **GPT** đa ngôn ngữ thì không?',
          guidance:
            'PhoBERT huấn luyện BPE trên đơn vị **từ đã phân tách** (`máy_học` là 1 đơn vị từ trước khi chia subword), nên attention trực tiếp nắm ngữ nghĩa mức từ tiếng Việt. Ngược lại, XLM-R dùng SentencePiece trực tiếp trên chuỗi ký tự thô đa ngôn ngữ để tương thích hàng trăm ngôn ngữ mà không phụ thuộc bộ tách từ ngoài khi deploy.',
        },
        {
          question:
            'So sánh độ hiệu quả tham số và bộ nhớ tối ưu giữa **Full Fine-Tuning**, **LoRA** và **QLoRA** khi tinh chỉnh mô hình 7B tham số.',
          guidance:
            'Full Fine-Tuning trong AdamW FP16 cần lưu trọng số (14GB) + gradient (14GB) + 2 trạng thái moment FP32 của Adam (~56GB) + kích hoạt, vượt xa GPU 24GB. LoRA chỉ tính trạng thái optimizer cho <1% tham số hạng thấp $A, B$; QLoRA lượng tử hóa trọng số gốc đóng băng $W_0$ xuống **4-bit NormalFloat (NF4)** nên chỉ tốn ~6–10GB VRAM trên một GPU đơn.',
        },
      ],
    },
  ],
};
