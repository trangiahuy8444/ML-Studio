import { COMMON_SETUP } from './commonSetup.ts';
import type { GradModule } from './types.ts';

export const MODULE_10: GradModule = {
  id: 10,
  title: 'Học Tăng cường & Đồ án Nghiên cứu Cao học',
  englishTitle: 'Reinforcement Learning & Graduate Capstone Research',
  color: '#fb7185',
  status: 'ready',
  overview:
    'Bài toán Multi-Armed Bandits (ε-greedy, UCB1, Thompson Sampling), Markov Decision Processes (MDPs) và Phương trình tối ưu Bellman; Quy hoạch động (Policy/Value Iteration); Học không mô hình (Monte Carlo, TD(0), SARSA, Q-Learning, DQN); Policy Gradient (REINFORCE, Actor-Critic, GAE & PPO) và Đồ án Cao học hoàn chỉnh (Ablation Study, Bootstrap CI, Calibration ECE & Phân tích lỗi).',
  outcomes: [
    'Phân tích đánh đổi Khám phá – Khai thác (Exploration–Exploitation) và cài đặt thuật toán UCB1 cùng suy luận hậu nghiệm Beta Thompson Sampling.',
    'Chứng minh tính co Banach $\\gamma$-contraction của toán tử tối ưu Bellman và giải MDP hữu hạn bằng Value Iteration.',
    'Phân biệt chiến lược On-policy (SARSA) với Off-policy (Q-Learning / DQN) và cài đặt GAE cùng hàm mục tiêu cắt xén PPO.',
    'Thiết kế thực nghiệm cao học chuẩn mực với hiệu chuẩn xác suất (Expected Calibration Error), kiểm định Bootstrap ghép cặp và nghiên cứu cắt giảm (Ablation Study).',
  ],
  sources: [
    'Sutton & Barto — Reinforcement Learning: An Introduction (2nd Ed.)',
    'Stanford CS229 (Part V: Reinforcement Learning)',
  ],
  units: [
    {
      id: 'm10-u1',
      moduleId: 10,
      order: 1,
      title: 'Multi-Armed Bandits (ε-greedy, UCB, Thompson Sampling)',
      summary:
        'Bài toán ra quyết định tuần tự dưới bất định thuần túy (Stateless RL): định nghĩa hối tiếc tích lũy (Cumulative Regret), chứng minh thuật toán tham lam đơn thuần có hối tiếc tuyến tính và ba chiến lược kinh điển đạt chặn hối tiếc dưới tuyến tính ($\\varepsilon$-greedy, Upper Confidence Bound UCB1 và Thompson Sampling).',
      outcomes: [
        'Phân biệt Cumulative Regret $L_T = T\\mu^* - \\sum_{t=1}^T \\mathbb{E}[r_t]$ và chặn dưới thông tin Lai & Robbins $\\Omega(\\log T)$.',
        'Giải thích ý nghĩa chặn trên tin cậy UCB1 theo bất đẳng thức tập trung Hoeffding.',
        'Cài đặt chọn hành động UCB1 và cập nhật hậu nghiệm phân phối liên hợp Beta–Bernoulli cho Thompson Sampling.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
          section: 'Sutton & Barto (Ch. 2: Multi-armed Bandits)',
          note: 'Cân bằng Khám phá–Khai thác, cập nhật gia tăng và chặn trên UCB.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§19.5 (Multi-Armed Bandits & Thompson Sampling)',
          note: 'Góc nhìn Bayes với hậu nghiệm Beta–Bernoulli và hồi phục Regret $O(\\sqrt{AT\\log T})$.',
        },
      ],
      sections: [
        {
          heading: '1. Hối tiếc tích lũy (Regret) & Cận trên tin cậy UCB1',
          body: String.raw`Mỗi bước $t = 1,\dots,T$, tác tử chọn một tay $a_t \in \{1,\dots,K\}$ và nhận phần thưởng ngẫu nhiên có kỳ vọng chưa biết $\mu_{a_t}$. Chiến lược $\varepsilon$-greedy cố định bỏ phí tỷ lệ hằng số $\varepsilon$ thử nghiệm mãi mãi dẫn đến Regret tuyến tính $\Theta(\varepsilon T)$. Theo nguyên lý **Lạc quan trước Bất định** (*Optimism in the Face of Uncertainty*), thuật toán **UCB1** cộng thêm bán kính khoảng tin cậy rút từ Bất đẳng thức Hoeffding:`,
          math: String.raw`A_t = \arg\max_{a} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right], \qquad L_T = \sum_{t=1}^T \big(\mu^* - \mathbb{E}[\mu_{A_t}]\big) = O(\log T)`,
          intuition:
            'Khi một tay $a$ ít được thử ($N_t(a)$ nhỏ), số hạng $\\sqrt{\\ln t / N_t(a)}$ phình to để thúc đẩy khám phá; càng kéo nhiều, sai số chuẩn co lại theo $1/\\sqrt{N}$ để ưu tiên khai thác tay tối ưu.',
        },
        {
          heading: '2. Thompson Sampling (Xác suất hậu nghiệm Bayes)',
          body: String.raw`Thay vì tối đa hóa cận trên tất định, **Thompson Sampling** duy trì phân phối hậu nghiệm cho kỳ vọng thưởng $\theta_a$ (với thưởng nhị phân Bernoulli, tiên nghiệm liên hợp là $\text{Beta}(\alpha_a, \beta_a)$). Ở mỗi bước, ta bốc một mẫu $\tilde{\theta}_a \sim p(\theta_a \mid \mathcal{D}_t)$ cho từng tay và chọn $\arg\max_a \tilde{\theta}_a$, sau đó cập nhật chính xác quy tắc Bayes:`,
          math: String.raw`(\alpha_{a}^{\text{new}}, \beta_{a}^{\text{new}}) = (\alpha_a + r_t,\; \beta_a + 1 - r_t), \qquad \mathbb{E}[\theta_a \mid \mathcal{D}] = \frac{\alpha_a}{\alpha_a + \beta_a}`,
          pitfall:
            'Khởi tạo giá trị kỳ vọng ban đầu quá thấp hoặc dùng hàm tham lam thuần túy ($\\varepsilon = 0$) rất dễ mắc kẹt vĩnh viễn tại một tay dưới tối ưu chỉ vì lần thử đầu tiên ngẫu nhiên bị thưởng bằng $0$.',
        },
      ],
      simulationId: 'm10-u1',
      exercises: [
        {
          id: 'm10-u1-ucb-thompson',
          title: 'Chiến lược UCB1 & Cập nhật Hậu nghiệm Thompson Sampling',
          prompt:
            'Cài đặt hai hàm thuần NumPy:\n1. `ucb1_select(Q, N, t, c=2.0)`: với mảng ước lượng `Q` `(A,)`, số lần chọn `N` `(A,)`, tổng lượt `t >= 1` và hệ số `c`. Nếu có tay nào chưa từng được thử (`N[a] == 0`), trả về chỉ số tay đầu tiên có `N[a] == 0`; ngược lại trả về `int(np.argmax(Q + c * np.sqrt(np.log(t) / N)))`.\n2. `thompson_beta_update(alphas, betas, arm, reward)`: không chỉnh sửa tại chỗ mảng đầu vào; cộng `reward` vào `alphas[arm]` và `1.0 - reward` vào `betas[arm]`, trả về `(new_alphas, new_betas, post_means)` với `post_means = new_alphas / (new_alphas + new_betas)`.',
          hint: 'Nhớ sao chép mảng bằng `np.array(alphas, dtype=float, copy=True)` để bảo vệ dữ liệu gốc. Kiểm tra `zero_idx = np.where(N == 0)[0]` trước khi tính `np.log(t) / N` nhằm tránh chia cho 0.',
          setup:
            COMMON_SETUP +
            String.raw`
_Q_bandit = np.array([0.5, 0.72, 0.65, 0.4])
_N_bandit = np.array([25, 10, 4, 20])
_N_zero = np.array([5, 0, 3, 0])
_al_ts = np.array([2.0, 4.0, 1.0])
_be_ts = np.array([2.0, 2.0, 3.0])
`,
          starter: String.raw`def ucb1_select(Q, N, t, c=2.0):
    raise NotImplementedError

def thompson_beta_update(alphas, betas, arm, reward):
    raise NotImplementedError
`,
          solution: String.raw`def ucb1_select(Q, N, t, c=2.0):
    Q = np.asarray(Q, dtype=float)
    N = np.asarray(N, dtype=float)
    unvisited = np.where(N == 0)[0]
    if unvisited.size > 0:
        return int(unvisited[0])
    scores = Q + float(c) * np.sqrt(np.log(float(t)) / N)
    return int(np.argmax(scores))

def thompson_beta_update(alphas, betas, arm, reward):
    new_a = np.array(alphas, dtype=float, copy=True)
    new_b = np.array(betas, dtype=float, copy=True)
    new_a[arm] += float(reward)
    new_b[arm] += 1.0 - float(reward)
    post_means = new_a / (new_a + new_b)
    return new_a, new_b, post_means
`,
          test: String.raw`def _run_checks():
    def _chk_ucb():
        a_zero = ucb1_select(_Q_bandit, _N_zero, 10, c=1.5)
        a_exp = ucb1_select(_Q_bandit, _N_bandit, 59, c=1.0)
        # arm 2 has N=4, sqrt(ln(59)/4) ~ 1.009 -> 0.65 + 1.009 > arm 1: 0.72 + sqrt(ln(59)/10) ~ 1.358
        return a_zero == 1 and a_exp == 2, f'a_zero={a_zero}, a_exp={a_exp}'
    c1 = _check('UCB1 uu tien tay chua thu & chon tay co can tren cao nhat', _chk_ucb)
    def _chk_ts():
        al_before = _al_ts.copy()
        na, nb, pm = thompson_beta_update(_al_ts, _be_ts, arm=2, reward=1)
        ok = (
            np.array_equal(_al_ts, al_before)
            and na[2] == 2.0 and nb[2] == 3.0
            and abs(pm[2] - 0.4) < 1e-8
        )
        return ok, f'new_a={na.tolist()}, post_means={pm.round(3).tolist()}'
    c2 = _check('Thompson Sampling cap nhat Beta-Bernoulli dung & khong mutate input', _chk_ts)
    _report('m10-u1-ucb-thompson', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong thực tế khuyến nghị tin tức hoặc thử nghiệm A/B trực tuyến (Contextual Bandits), tại sao cập nhật mỗi tay độc lập như UCB1/Thompson chuẩn lại chưa đủ và **LinUCB** giải quyết ra sao?',
          guidance:
            'Bandit cổ điển không quan sát ngữ cảnh $x_t$ (đặc trưng người dùng/sản phẩm) nên mọi người dùng bị áp cùng một tay tốt nhất trung bình. **Contextual Bandits (LinUCB)** giả định kỳ vọng thưởng tuyến tính $\\mathbb{E}[r_t \\mid x_t, a] = \\theta_a^\\top x_t$ và cộng cận trên tin cậy Elip $\\alpha \\sqrt{x_t^\\top A_a^{-1} x_t}$ từ hồi quy Ridge trực tuyến.',
        },
        {
          question:
            'Khi phân phối thưởng thay đổi theo thời gian (**Non-stationary Bandits**), công thức trung bình mẫu $Q_n = Q_{n-1} + \\frac{1}{n}(r_n - Q_{n-1})$ gặp vấn đề gì?',
          guidance:
            'Trọng số bước giảm $1/n \\to 0$ làm bộ ước lượng trở nên bảo thủ trước môi trường thay đổi, quên rất chậm quá khứ. Giải pháp kinh điển là dùng bước học hằng số $\\alpha \\in (0, 1)$ (EMA ưu tiên quan sát gần) hoặc Sliding-Window UCB.',
        },
      ],
    },
    {
      id: 'm10-u2',
      moduleId: 10,
      order: 2,
      title: 'MDPs, Phương trình Bellman & Dynamic Programming',
      summary:
        'Chuyển sang quyết định tuần tự có trạng thái: định nghĩa Markov Decision Process $(S, A, P, R, \\gamma)$, hệ thống phương trình kỳ vọng Bellman cho hàm giá trị $V^\\pi, Q^\\pi$, phương trình tối ưu Bellman, định lý ánh xạ co $\\gamma$-Contraction và thuật toán Quy hoạch động Value Iteration / Policy Iteration.',
      outcomes: [
        'Viết phương trình kỳ vọng Bellman cho chính sách $\\pi$ cố định và phương trình tối ưu Bellman cho $V^*$.',
        'Chứng minh toán tử Bellman tối ưu $\\mathcal{T}^*$ là ánh xạ co $\\gamma$ theo chuẩn $\\|\\cdot\\|_\\infty$, đảm bảo nghiệm duy nhất bất chấp khởi tạo.',
        'Cài đặt thuật toán **Value Iteration** hội tụ tới $V^*$ và trích xuất chính sách tham lam tối ưu $\\pi^*$.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
          section: 'Sutton & Barto (Ch. 3–4: Finite MDPs & Dynamic Programming)',
          note: 'Phương trình Bellman, Policy Evaluation, Policy Iteration và Value Iteration.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part V (Markov Decision Processes, Value Iteration & Policy Iteration)',
          note: 'Phân tích MDP hữu hạn và bằng chứng hội tụ tuyến tính theo hệ số chiết khấu $\\gamma$.',
        },
      ],
      sections: [
        {
          heading: '1. Markov Decision Process & Phương trình tối ưu Bellman',
          body: String.raw`Một **MDP** được xác định bởi bộ năm $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ với hàm chuyển tiếp $P(s' \mid s, a)$, phần thưởng $R(s, a, s')$ và hệ số chiết khấu $\gamma \in [0, 1)$. Hàm giá trị tối ưu $V^*(s) = \max_\pi \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t R_{t+1} \mid S_0 = s]$ phải thỏa **Phương trình tối ưu Bellman** phi tuyến:`,
          math: String.raw`V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\Big[R(s, a, s') + \gamma V^*(s')\Big] = (\mathcal{T}^* V^*)(s)`,
          intuition:
            'Trong khi phương trình kỳ vọng Bellman cho $V^\\pi$ là hệ phương trình tuyến tính $V^\\pi = R^\\pi + \\gamma P^\\pi V^\\pi$, phương trình tối ưu Bellman chứa toán tử $\\max_a$ không tuyến tính nên giải lặp thay vì giải nghịch đảo ma trận trực tiếp.',
        },
        {
          heading: '2. Tính co Banach & Value Iteration',
          body: String.raw`Với bất kỳ hai vector giá trị $U, V \in \mathbb{R}^{|\mathcal{S}|}$, toán tử $\mathcal{T}^*$ luôn thu hẹp khoảng cách theo chuẩn vô cực ít nhất $\gamma$ lần mỗi bước:`,
          math: String.raw`\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty \implies \|V_{k+1} - V^*\|_\infty \le \gamma^{k+1}\|V_0 - V^*\|_\infty`,
          pitfall:
            'Mọi phép cộng cùng một hằng số $c$ vào hàm phần thưởng $R\' = R + c$ trong MDP bài toán hoàn thành theo tập (Episodic với thời gian kết thúc biến thiên, như mê cung) sẽ làm thay đổi chính sách tối ưu! (Khác với MDP vô hạn chiết khấu thuần túy chỉ dịch $V^*$ thêm $c / (1-\\gamma)$.)',
        },
      ],
      simulationId: 'm10-u2',
      exercises: [
        {
          id: 'm10-u2-value-iteration',
          title: 'Quy hoạch động MDP: Thuật toán Value Iteration',
          prompt:
            'Cài đặt `value_iteration(P, R, gamma=0.9, tol=1e-8, max_iter=1000)` cho MDP hữu hạn:\n- `P` `(S, A, S)` (xác suất chuyển $P[s, a, s\']$), `R` `(S, A, S)` (phần thưởng $R[s, a, s\']$).\n- Khởi tạo `V = np.zeros(S)`, lặp tới khi `np.max(np.abs(V_new - V)) < tol` (tối đa `max_iter` vòng):\n  - Tính bảng $Q(s, a) = \\sum_{s\'} P[s,a,s\'] (R[s,a,s\'] + \\gamma V[s\'])$.\n  - Cập nhật `V_new = np.max(Q, axis=1)`.\n- Trích xuất `policy = np.argmax(Q_final, axis=1)` (tính trên `V` cuối cùng) và trả về `(V, policy)`.',
          hint: 'Tính vector hóa hoàn toàn một bước cập nhật Bellman chỉ trong 1 dòng: `Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)`. Sau vòng lặp, nhớ tính lại `Q` một lần nữa theo `V` đã hội tụ trước khi lấy `np.argmax(Q, axis=1)`.',
          setup:
            COMMON_SETUP +
            String.raw`
# 3-state MDP:
# State 0 -> a=0 goes to 1 (reward 1.0); a=1 stays 0 (reward 0.0)
# State 1 -> a=0 risky to 2 with prob 0.8 (rew 5.0) or 0 prob 0.2 (rew -1.0); a=1 safe to 2 prob 1.0 (rew 2.0)
# State 2 -> terminal absorbing state (reward 0)
_P_vi = np.zeros((3, 2, 3))
_R_vi = np.zeros((3, 2, 3))
_P_vi[0, 0, 1] = 1.0; _R_vi[0, 0, 1] = 1.0
_P_vi[0, 1, 0] = 1.0; _R_vi[0, 1, 0] = 0.2
_P_vi[1, 0, 2] = 0.8; _P_vi[1, 0, 0] = 0.2; _R_vi[1, 0, 2] = 5.0; _R_vi[1, 0, 0] = -1.0
_P_vi[1, 1, 2] = 1.0; _R_vi[1, 1, 2] = 2.0
_P_vi[2, 0, 2] = 1.0
_P_vi[2, 1, 2] = 1.0
`,
          starter: String.raw`def value_iteration(P, R, gamma=0.9, tol=1e-8, max_iter=1000):
    raise NotImplementedError
`,
          solution: String.raw`def value_iteration(P, R, gamma=0.9, tol=1e-8, max_iter=1000):
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    S = P.shape[0]
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        Q = np.sum(P * (R + float(gamma) * V[None, None, :]), axis=2)
        V_new = np.max(Q, axis=1)
        diff = float(np.max(np.abs(V_new - V)))
        V = V_new
        if diff < tol:
            break
    Q_final = np.sum(P * (R + float(gamma) * V[None, None, :]), axis=2)
    policy = np.argmax(Q_final, axis=1)
    return V, policy
`,
          test: String.raw`def _run_checks():
    def _chk_bellman():
        V, pi = value_iteration(_P_vi, _R_vi, gamma=0.9, tol=1e-10)
        Q_check = np.sum(_P_vi * (_R_vi + 0.9 * V[None, None, :]), axis=2)
        bellman_res = float(np.max(np.abs(np.max(Q_check, axis=1) - V)))
        return (
            V.shape == (3,) and pi.shape == (3,) and bellman_res < 1e-8
            and pi[0] == 0 and pi[1] == 0 and abs(V[2]) < 1e-9,
            f'V={V.round(4)}, pi={pi.tolist()}, bellman_res={bellman_res:.2e}'
        )
    c1 = _check('Value Iteration thoa man diem bat dong Bellman & chon chinh sach toi uu', _chk_bellman)
    def _chk_analytic():
        # Exact analytical solution: V1 = 0.8*5 + 0.2*(-1 + 0.9*V0), V0 = 1 + 0.9*V1
        # -> V1 = 3.8 + 0.18*(1 + 0.9*V1) = 3.98 + 0.162*V1 -> 0.838*V1 = 3.98 -> V1 = 3.98/0.838
        V, _ = value_iteration(_P_vi, _R_vi, gamma=0.9, tol=1e-10)
        v1_exact = 3.98 / 0.838
        v0_exact = 1.0 + 0.9 * v1_exact
        return abs(V[1] - v1_exact) < 1e-7 and abs(V[0] - v0_exact) < 1e-7, f'V={[round(float(x), 6) for x in V]}'
    c2 = _check('Gia tri hoi tu khop nghiem giai tich he phuong trinh Bellman', _chk_analytic)
    _report('m10-u2-value-iteration', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'So sánh **Policy Iteration** và **Value Iteration** về số vòng lặp ngoài vs chi phí mỗi vòng. Khi $|\\mathcal{S}| = 10{,}000$, tại sao Modified Policy Iteration thường nhanh nhất?',
          guidance:
            'Value Iteration tốn $O(|S|^2 |A|)$ mỗi vòng và hội tụ chậm theo cấp số nhân $\\gamma^k$ (tốn hàng nghìn bước nếu $\\gamma = 0.99$). Policy Iteration thường hội tụ chỉ sau vài bước cải thiện chính sách, nhưng bước Policy Evaluation chính xác phải giải hệ tuyến tính $|S| \\times |S|$ ($O(|S|^3)$). Modified Policy Iteration chạy lặp k bước Evaluation rút gọn để dung hòa cả hai.',
        },
        {
          question:
            'Tại sao quy hoạch động cần biết trước ma trận chuyển $P(s\' \\mid s, a)$ và $R$, và làm sao chuyển sang cài đặt **Model-Free** khi không có $P$?',
          guidance:
            'Thay vì tính kỳ vọng đầy đủ $\\sum_{s\'} P(s\' \\mid s, a)[\\dots]$ (full-width backup), Model-Free RL (Monte Carlo và Temporal Difference) thay thế kỳ vọng bằng một **quỹ đạo mẫu (sample backup)** trực tiếp từ tương tác với môi trường.',
        },
      ],
    },
    {
      id: 'm10-u3',
      moduleId: 10,
      order: 3,
      title: 'Model-Free RL: Monte Carlo, TD(0), SARSA, Q-Learning & DQN',
      summary:
        'Học tăng cường không cần biết động lực học môi trường: đối chiếu Monte Carlo (không chệch, phương sai cao) với Temporal Difference TD(0) (Bootstrapping), sự khác biệt sâu sắc giữa On-policy **SARSA** và Off-policy **Q-Learning** trên bài toán Cliff Walking, và mở rộng lên **Deep Q-Network (DQN)**.',
      outcomes: [
        'Phân tích đánh đổi Bias–Variance và điều kiện cần kết thúc tập (Episode Termination) giữa Monte Carlo và TD.',
        'Viết sai số TD và quy tắc cập nhật cho **SARSA** (On-policy) vs **Q-Learning** (Off-policy) tại trạng thái kế tiếp và trạng thái kết thúc `done=True`.',
        'Giải thích hai trụ cột ổn định hóa **DQN**: Experience Replay (phá tương quan chuỗi thời gian) và Target Network (cố định mục tiêu TD).',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
          section: 'Sutton & Barto (Ch. 5–6 & Ch. 9–11: MC, TD Learning, SARSA, Q-Learning & Deadly Triad)',
          note: 'Ví dụ Cliff Walking kinh điển và bộ ba nguy hiểm Function Approximation + Bootstrapping + Off-policy.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part V (Fitted Value Iteration, Model-Free Q-Learning & Function Approximation)',
          note: 'Chuyển tiếp từ quy hoạch động dạng bảng sang xấp xỉ hàm giá trị và DQN.',
        },
      ],
      sections: [
        {
          heading: '1. Monte Carlo vs Temporal Difference TD(0)',
          body: String.raw`Monte Carlo (MC) phải đợi đi hết cả tập để thu được tổng thưởng thực $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ (không chệch nhưng phương sai rất lớn). Ngược lại, **TD Learning** cập nhật trực tiếp ngay sau 1 bước bằng kỹ thuật **Bootstrapping** (ước lượng phần tương lai bằng chính $Q$ hoặc $V$ hiện tại):`,
          math: String.raw`\delta_t^{\text{TD}} = R_{t+1} + \gamma V(S_{t+1}) - V(S_t), \qquad V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t^{\text{TD}}`,
        },
        {
          heading: '2. On-Policy SARSA vs Off-Policy Q-Learning & DQN',
          body: String.raw`- **SARSA** đánh giá và cải thiện chính chính sách hành vi $\varepsilon$-greedy đang chạy, do đó tính cả rủi ro của hành động $\varepsilon$-ngẫu nhiên ở bước kế $a_{t+1}$ (tránh đi sát mép vực trong Cliff Walking).
- **Q-Learning** cập nhật hướng về chính sách tham lam tối ưu $\max_{a\'} Q(s_{t+1}, a\')$ (Off-policy) bất kể hành vi thám hiểm thực tế đi bước nào:`,
          math: String.raw`\delta_{\text{SARSA}} = r + \gamma (1 - \text{done})\, Q(s', a') - Q(s, a), \qquad \delta_{\text{Q}} = r + \gamma (1 - \text{done}) \max_{b} Q(s', b) - Q(s, a)`,
          pitfall:
            'Kết hợp **Xấp xỉ hàm phi tuyến (Neural Network) + Bootstrapping + Off-policy** tạo thành **Bộ ba Tử thần (Deadly Triad)** dễ phân kỳ! **DQN** chế ngự bằng: (1) **Experience Replay Buffer** lấy mẫu mini-batch ngẫu nhiên phá tự tương quan liên tiếp; (2) **Target Network $\\theta^-$** giữ cố định mục tiêu trong $C$ bước và **Double DQN** tách chọn hành động khỏi định giá để chống lỗi *Overestimation Bias* của toán tử $\\max$.',
        },
      ],
      simulationId: 'm10-u3',
      exercises: [
        {
          id: 'm10-u3-td-qlearning',
          title: 'Cập nhật On-Policy SARSA vs Off-Policy Q-Learning',
          prompt:
            'Cài đặt `sarsa_qlearning_updates(Q_sarsa, Q_q, s, a, r, s_next, a_next, done, alpha=0.1, gamma=0.95)`:\n- Tạo bản sao không chỉnh sửa tại chỗ `Q_sarsa` và `Q_q`.\n- Tính sai số TD (`td_err_sarsa`, `td_err_q`) chú ý: khi `done=True`, phần chiết khấu tương lai bắt buộc bằng `0.0`!\n- Cập nhật tại ô `(s, a)` cho cả hai bảng Q với bước học `alpha` và trả về `(Q_sarsa_new, Q_q_new, td_err_sarsa, td_err_q)`.',
          hint: 'Dùng `boot_s = 0.0 if done else Q_sarsa[s_next, a_next]` và `boot_q = 0.0 if done else np.max(Q_q[s_next])`. Nhớ `Q_s_new = np.array(Q_sarsa, dtype=float, copy=True)`.',
          setup:
            COMMON_SETUP +
            String.raw`
_Qs_init = np.array([
    [1.0, 2.0, -5.0],
    [4.0, 0.5, -10.0],
    [0.0, 0.0, 0.0],
])
_Qq_init = _Qs_init.copy()
`,
          starter: String.raw`def sarsa_qlearning_updates(Q_sarsa, Q_q, s, a, r, s_next, a_next, done, alpha=0.1, gamma=0.95):
    raise NotImplementedError
`,
          solution: String.raw`def sarsa_qlearning_updates(Q_sarsa, Q_q, s, a, r, s_next, a_next, done, alpha=0.1, gamma=0.95):
    Qs_new = np.array(Q_sarsa, dtype=float, copy=True)
    Qq_new = np.array(Q_q, dtype=float, copy=True)
    boot_s = 0.0 if done else float(Q_sarsa[s_next, a_next])
    boot_q = 0.0 if done else float(np.max(Q_q[s_next]))
    td_err_sarsa = float(r) + float(gamma) * boot_s - float(Q_sarsa[s, a])
    td_err_q = float(r) + float(gamma) * boot_q - float(Q_q[s, a])
    Qs_new[s, a] += float(alpha) * td_err_sarsa
    Qq_new[s, a] += float(alpha) * td_err_q
    return Qs_new, Qq_new, td_err_sarsa, td_err_q
`,
          test: String.raw`def _run_checks():
    def _chk_cliff():
        # Transition from (0, 1) to state 1 where explorer picks risky a_next=2 (Q=-10) while max Q=4.0
        Qs, Qq, ts, tq = sarsa_qlearning_updates(_Qs_init, _Qq_init, 0, 1, -1.0, 1, 2, False, alpha=0.2, gamma=0.9)
        # Expected SARSA target: -1 + 0.9*(-10) - 2 = -12.0; Q target: -1 + 0.9*4.0 - 2 = +0.6
        ok = (
            abs(ts + 12.0) < 1e-8 and abs(tq - 0.6) < 1e-8
            and abs(Qs[0, 1] - (2.0 + 0.2 * (-12.0))) < 1e-8
            and abs(Qq[0, 1] - (2.0 + 0.2 * 0.6)) < 1e-8
            and abs(_Qs_init[0, 1] - 2.0) < 1e-9
        )
        return ok, f'td_sarsa={ts:.3f}, td_q={tq:.3f}'
    c1 = _check('SARSA (on-policy) phan anh rui ro tham hiem vs Q-Learning (off-policy) theo max', _chk_cliff)
    def _chk_done():
        Qs, Qq, ts, tq = sarsa_qlearning_updates(_Qs_init, _Qq_init, 0, 1, 10.0, 1, 0, True, alpha=0.5, gamma=0.95)
        ok = abs(ts - 8.0) < 1e-8 and abs(tq - 8.0) < 1e-8 and abs(Qs[0, 1] - 6.0) < 1e-8
        return ok, f'done_td={ts:.3f}, Q_new={Qs[0, 1]:.3f}'
    c2 = _check('Trang thai ket thuc done=True cat hoan toan bootstrap tuong lai', _chk_done)
    _report('m10-u3-td-qlearning', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong **Double Q-Learning / Double DQN**, vì sao chỉ cần dùng 2 bộ tham số $(Q_{\\text{online}}, Q_{\\text{target}})$ tách hành động chọn $\\arg\\max_a Q_{\\text{online}}(s\', a)$ khỏi bước lấy giá trị $Q_{\\text{target}}(s\', a^*)$ là giảm được Overestimation Bias?',
          guidance:
            'Do bất đẳng thức Jensen trên biến ngẫu nhiên nhiễu ước lượng: $\\mathbb{E}[\\max_a Q(s\', a)] \\ge \\max_a \\mathbb{E}[Q(s\', a)]$, nên phép $\\max$ trên cùng một ước lượng nhiễu luôn thiên lệch dương tích lũy theo chu trình bootstrap. Khi tách hai ước lượng có sai số độc lập (online chọn `argmax`, target đọc giá trị), sai số dương ở bộ chọn không tương quan dương với giá trị đọc ra, triệt tiêu thiên lệch $\\max$.',
        },
        {
          question:
            'Nhược điểm cốt yếu của các phương pháp dựa trên giá trị (**Q-Learning / DQN**) khi áp dụng cho bài toán điều khiển robot có không gian hành động **liên tục nhiều chiều** (như khớp mô-men xoắn 12 bậc tự do) là gì?',
          guidance:
            'Không thể duyệt $\\arg\\max_{a \\in \\mathbb{R}^{12}} Q(s, a)$ ở mọi bước thời gian hoặc dựng bảng $Q$ rời rạc do Lời nguyền Chiều cao! Do đó bài toán liên tục đòi hỏi trực tiếp tham số hóa chính sách **Policy Gradient / Actor-Critic**.',
        },
      ],
    },
    {
      id: 'm10-u4',
      moduleId: 10,
      order: 4,
      title: 'Policy Gradient: REINFORCE, Actor-Critic & PPO',
      summary:
        'Tối ưu trực tiếp chính sách tham số $\\pi_\\theta(a \\mid s)$ qua Định lý Policy Gradient và Log-Derivative Trick: thuật toán REINFORCE có baseline, Actor-Critic giảm phương sai, Ước lượng Lợi thế Tổng quát (Generalized Advantage Estimation — GAE) và hàm mục tiêu Cắt xén ổn định của **PPO** (Schulman et al., 2017).',
      outcomes: [
        'Dẫn xuất Policy Gradient $\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\pi_\\theta}[\\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)\\, \\hat{A}_t]$ và chứng minh thêm Baseline $b(s_t)$ không gây thiên lệch nhưng giảm mạnh phương sai.',
        'Cài đặt **GAE($\\lambda$)** dung hòa cân bằng Bias–Variance xuyên suốt giữa TD(0) và Monte Carlo.',
        'Cài đặt hàm mục tiêu cắt xén **PPO-Clip** và giải thích vai trò bất đối xứng của phép $\\min(r\\hat{A}, \\text{clip}(r)\\hat{A})$ khi lợi thế âm vs dương.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
          section: 'Sutton & Barto (Ch. 13: Policy Gradient Methods & Actor-Critic)',
          note: 'Định lý Policy Gradient, REINFORCE with Baseline và Actor-Critic một bước.',
        },
        {
          source: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
          section: 'Part V — Policy Gradient & Proximal Policy Optimization (PPO)',
          note: 'Trust region, ratio clipping và ứng dụng vào căn chỉnh LLM (RLHF).',
        },
      ],
      sections: [
        {
          heading: '1. Định lý Policy Gradient & Baseline Advantage',
          body: String.raw`Dùng thủ thuật đạo hàm log ($\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$), ta tính được gradient của kỳ vọng tổng thưởng mà không cần đạo hàm xuyên qua hàm chuyển động lực học môi trường $P$:`,
          math: String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\right], \qquad \hat{A}_t = Q(s_t, a_t) - V(s_t)`,
          intuition:
            'Bất kỳ hàm Baseline $b(s_t)$ nào chỉ phụ thuộc vào trạng thái đều có $\\mathbb{E}_{a_t}[\\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) b(s_t)] = b(s_t) \\nabla_\\theta \\sum_a \\pi_\\theta = 0$, nên chọn $b(s_t) = V_\\phi(s_t)$ vừa giữ tính không chệch vừa biến trọng số nhân thành **Advantage** $\\hat{A}_t$ (đo độ tốt *tương đối* của hành động so với trung bình).',
        },
        {
          heading: '2. GAE (Generalized Advantage Estimation) & Hàm mục tiêu cắt xén PPO',
          body: String.raw`**GAE** trộn luỹ thừa các sai số TD $\delta_t = r_t + \gamma V(s_{t+1})(1 - \text{done}_t) - V(s_t)$ bằng hệ số $\lambda \in [0, 1]$ (` + '`lam=0`' + String.raw` cho TD(0) bias cao/variance thấp; ` + '`lam=1`' + String.raw` cho Monte Carlo). Sau đó, **PPO** ngăn bước cập nhật chính sách nhảy quá xa khỏi phân phối lấy mẫu cũ $\pi_{\theta_{\text{old}}}$ dựa trên tỷ số $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$:`,
          math: String.raw`\hat{A}_t^{\text{GAE}} = \delta_t + (\gamma \lambda)(1 - \text{done}_t)\hat{A}_{t+1}^{\text{GAE}}, \qquad L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\Big(r_t(\theta)\hat{A}_t,\; \text{clip}\big(r_t(\theta), 1-\varepsilon, 1+\varepsilon\big)\hat{A}_t\Big)\right]`,
          pitfall:
            'Nếu cài đặt `clip` rồi lấy trực tiếp thay vì bao bên ngoài bằng $\\min(r\\hat{A}, \\text{clip}(r)\\hat{A})$, hàm mục tiêu sẽ không phạt trường hợp chính sách bước lỗi sang vùng có lợi về mặt cắt nhưng làm tệ đi hàm thật (chẳng hạn khi $\\hat{A} < 0$ và $r > 1 + \\varepsilon$, ta **vẫn cần giữ** phần dốc đi xuống của $r\\hat{A}$ để kéo tỷ số $r$ giảm trở về vùng an toàn!).',
        },
      ],
      simulationId: 'm10-u4',
      exercises: [
        {
          id: 'm10-u4-gae-ppo',
          title: 'Generalized Advantage Estimation (GAE) & PPO-Clip Objective',
          prompt:
            'Cài đặt hai hàm thuần NumPy:\n1. `compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95)`: duyệt ngược từ `t = T - 1` về `0` tính `advantages` `(T,)` và `returns = advantages + values` `(T,)`. Lưu ý `non_term = 1.0 - float(dones[t])` cắt cả bootstrap `v_next` lẫn `gae` truyền từ `t+1`!\n2. `ppo_clipped_objective(logp_new, logp_old, advantages, clip_eps=0.2)`: tính tỷ số `ratio = np.exp(logp_new - logp_old)`, trả về giá trị trung bình vô hướng `float(np.mean(np.minimum(ratio * advantages, np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages)))`.',
          hint: 'Trong vòng lặp ngược GAE: `v_next = next_value if t == T - 1 else values[t + 1]`, `delta = rewards[t] + gamma * v_next * non_term - values[t]`, `gae = delta + gamma * lam * non_term * gae`, rồi `advantages[t] = gae`.',
          setup:
            COMMON_SETUP +
            String.raw`
_rew_gae = np.array([1.0, 2.0, 0.5, 4.0])
_val_gae = np.array([0.8, 1.5, 1.2, 2.0])
_done_gae = np.array([0.0, 1.0, 0.0, 0.0])  # episode ends at t=1!
_lp_old = np.array([-1.0, -1.0, -1.0, -1.0])
_lp_new = np.array([-1.0 + np.log(1.5), -1.0 + np.log(0.6), -1.0 + np.log(0.5), -1.0 + np.log(1.4)])
_adv_ppo = np.array([2.0, 2.0, -1.0, -1.0])
`,
          starter: String.raw`def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    raise NotImplementedError

def ppo_clipped_objective(logp_new, logp_old, advantages, clip_eps=0.2):
    raise NotImplementedError
`,
          solution: String.raw`def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=float)
    T = rewards.shape[0]
    advantages = np.zeros(T, dtype=float)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        non_term = 1.0 - float(dones[t])
        v_next = float(next_value) if t == T - 1 else float(values[t + 1])
        delta = rewards[t] + float(gamma) * v_next * non_term - values[t]
        gae = delta + float(gamma) * float(lam) * non_term * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

def ppo_clipped_objective(logp_new, logp_old, advantages, clip_eps=0.2):
    logp_new = np.asarray(logp_new, dtype=float)
    logp_old = np.asarray(logp_old, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    ratio = np.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantages
    return float(np.mean(np.minimum(surr1, surr2)))
`,
          test: String.raw`def _run_checks():
    def _chk_gae():
        adv0, ret0 = compute_gae(_rew_gae, _val_gae, 3.0, _done_gae, gamma=0.9, lam=0.0)
        # At t=1 (done=1), delta_1 = 2.0 + 0 - 1.5 = 0.5; at t=0, delta_0 = 1.0 + 0.9*1.5 - 0.8 = 1.55
        adv_lam, _ = compute_gae(_rew_gae, _val_gae, 3.0, _done_gae, gamma=0.9, lam=0.8)
        # Since done[1]=1, adv at t=1 is strictly delta_1=0.5 regardless of lam; adv at t=0 = 1.55 + 0.9*0.8*0.5 = 1.91
        ok = (
            abs(adv0[1] - 0.5) < 1e-8 and abs(adv_lam[1] - 0.5) < 1e-8
            and abs(adv_lam[0] - 1.91) < 1e-7 and np.allclose(ret0, adv0 + _val_gae)
        )
        return ok, f'adv_lam0={adv0.round(3)}, adv_lam0.8={adv_lam.round(3)}'
    c1 = _check('GAE tinh dung delta TD & ngat horizon khi done=1', _chk_gae)
    def _chk_ppo():
        obj = ppo_clipped_objective(_lp_new, _lp_old, _adv_ppo, clip_eps=0.2)
        # terms: min(1.5*2, 1.2*2)=2.4; min(0.6*2, 0.8*2)=1.2; min(0.5*(-1), 0.8*(-1))=-0.8; min(1.4*(-1), 1.2*(-1))=-1.4 -> sum=1.4 -> mean=0.35
        return abs(obj - 0.35) < 1e-7, f'ppo_obj={obj:.6f} (expected 0.35)'
    c2 = _check('PPO-Clip xu ly bat doi xung dung min(surr1, surr2) cho ca A > 0 va A < 0', _chk_ppo)
    _report('m10-u4-gae-ppo', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong quy trình **RLHF / Alignment** cho Mô hình Ngôn ngữ Lớn (InstructGPT / ChatGPT), tại sao PPO thường cộng thêm hệ số phạt $\\beta\\,\\text{KL}(\\pi_\\theta \\parallel \\pi_{\\text{ref}})$ vào phần thưởng từng token ngoài clipping tỷ số $r_t(\\theta)$?',
          guidance:
            'Reward Model chỉ là hàm xấp xỉ được huấn luyện trên dữ liệu so sánh cặp hữu hạn, nên tối ưu hóa quá mức có thể khai thác kẽ hở của Reward Model (**Reward Hacking** — sinh câu dài vô nghĩa hoặc lặp từ đạt điểm cao ảo). Số hạng phạt KL giữ chính sách mới không trôi dạt quá xa khỏi mô hình SFT gốc $\\pi_{\\text{ref}}$.',
        },
        {
          question:
            'So sánh **PPO** với **GRPO (Group Relative Policy Optimization)** về chi phí mô hình phê bình (Critic / Value Network)?',
          guidance:
            'PPO cần một mạng Critic riêng biệt kích thước xấp xỉ mạng Policy để ước lượng $V(s_t)$ và tính GAE, tốn gần gấp đôi VRAM trên GPU. GRPO lấy mẫu một nhóm $G$ câu trả lời cho cùng một prompt và chuẩn hóa phần thưởng trực tiếp theo trung bình/độ lệch chuẩn của chính nhóm đó làm lợi thế ($A_i = (r_i - \\bar{r})/\\text{std}(r)$), loại bỏ hoàn toàn mạng Critic.',
        },
      ],
    },
    {
      id: 'm10-u5',
      moduleId: 10,
      order: 5,
      title: 'Đồ án Cao học: Thực nghiệm tái lập, Ablation Study & Phân tích lỗi',
      summary:
        'Quy trình chuẩn nghiên cứu cao học hoàn chỉnh cho bài báo và luận văn Machine Learning: kiểm soát tái lập đa hạt giống (Seed Variance), thí nghiệm cắt giảm thành phần (**Ablation Study**), kiểm định giả thuyết phi tham số **Paired Bootstrap Significance**, đánh giá hiệu chuẩn xác suất (**Expected Calibration Error — ECE**) và phân tích lỗi định lượng.',
      outcomes: [
        'Thiết kế bảng **Ablation Study** chứng minh từng đóng góp của một thành phần đề xuất thay vì chỉ báo cáo điểm tổng.',
        'Đánh giá độ tin cậy xác suất dự đoán bằng **Expected Calibration Error (ECE)** và áp dụng hậu xử lý Temperature Scaling trên tập Validation.',
        'Cài đặt hàm tính ECE và kiểm định ý nghĩa thống kê **Paired Bootstrap Resampling** cho khoảng tin cậy 95% và p-value giữa hai mô hình.',
      ],
      references: [
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
          section: '§5.7 & §14.7 (Calibration, Confidence Intervals & Reproducible Evaluation)',
          note: 'Đánh giá xác suất hiệu chuẩn (Reliability Diagram / ECE) và so sánh mô hình có ý nghĩa thống kê.',
        },
        {
          source: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
          section: '§5.2 (The Bootstrap for Uncertainty Quantification)',
          note: 'Ước lượng sai số chuẩn và khoảng tin cậy không cần giả định phân phối chuẩn.',
        },
      ],
      sections: [
        {
          heading: '1. Hiệu chuẩn Xác suất (Calibration) & Expected Calibration Error',
          body: String.raw`Một mô hình có độ chính xác cao vẫn có thể tự tin sai lầm nghiêm trọng (Overconfident — dự đoán xác suất $0.99$ nhưng thực tế chỉ đúng $75\%$ trong nhóm đó), gây thảm họa trong chẩn đoán y khoa hay ra quyết định tài chính. Chia miền dự đoán $[0, 1]$ thành $M$ khoảng (bins) đều $B_m$: **ECE** đo tổng trung bình có trọng số độ lệch giữa tần suất thực nghiệm ` + '`acc(B_m)`' + String.raw` và độ tin cậy dự đoán trung bình ` + '`conf(B_m)`' + String.raw`:`,
          math: String.raw`\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \,\Big|\text{acc}(B_m) - \text{conf}(B_m)\Big|`,
          intuition:
            'Hậu xử lý đơn giản mà rất hiệu quả cho bài toán phân loại Neural Net là **Temperature Scaling** ($\\text{softmax}(z / T)$ với tham số duy nhất $T > 0$ tối ưu NLL trên tập Validation), giúp giảm mạnh ECE mà không làm thay đổi thứ hạng $\\arg\\max$ (Accuracy giữ nguyên 100%).',
        },
        {
          heading: '2. Kiểm định Bootstrap ghép cặp & Khung Đồ án Tái lập',
          body:
            String.raw`Để kết luận mô hình đề xuất $B$ thực sự vượt Baseline $A$ trên cùng tập Test $N$ mẫu mà không phải may mắn ngẫu nhiên, ta dùng **Paired Bootstrap** lấy mẫu có hoàn lại $B_{\text{boot}}$ lần từ chính hiệu điểm từng mẫu $d_i = s_i^{(B)} - s_i^{(A)}$ để rút ra khoảng tin cậy 95% và p-value thực nghiệm. Mọi đồ án cao học bắt buộc bảo đảm 4 tiêu chí:
1. **Reproducibility**: cố định seed cho mọi thư viện, chia tập không rò rỉ (` +
            '`stratified`' +
            String.raw` / ` +
            '`temporal`' +
            String.raw`), tách sạch Hyperparameter Search trên Val khỏi Test.
2. **Ablation Study**: bật/tắt từng thành phần một (chứ không thay 3 thứ cùng lúc) trên cùng lịch trình huấn luyện.
3. **Multi-seed Reporting**: báo cáo ` +
            '`mean ± std`' +
            String.raw` (qua $\ge 5$ random seeds) kèm kiểm định ý nghĩa thống kê.
4. **Quantitative Error Analysis**: phân nhóm các ca sai điển hình (Edge cases, lớp thiểu số, phân phối dịch chuyển).`,
          math: String.raw`p\text{-value}_{\text{boot}} = \frac{1}{B_{\text{boot}}} \sum_{b=1}^{B_{\text{boot}}} \mathbb{I}\!\left[\bar{d}^{*(b)} \le 0\right], \qquad \text{CI}_{95\%} = \big[Q_{2.5\%}(\bar{d}^*),\; Q_{97.5\%}(\bar{d}^*)\big]`,
        },
      ],
      simulationId: 'm10-u5',
      exercises: [
        {
          id: 'm10-u5-eval-ece-ablation',
          title: 'Expected Calibration Error (ECE) & Paired Bootstrap Test',
          prompt:
            'Cài đặt hai công cụ đánh giá chuẩn nghiên cứu:\n1. `expected_calibration_error(probs, labels, n_bins=10)`: với dự đoán nhị phân `probs` `(N,)` trong `[0, 1]` và nhãn thật `labels` `(N,)` (`0` hoặc `1`), gán chỉ số bin bằng `bin_idx = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)` và tính vô hướng $\\text{ECE} = \\sum_{b} \\frac{|B_b|}{N} |\\text{mean}(\\text{labels}[B_b]) - \\text{mean}(\\text{probs}[B_b])|$.\n2. `paired_bootstrap_pvalue(scores_a, scores_b, n_boot=2000, seed=42)`: xét hiệu `diffs = scores_b - scores_a`, dùng `rng = np.random.default_rng(seed)` bốc `n_boot` mẫu bootstrap kích thước `N` (`rng.integers(0, N, size=(n_boot, N))`), tính `boot_means` rồi trả về `(mean_diff, ci_low, ci_high, p_value)` với percentile `[2.5, 97.5]` và `p_value = float(np.mean(boot_means <= 0.0))`.',
          hint: 'Dùng `np.percentile(boot_means, 2.5)` và `np.percentile(boot_means, 97.5)` cho khoảng tin cậy 95%. Chú ý bỏ qua các bin rỗng (`if np.any(mask)`) trong khi tính ECE.',
          setup:
            COMMON_SETUP +
            String.raw`
_probs_perfect = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                           0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
_labels_perfect = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
_probs_overconf = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
_labels_overconf = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float) # conf=0.95, acc=0.60 -> ECE=0.35
_scores_A = np.array([0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.])
_scores_B = np.array([1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1.])
`,
          starter: String.raw`def expected_calibration_error(probs, labels, n_bins=10):
    raise NotImplementedError

def paired_bootstrap_pvalue(scores_a, scores_b, n_boot=2000, seed=42):
    raise NotImplementedError
`,
          solution: String.raw`def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    N = probs.shape[0]
    bin_idx = np.clip((probs * int(n_bins)).astype(int), 0, int(n_bins) - 1)
    ece = 0.0
    for b in range(int(n_bins)):
        mask = bin_idx == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / float(N)) * abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
    return float(ece)

def paired_bootstrap_pvalue(scores_a, scores_b, n_boot=2000, seed=42):
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diffs = scores_b - scores_a
    N = diffs.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(int(n_boot), N))
    boot_means = np.mean(diffs[idx], axis=1)
    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    p_value = float(np.mean(boot_means <= 0.0))
    return mean_diff, ci_low, ci_high, p_value
`,
          test: String.raw`def _run_checks():
    def _chk_ece():
        e0 = expected_calibration_error(_probs_perfect, _labels_perfect, n_bins=10)
        e1 = expected_calibration_error(_probs_overconf, _labels_overconf, n_bins=10)
        return abs(e0) < 1e-8 and abs(e1 - 0.35) < 1e-7, f'ece_perfect={e0:.6f}, ece_overconf={e1:.4f}'
    c1 = _check('ECE = 0.0 khi hieu chuan hoan hao & = 0.35 khi overconfident', _chk_ece)
    def _chk_boot():
        md, lo, hi, pv = paired_bootstrap_pvalue(_scores_A, _scores_B, n_boot=2000, seed=42)
        _, _, _, pv_same = paired_bootstrap_pvalue(_scores_A, _scores_A, n_boot=500, seed=42)
        ok = abs(md - 0.45) < 1e-8 and lo > 0.15 and hi < 0.75 and pv < 0.01 and pv_same == 1.0
        return ok, f'diff={md:.2f}, CI95=[{lo:.3f}, {hi:.3f}], p={pv:.4f}'
    c2 = _check('Paired Bootstrap tra dung khoang tin cay 95% va p-value < 0.01', _chk_boot)
    _report('m10-u5-eval-ece-ablation', c1, c2)
_run_checks()
`,
        },
      ],
      seminar: [
        {
          question:
            'Trong đồ án tốt nghiệp hoặc bài báo ML, khi phát hiện mô hình mới tăng Accuracy từ 88.2% lên 88.5% trên tập Test 1,000 mẫu, quy trình kiểm chứng nào giúp phân biệt cải tiến thực sự với nhiễu ngẫu nhiên của hạt giống khởi tạo?',
          guidance:
            'Bắt buộc thực hiện đồng thời: (1) Chạy ít nhất 5 random seeds độc lập cho cả Baseline lẫn Mô hình đề xuất với cùng ngân sách HPO (Hyperparameter Tuning budget) để xem khoảng `mean ± std` có chồng lấn không; (2) Chạy **Paired Bootstrap / McNemar Test** ngay trên vector dự đoán ở cùng tập Test; (3) Kiểm tra **Ablation Study** từng thành phần để chứng minh nguyên nhân cải thiện khớp với giả thuyết lý thuyết.',
        },
        {
          question:
            'Tại sao kiểm chứng bằng **ECE** và **Negative Log-Likelihood (NLL)** song hành với Accuracy/F1 lại bắt buộc đối với hệ thống ML triển khai thực tế (y khoa, xe tự hành, chấm điểm tín dụng)?',
          guidance:
            'Accuracy chỉ kiểm tra tính đúng của `argmax`, bỏ qua hoàn toàn độ tin cậy định lượng. Một mạng sâu không hiệu chuẩn có thể đạt 95% Accuracy nhưng gán xác suất $0.99$ cho các ca sai biên ngoài phân phối (Out-of-Distribution). ECE và NLL/Brier Score đo khả năng đưa ra quyết định ngưỡng rủi ro (Selective Prediction / Human-in-the-loop) an toàn.',
        },
      ],
    },
  ],
};
