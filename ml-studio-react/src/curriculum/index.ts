import { MODULE_01 } from './module01.ts';
import type { GradModule, GradResourceItem, GradUnit } from './types.ts';

export type {
  GradExercise,
  GradModule,
  GradReference,
  GradResourceItem,
  GradSection,
  GradSeminar,
  GradSimulationId,
  GradUnit,
} from './types';

export const GRAD_MODULES: GradModule[] = [
  MODULE_01,
  {
    id: 2,
    title: 'Mô hình Tuyến tính & GLMs',
    englishTitle: 'Linear Models & Generalized Linear Models',
    color: '#84e0a8',
    status: 'planned',
    overview:
      'Phân tích hình học và xác suất của Hồi quy tuyến tính OLS (phép chiếu trực giao & Normal Equations), điều chuẩn Ridge/Lasso từ góc nhìn MAP, phân loại tuyến tính Logistic/Softmax Regression và khung tổng quát Exponential Family GLMs.',
    outcomes: [
      'Chứng minh nghiệm OLS qua phép chiếu trực giao và xác định điều kiện suy biến đa cộng tuyến.',
      'Giải thích sự khác biệt hình học giữa phạt L1 (tạo nghiệm thưa) và phạt L2.',
      'Dẫn xuất mô hình Logistic, Softmax và hồi quy Poisson từ họ phân phối mũ (Exponential Family).',
    ],
    sources: [
      'UC Berkeley CS189 (Lecture 06–11)',
      'Stanford CS229 (Part I: Linear Models & GLMs)',
      'ISLR (Ch. 3–4, Ch. 6)',
      'Murphy PML (Ch. 10–11)',
    ],
    plannedTopics: [
      'Unit 2.1: OLS dưới góc nhìn Hình học (Phép chiếu) & Xác suất (Gaussian MLE)',
      'Unit 2.2: Regularization Ridge ($L_2$), Lasso ($L_1$) & ElasticNet từ MAP',
      'Unit 2.3: Logistic Regression, Softmax & Cross-Entropy',
      'Unit 2.4: Generalized Linear Models (GLMs) & Exponential Family',
    ],
    units: [],
  },
  {
    id: 3,
    title: 'Tối ưu hóa cho Học máy',
    englishTitle: 'Optimization for Machine Learning',
    color: '#ffd173',
    status: 'planned',
    overview:
      'Lý thuyết tối ưu lồi (tập lồi, hàm lồi, điều kiện KKT, Condition Number của ma trận Hessian), phân tích tốc độ hội tụ Gradient Descent, các biến thể Stochastic/Adaptive (SGD, Momentum, Nesterov, RMSProp, Adam) và phương pháp bậc 2 (Newton–Raphson, L-BFGS).',
    outcomes: [
      'Chứng minh chặn hội tụ của Gradient Descent trên hàm L-smooth và $\\mu$-strongly convex.',
      'Chẩn đoán ảnh hưởng của Condition Number $\\kappa(H)$ lên quỹ đạo cập nhật và thiết lập Momentum/Adam.',
      'Cài đặt SGD, Adam và Newton–Raphson cho bài toán tối ưu lồi không có nghiệm đóng.',
    ],
    sources: [
      'UC Berkeley CS189 (Lecture 12–13)',
      'Murphy PML (Ch. 8: Optimization)',
      'Vũ Hữu Tiệp (Ch. 7–8)',
    ],
    plannedTopics: [
      'Unit 3.1: Tập lồi, Hàm lồi, Điều kiện KKT & Condition Number',
      'Unit 3.2: Gradient Descent & Phân tích tốc độ hội tụ',
      'Unit 3.3: SGD, Mini-batch, Momentum, Nesterov, RMSProp & Adam',
      'Unit 3.4: Phương pháp bậc 2: Newton–Raphson & Quasi-Newton (L-BFGS)',
    ],
    units: [],
  },
  {
    id: 4,
    title: 'Mô hình Sinh, Kernel & Support Vector Machines',
    englishTitle: 'Generative Classifiers, Kernel Methods & SVM',
    color: '#f28db9',
    status: 'planned',
    overview:
      'So sánh hệ hình Generative vs Discriminative; phân tích Gaussian Discriminant Analysis (LDA/QDA) và Naive Bayes; lý thuyết lề cực đại Hard/Soft-Margin SVM (Primal & Lagrangian Dual); Kernel Trick, Định lý Mercer và không gian RKHS.',
    outcomes: [
      'Dẫn xuất biên quyết định tuyến tính (LDA) và bậc hai (QDA) từ giả định Gaussian.',
      'Chuyển đổi bài toán SVM từ dạng gốc (Primal) sang dạng đối ngẫu (Dual) bằng nhân tử Lagrange.',
      'Thiết kế hàm nhân RBF/Polynomial hợp lệ theo điều kiện Mercer và phân tích vai trò Support Vectors.',
    ],
    sources: [
      'Stanford CS229 (Part II–III: Generative Algorithms & Kernel Methods/SVM)',
      'Murphy PML (Ch. 9, Ch. 17)',
      'Vũ Hữu Tiệp (Ch. 19–22)',
    ],
    plannedTopics: [
      'Unit 4.1: Generative vs. Discriminative; GDA (LDA/QDA) & Naive Bayes',
      'Unit 4.2: Maximum Margin Classifier & Hard/Soft-Margin SVM (Primal & Dual)',
      'Unit 4.3: Kernel Trick, Định lý Mercer & RKHS (RBF, Polynomial Kernels)',
    ],
    units: [],
  },
  {
    id: 5,
    title: 'Phương pháp Phi tham số & Học Tổ hợp',
    englishTitle: 'Nonparametric Models & Ensemble Learning',
    color: '#a8a7ff',
    status: 'planned',
    overview:
      'Học dựa trên láng giềng KNN và lời nguyền chiều cao (Curse of Dimensionality); Cây quyết định CART (Gini, Entropy, tỉa nhánh); Bagging & Random Forests; và các thuật toán Boosting tuần tự (AdaBoost, Gradient Boosting / GBDT).',
    outcomes: [
      'Chứng minh lý do khoảng cách Euclid mất khả năng phân biệt trong không gian rất nhiều chiều.',
      'Cài đặt hàm tách nút cây quyết định theo Information Gain và Gini Impurity.',
      'Phân tích vì sao Bagging giảm Variance còn Boosting (GBDT) giảm chủ yếu Bias.',
    ],
    sources: [
      'ISLR (Ch. 8: Tree-Based Methods)',
      'Murphy PML (Ch. 16, Ch. 18)',
    ],
    plannedTopics: [
      'Unit 5.1: KNN, Metric Learning & Curse of Dimensionality',
      'Unit 5.2: Decision Trees (CART, Gini/Entropy, Pruning)',
      'Unit 5.3: Bagging & Random Forests (Giảm phương sai)',
      'Unit 5.4: Boosting: AdaBoost & Gradient Boosting (GBDT / XGBoost)',
    ],
    units: [],
  },
  {
    id: 6,
    title: 'Học Không giám sát & Mô hình Biến tiềm ẩn',
    englishTitle: 'Unsupervised Learning & Latent Variable Models',
    color: '#ff9b6a',
    status: 'planned',
    overview:
      'Thuật toán phân cụm K-means, Hierarchical và Spectral Clustering (Graph Cut); ước lượng mật độ KDE và Gaussian Mixture Models (GMM); chứng minh hội tụ thuật toán EM qua Bất đẳng thức Jensen; giảm chiều PCA qua SVD, Kernel PCA và trực quan hóa phi tuyến t-SNE/UMAP.',
    outcomes: [
      'Chứng minh bước E và bước M trong thuật toán EM đảm bảo Log-Likelihood không giảm nhờ Bất đẳng thức Jensen.',
      'Liên hệ PCA dưới hai góc nhìn tương đương: cực đại hóa phương sai chiếu và tối thiểu hóa sai số tái tạo via SVD.',
      'Áp dụng GMM và Spectral Clustering trên các cấu trúc cụm phi lồi.',
    ],
    sources: [
      'UC Berkeley CS189 (Lecture 04–05: K-means, Density Estimation & GMM)',
      'Stanford CS229 (Part IV: Unsupervised Learning, PCA, EM)',
      'Murphy PML (Ch. 20–21)',
    ],
    plannedTopics: [
      'Unit 6.1: Phân cụm: K-means, Hierarchical & Spectral Clustering',
      'Unit 6.2: Ước lượng mật độ (KDE) & Gaussian Mixture Models (GMM)',
      'Unit 6.3: Thuật toán Expectation-Maximization (EM) & Bất đẳng thức Jensen',
      'Unit 6.4: Giảm chiều: PCA (SVD), Kernel PCA, ICA & t-SNE/UMAP',
    ],
    units: [],
  },
  {
    id: 7,
    title: 'Deep Learning & Thị giác Máy tính',
    englishTitle: 'Deep Neural Networks & Computer Vision',
    color: '#38bdf8',
    status: 'planned',
    overview:
      'Biểu diễn ảnh số và các kỹ thuật xử lý ảnh cổ điển (Filtering, Histogram Equalization, Edge Detection, OpenCV); MLP, Định lý Xấp xỉ Phổ quát và Reverse-mode Autodiff (Backpropagation); kỹ thuật huấn luyện mạng rất sâu (Xavier/He Init, BatchNorm/LayerNorm, Residual Connections) và kiến trúc CNN đến Vision Transformer.',
    outcomes: [
      'Cài đặt cơ chế Reverse-mode Automatic Differentiation (tính toán đồ thị và lan truyền ngược) bằng NumPy.',
      'Hiểu bản chất bất biến dịch chuyển, trường tiếp nhận (Receptive Field) của tầng Conv2D và tác dụng của Skip Connection.',
      'Kết hợp tiền xử lý thị giác cổ điển với mạng CNN hiện đại cho bài toán phân loại ảnh.',
    ],
    sources: [
      'Thư mục 05_Thi_giac_May_tinh_Computer_Vision (C1–C3, Bai1–3)',
      'Stanford CS229 (Deep Learning Notes)',
      'Murphy PML (Ch. 13–14)',
    ],
    plannedTopics: [
      'Unit 7.1: Biểu diễn ảnh & Xử lý ảnh cổ điển (Filtering, Histogram, OpenCV)',
      'Unit 7.2: MLP, Universal Approximation & Reverse-mode Autodiff (Backprop)',
      'Unit 7.3: Huấn luyện mạng sâu: Xavier/He Init, BatchNorm, Dropout & Residual',
      'Unit 7.4: Kiến trúc CNN (Conv, Pooling, Receptive Field, ResNet → ViT)',
    ],
    units: [],
  },
  {
    id: 8,
    title: 'Xử lý Ngôn ngữ Tự nhiên, Transformer & LLMs',
    englishTitle: 'NLP, Transformers & Large Language Models',
    color: '#c084fc',
    status: 'planned',
    overview:
      'Từ biểu diễn từ tĩnh Word2Vec (Skip-gram, CBOW, Negative Sampling) tới mạng hồi tiếp RNN/LSTM giải quyết triệt tiêu gradient; cơ chế Bahdanau Attention trong Seq2Seq; kiến trúc Transformer (Scaled Dot-Product, Multi-Head Attention, Positional Encoding); và các mô hình tiền huấn luyện (BERT, PhoBERT tiếng Việt, GPT, Fine-tuning & RAG).',
    outcomes: [
      'Dẫn xuất hàm mục tiêu Negative Sampling của Word2Vec và phân tích cơ chế cổng giữ luồng gradient trong LSTM.',
      'Cài đặt đầy đủ Scaled Dot-Product Multi-Head Attention và Causal Mask từ đầu.',
      'Đánh giá quy trình tiền huấn luyện và tinh chỉnh mô hình ngôn ngữ (bao gồm PhoBERT cho tiếng Việt & RAG có trích dẫn).',
    ],
    sources: [
      'Mikolov et al. (2013) — Word2Vec & Negative Sampling',
      'Hochreiter & Schmidhuber (1997) — LSTM; Goldberg (2016)',
      'Bahdanau et al. (2015); Vaswani et al. (2017) — Attention Is All You Need',
      'Nguyen & Nguyen (2020) — PhoBERT; Jurafsky & Martin (Ch. 5, 14)',
    ],
    plannedTopics: [
      'Unit 8.1: Word2Vec (Skip-gram, CBOW, Negative Sampling) & Dense Embeddings',
      'Unit 8.2: RNN, BPTT & LSTM/GRU (Giải quyết Gradient triệt tiêu)',
      'Unit 8.3: Seq2Seq & Cơ chế Bahdanau Attention',
      'Unit 8.4: Transformer (Multi-Head Attention, Positional Encoding, LayerNorm)',
      'Unit 8.5: Pre-trained LMs (BERT, PhoBERT tiếng Việt, GPT), Fine-tuning & RAG',
    ],
    units: [],
  },
  {
    id: 9,
    title: 'Học trên Dữ liệu Đồ thị (Graph Neural Networks)',
    englishTitle: 'Graph Representation Learning & GNNs',
    color: '#2dd4bf',
    status: 'planned',
    overview:
      'Lý thuyết phổ đồ thị (Graph Laplacian & Spectral Convolutions), khung truyền thông điệp tổng quát (Message Passing — MPNN), phân tích toán học các kiến trúc chuẩn GCN (Kipf & Welling), GraphSAGE, Graph Attention Networks (GAT), bài toán phân loại Nút/Cạnh/Đồ thị và đồ thị động (Dynamic Graphs).',
    outcomes: [
      'Phân tích phổ trị riêng của ma trận Graph Laplacian chuẩn hóa và phép xấp xỉ bậc nhất Chebyshev trong GCN.',
      'Cài đặt lớp truyền thông điệp GCN và Masked Softmax Attention (GAT) bảo toàn tính bất biến hoán vị nút.',
      'Thiết kế quy trình chia tập và đánh giá hợp lệ cho bài toán trên một đồ thị lớn (Transductive) và nhiều đồ thị (Inductive).',
    ],
    sources: [
      'Zhiyuan Liu & Jie Zhou — Introduction to Graph Neural Networks (§2.3, Ch. 4–5, Ch. 7, Ch. 11)',
      'Phòng lab GNN tích hợp sẵn trên ML Studio',
    ],
    plannedTopics: [
      'Unit 9.1: Graph Laplacian & Spectral Graph Convolutions',
      'Unit 9.2: Khung Message Passing (MPNN) & Tính đẳng biến hoán vị',
      'Unit 9.3: Kiến trúc GCN, GraphSAGE & Graph Attention Networks (GAT)',
      'Unit 9.4: Phân loại Nút/Cạnh/Đồ thị & Đánh giá Đồ thị động',
    ],
    units: [],
  },
  {
    id: 10,
    title: 'Học Tăng cường & Đồ án Nghiên cứu Cao học',
    englishTitle: 'Reinforcement Learning & Graduate Capstone Research',
    color: '#fb7185',
    status: 'planned',
    overview:
      'Bài toán Multi-Armed Bandits (ε-greedy, UCB, Thompson Sampling), Markov Decision Processes (MDPs) và Phương trình Bellman; Quy hoạch động (Policy/Value Iteration); Học không mô hình (Monte Carlo, TD(0), SARSA, Q-Learning, DQN); Policy Gradient (REINFORCE, Actor-Critic, PPO) và Đồ án Cao học hoàn chỉnh (Ablation Study, Error Analysis, Reproducibility).',
    outcomes: [
      'Giải phương trình tối ưu Bellman bằng Value Iteration / Policy Iteration trên bài toán MDP rời rạc.',
      'Phân biệt chiến lược On-policy (SARSA) và Off-policy (Q-Learning) cùng định lý hội tụ xấp xỉ hàm.',
      'Thực hiện hoàn chỉnh một nghiên cứu thực nghiệm độc lập có Ablation Study, Seed Variance và phân tích lỗi định lượng.',
    ],
    sources: [
      'Sutton & Barto — Reinforcement Learning: An Introduction (2nd Ed.)',
      'Stanford CS229 (Part V: Reinforcement Learning)',
    ],
    plannedTopics: [
      'Unit 10.1: Multi-Armed Bandits (ε-greedy, UCB, Thompson Sampling)',
      'Unit 10.2: MDPs, Phương trình Bellman & Dynamic Programming',
      'Unit 10.3: Model-Free RL: Monte Carlo, TD(0), SARSA, Q-Learning & DQN',
      'Unit 10.4: Policy Gradient: REINFORCE, Actor-Critic & PPO',
      'Unit 10.5: Đồ án Cao học: Thực nghiệm tái lập, Ablation Study & Phân tích lỗi',
    ],
    units: [],
  },
];

export function getModule(id: number): GradModule | undefined {
  return GRAD_MODULES.find((m) => m.id === id);
}

export function getUnit(unitId: string): { module: GradModule; unit: GradUnit } | undefined {
  for (const mod of GRAD_MODULES) {
    const unit = mod.units.find((u) => u.id === unitId);
    if (unit) return { module: mod, unit };
  }
  return undefined;
}

export function allReadyUnits(): GradUnit[] {
  return GRAD_MODULES.flatMap((m) => m.units);
}

export const GRAD_RESOURCES: GradResourceItem[] = [
  {
    id: 'R01',
    title: 'Probabilistic Machine Learning: An Introduction',
    authors: 'Kevin P. Murphy (MIT Press)',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
    modules: 'Module 1, 2, 3, 4, 5, 6, 7',
    note: 'Giáo trình trụ cột về suy luận Bayes, tối ưu hóa, GLMs, mô hình sinh và đồ thị xác suất.',
  },
  {
    id: 'R02',
    title: 'Stanford CS229: Machine Learning Complete Lecture Notes',
    authors: 'Andrew Ng & Tengyu Ma',
    path: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
    modules: 'Module 1, 2, 4, 6, 7, 10',
    note: 'Tài liệu nền tảng về GLMs, SVM/Kernel, Learning Theory (Hoeffding/VC), EM và Reinforcement Learning.',
  },
  {
    id: 'R03',
    title: 'UC Berkeley CS189/289A: Introduction to Machine Learning (Lectures 01–13)',
    authors: 'UC Berkeley',
    path: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/',
    modules: 'Module 1, 2, 3, 6',
    note: 'Bài giảng chuẩn cao học về Data Tools, K-means, GMM, Linear/Logistic Regression, Bias–Variance và SGD.',
  },
  {
    id: 'R04',
    title: 'An Introduction to Statistical Learning (7th Printing)',
    authors: 'James, Witten, Hastie & Tibshirani',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
    modules: 'Module 1, 2, 5',
    note: 'Tài liệu chuẩn mực về Resampling, Cross-Validation, Regularization và Tree-Based Ensembles.',
  },
  {
    id: 'R05',
    title: 'Machine Learning cơ bản (Bản tiêu chuẩn & Bản in màu)',
    authors: 'Vũ Hữu Tiệp',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/book_ML_color.pdf',
    modules: 'Module 1, 3, 4',
    note: 'Trình bày trực quan tiếng Việt về đại số tuyến tính, tối ưu lồi và SVM.',
  },
  {
    id: 'R06',
    title: 'Core NLP & Transformer Research Papers (2013–2020)',
    authors: 'Mikolov, Hochreiter, Bahdanau, Vaswani, Goldberg, Jurafsky & Martin, Nguyen (PhoBERT)',
    path: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/',
    modules: 'Module 8',
    note: 'Tuyển tập các bài báo gốc định hình Word2Vec, LSTM, Attention, Transformer và PhoBERT tiếng Việt.',
  },
  {
    id: 'R07',
    title: 'Computer Vision Lectures & OpenCV Practice',
    authors: 'Bộ bài giảng Thị giác Máy tính',
    path: '05_Thi_giac_May_tinh_Computer_Vision/',
    modules: 'Module 7',
    note: 'Cơ sở biểu diễn ảnh, bộ lọc không gian, tăng cường ảnh và thực hành Python/OpenCV.',
  },
  {
    id: 'R08',
    title: 'Introduction to Graph Neural Networks',
    authors: 'Zhiyuan Liu & Jie Zhou',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
    modules: 'Module 9',
    note: 'Giáo trình chuyên sâu về Graph Convolutions, Message Passing, GAT và đồ thị động.',
  },
  {
    id: 'R09',
    title: 'Reinforcement Learning: An Introduction (2nd Edition)',
    authors: 'Richard S. Sutton & Andrew G. Barto',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
    modules: 'Module 10',
    note: 'Kinh điển số 1 về Bandits, MDPs, Dynamic Programming, TD Learning và Policy Gradient.',
  },
];
