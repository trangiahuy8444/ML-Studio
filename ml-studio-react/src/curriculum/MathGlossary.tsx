import { BookOpen } from 'lucide-react';
import { MathText } from '../components/MathText';
import type { GradSymbolDef, GradUnit } from './types';

interface SymbolEntry extends GradSymbolDef {
  patterns: RegExp[];
}

const SYMBOL_BANK: SymbolEntry[] = [
  {
    symbol: '$\\mathbb{R}^d,\\; \\mathbb{R}^{n \\times d}$',
    readAs: 'Không gian số thực $d$ chiều / Bảng $n$ dòng $d$ cột',
    meaning:
      'Một điểm dữ liệu có $d$ con số gọi là thuộc $\\mathbb{R}^d$; một bảng Excel có $n$ dòng mẫu và $d$ cột đặc trưng gọi là ma trận thuộc $\\mathbb{R}^{n \\times d}$.',
    patterns: [/\\mathbb\{R\}/],
  },
  {
    symbol: '$x \\in S$',
    readAs: 'Phần tử $x$ thuộc tập $S$',
    meaning: 'Cách viết ngắn nói biến $x$ nằm trong danh sách hoặc miền giá trị $S$ (giống `x in S` trong Python).',
    patterns: [/\\in\b/, /\\notin\b/],
  },
  {
    symbol: '$\\sum_{i=1}^n a_i$',
    readAs: 'Tổng Sigma từ $i=1$ đến $n$',
    meaning: 'Chính là vòng lặp cộng dồn `total = 0; for i in range(n): total += a[i]` hoặc `np.sum(a)`.',
    patterns: [/\\sum\b/],
  },
  {
    symbol: '$\\prod_{i=1}^n a_i$',
    readAs: 'Tích Pi từ $i=1$ đến $n$',
    meaning: 'Vòng lặp nhân dồn `prod = 1; for a_i in a: prod *= a_i` hoặc `np.prod(a)`.',
    patterns: [/\\prod\b/],
  },
  {
    symbol: '$x^\\top, \\; X^\\top$',
    readAs: 'Chuyển vị (Transpose)',
    meaning:
      'Xoay bảng ngang thành dọc (dòng đổi thành cột). Trong Python/NumPy chính là `X.T`. Dùng để nhân tích vô hướng $a^\\top b = \\sum a_i b_i$.',
    patterns: [/\^\\top\b/, /\^\{\\top\}/],
  },
  {
    symbol: '$\\|x\\|_2, \\; \\|x\\|_1$',
    readAs: 'Chuẩn (Norm) / Độ dài vector',
    meaning:
      '$\\|x\\|_2$ là độ dài đường chim bay (Pythagoras $\\sqrt{\\sum x_i^2}$); $\\|x\\|_1$ là tổng giá trị tuyệt đối $\\sum |x_i|$; $\\|X\\|_F$ là căn tổng bình phương mọi ô trong ma trận.',
    patterns: [/\\\|/],
  },
  {
    symbol: '$\\nabla_w L(w)$',
    readAs: 'Gradient (Véc-tơ đạo hàm riêng theo $w$)',
    meaning:
      'Mảng chứa độ dốc theo từng tham số `[dL/dw_0, dL/dw_1, ...]`. Luôn chỉ theo hướng làm $L$ tăng nhanh nhất — muốn giảm sai số thì đi ngược dấu `w = w - lr * grad`.',
    patterns: [/\\nabla\b/],
  },
  {
    symbol: '$\\frac{\\partial L}{\\partial w}$',
    readAs: 'Đạo hàm riêng của $L$ theo $w$',
    meaning:
      'Đo xem khi nhích nhẹ biến $w$ lên $0.001$ (giữ nguyên mọi biến khác) thì $L$ thay đổi gấp mấy lần lượng nhích đó.',
    patterns: [/\\partial\b/],
  },
  {
    symbol: '$\\arg\\min_w f(w), \\; \\arg\\max_w$',
    readAs: 'Giá trị $w$ đạt cực tiểu / cực đại',
    meaning:
      'Khác với `min` trả về giá trị nhỏ nhất, `argmin` trả về **vị trí / tham số $w$** làm hàm nhỏ nhất (giống `np.argmin` trả về chỉ số).',
    patterns: [/\\arg\\min/, /\\arg\\max/, /\\argmin/, /\\argmax/],
  },
  {
    symbol: '$\\mu, \\; \\bar{x}$',
    readAs: 'Trung bình (Mean / Muy)',
    meaning: 'Giá trị trung bình cộng của một dãy số (`np.mean(x)`).',
    patterns: [/\\mu\b/, /\\bar\{/],
  },
  {
    symbol: '$\\sigma, \\; \\sigma^2, \\; \\Sigma$',
    readAs: 'Độ lệch chuẩn ($\\sigma$), Phương sai ($\\sigma^2$), Ma trận hiệp phương sai ($\\Sigma$)',
    meaning:
      '$\\sigma$ (`np.std`) đo mức độ phân tán quanh trung bình; $\\sigma^2$ là bình phương của $\\sigma$ (`np.var`); $\\Sigma$ là bảng đo độ biến thiên chung giữa từng cặp cột (`np.cov`). (Phân biệt $\\Sigma$ lớn in nghiêng với dấu tổng $\\sum$!).',
    patterns: [/\\sigma\b/, /\\Sigma\b/],
  },
  {
    symbol: '$p(y \\mid x)$',
    readAs: 'Xác suất có điều kiện: $y$ khi ĐÃ BIẾT $x$',
    meaning:
      'Dấu gạch đứng `|` đọc là **"biết rằng / cho trước"**. Nghĩa là khi đã quan sát được dữ liệu đầu vào $x$, xác suất nhãn là $y$ bằng bao nhiêu.',
    patterns: [/\\mid\b/],
  },
  {
    symbol: '$\\mathbb{E}[X], \\; \\text{Var}(X)$',
    readAs: 'Kỳ vọng toán (Expected Value) & Phương sai',
    meaning:
      '$\\mathbb{E}[X]$ là giá trị trung bình dài hạn nếu lặp lại thí nghiệm vô số lần; $\\text{Var}(X)$ đo độ dao động trung bình bình phương quanh kỳ vọng.',
    patterns: [/\\mathbb\{E\}/, /\\text\{Var\}/],
  },
  {
    symbol: '$\\mathcal{N}(\\mu, \\sigma^2)$',
    readAs: 'Phân phối chuẩn Gaussian (hình chuông)',
    meaning: 'Phân phối hình quả chuông đối xứng xoay quanh tâm $\\mu$ và bề rộng chuông do $\\sigma$ quyết định.',
    patterns: [/\\mathcal\{N\}/],
  },
  {
    symbol: '$\\odot$',
    readAs: 'Nhân từng phần tử (Hadamard / Element-wise)',
    meaning:
      'Nhân hai mảng cùng kích thước theo đúng từng ô tương ứng — chính là toán tử `*` thông thường trên `np.array` (khác với `@` là nhân ma trận).',
    patterns: [/\\odot\b/],
  },
  {
    symbol: '$\\propto$',
    readAs: 'Tỷ lệ thuận với (Proportional to)',
    meaning: 'Hai vế chỉ chênh nhau một hằng số chuẩn hóa dương không phụ thuộc biến đang tối ưu nên có thể tạm bỏ qua hằng số đó.',
    patterns: [/\\propto\b/],
  },
  {
    symbol: '$\\implies, \\; \\Longleftrightarrow$',
    readAs: 'Suy ra / Tương đương với',
    meaning: 'Dấu nối bước biến đổi logic hoặc đại số (chỉ ra công thức bên phải được rút ra từ biểu thức bên trái).',
    patterns: [/\\implies\b/, /\\Longleftrightarrow\b/, /\\iff\b/],
  },
  {
    symbol: '$\\mathbb{I}[\\text{điều kiện}]$',
    readAs: 'Hàm chỉ thị (Indicator Function)',
    meaning: 'Trả về `1` nếu điều kiện bên trong đúng, `0` nếu sai — giống hệt `float(cond)` hay `mask.astype(float)` trong NumPy.',
    patterns: [/\\mathbb\{I\}/],
  },
  {
    symbol: '$\\lambda, \\; \\alpha, \\; \\eta, \\; \\gamma$',
    readAs: 'Tham số điều chỉnh (Hyperparameters: Lambda, Alpha, Eta, Gamma)',
    meaning:
      'Các chữ Hy Lạp thường dùng đặt tên cho các núm vặn cấu hình thuật toán như: độ phạt $\\lambda$, tốc độ học (learning rate) $\\alpha$ hoặc $\\eta$, hệ số chiết khấu $\\gamma$.',
    patterns: [/\\lambda\b/, /\\alpha\b/, /\\eta\b/, /\\gamma\b/],
  },
  {
    symbol: '$\\text{Tr}(A)$',
    readAs: 'Vết ma trận (Trace)',
    meaning: 'Tổng các phần tử nằm trên đường chéo chính của ma trận vuông (`np.trace(A)`).',
    patterns: [/\\text\{Tr\}/],
  },
  {
    symbol: '$\\forall, \\; \\exists$',
    readAs: 'Với mọi / Tồn tại',
    meaning: '$\\forall x$: đúng cho mọi giá trị $x$ (`all(...)`); $\\exists x$: có ít nhất một giá trị $x$ thỏa mãn (`any(...)`).',
    patterns: [/\\forall\b/, /\\exists\b/],
  },
  {
    symbol: '$\\mathcal{H}, \\; \\mathcal{D}, \\; \\mathcal{L}$',
    readAs: 'Tập giả thuyết $\\mathcal{H}$, Tập dữ liệu $\\mathcal{D}$, Hàm mất mát $\\mathcal{L}$',
    meaning: 'Các chữ cái in hoa nét uốn dùng ký hiệu cho cả một "tập hợp" hoặc hàm mục tiêu chung của bài toán.',
    patterns: [/\\mathcal\{[HDL]\}/],
  },
];

export function detectUnitSymbols(unit: GradUnit): GradSymbolDef[] {
  const custom: GradSymbolDef[] = [];
  const seen = new Set<string>();

  for (const sec of unit.sections) {
    if (sec.symbols) {
      for (const s of sec.symbols) {
        if (!seen.has(s.symbol)) {
          seen.add(s.symbol);
          custom.push(s);
        }
      }
    }
  }

  const corpus = [
    unit.summary,
    unit.bigPicture ?? '',
    ...unit.outcomes,
    ...unit.sections.flatMap((s) => [s.body, s.math ?? '', s.intuition ?? '', s.pitfall ?? '']),
    ...unit.exercises.flatMap((e) => [e.prompt, e.hint]),
  ].join('\n');

  const autoMatched: GradSymbolDef[] = [];
  for (const entry of SYMBOL_BANK) {
    if (seen.has(entry.symbol)) continue;
    if (entry.patterns.some((re) => re.test(corpus))) {
      seen.add(entry.symbol);
      autoMatched.push({ symbol: entry.symbol, readAs: entry.readAs, meaning: entry.meaning });
    }
  }

  return [...custom, ...autoMatched].slice(0, 14);
}

export function UnitSymbolGlossary({ unit, defaultOpen }: { unit: GradUnit; defaultOpen: boolean }) {
  const symbols = detectUnitSymbols(unit);
  if (symbols.length === 0) return null;

  return (
    <details className="symbol-glossary-box" open={defaultOpen}>
      <summary>
        <BookOpen size={15} />
        <span>
          Bảng giải mã nhanh {symbols.length} ký hiệu toán học dùng trong Bài {unit.moduleId}.{unit.order} (nếu thấy ký hiệu lạ hãy tra tại đây)
        </span>
      </summary>
      <div className="symbol-glossary-grid">
        {symbols.map((item) => (
          <div className="symbol-item" key={item.symbol}>
            <div className="symbol-badge">
              <MathText text={item.symbol} />
            </div>
            <div className="symbol-desc">
              <div className="symbol-read">
                <MathText text={item.readAs} />
              </div>
              <div className="symbol-meaning">
                <MathText text={item.meaning} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
