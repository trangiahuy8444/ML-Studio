import { useMemo, useState } from 'react';
import { Chart } from '../components/Chart';
import { solve } from '../lib/domain';
import type { PlotTrace } from '../types';
import type { GradSimulationId } from './types';

const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : 'NaN');

function sig(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))) : Math.exp(z) / (1 + Math.exp(z));
}

/** Unit 4.1: LDA (Shared Σ) vs QDA (Distinct Σ_0 ≠ Σ_1) */
function SimGdaLdaQda() {
  const [modelType, setModelType] = useState<'lda' | 'qda'>('qda');
  const [rotScale, setRotScale] = useState(0.75);

  const { traces, acc } = useMemo(() => {
    const mu0 = [-1.0, -0.4];
    const mu1 = [1.0, 0.4];
    // Class 0 points (isotropic-like)
    const c0 = [
      [-1.8, -0.9], [-1.4, 0.3], [-0.9, -1.1], [-0.5, -0.3], [-1.1, 0.2], [-0.4, -0.9], [-1.5, -0.5], [-0.8, 0.4],
    ];
    // Class 1 points stretched & tilted by rotScale
    const raw1 = [
      [0.3, -1.2], [0.7, -0.5], [1.0, 0.2], [1.3, 0.9], [1.7, 1.5], [0.6, 0.8], [1.4, -0.1], [0.9, 1.1],
    ];
    const c1 = raw1.map(([x, y]) => [mu1[0] + (x - mu1[0]) * 1.15, mu1[1] + rotScale * (x - mu1[0]) + (y - mu1[1]) * 0.45]);

    const cov2 = (pts: number[][], mu: number[]) => {
      let s00 = 1e-3, s01 = 0, s11 = 1e-3;
      for (const [x, y] of pts) {
        const dx = x - mu[0], dy = y - mu[1];
        s00 += dx * dx;
        s01 += dx * dy;
        s11 += dy * dy;
      }
      const n = pts.length;
      return [[s00 / n, s01 / n], [s01 / n, s11 / n]];
    };

    const m0 = [c0.reduce((s, r) => s + r[0], 0) / c0.length, c0.reduce((s, r) => s + r[1], 0) / c0.length];
    const m1 = [c1.reduce((s, r) => s + r[0], 0) / c1.length, c1.reduce((s, r) => s + r[1], 0) / c1.length];
    const S0 = cov2(c0, m0);
    const S1 = cov2(c1, m1);
    const Spooled = [
      [0.5 * (S0[0][0] + S1[0][0]), 0.5 * (S0[0][1] + S1[0][1])],
      [0.5 * (S0[1][0] + S1[1][0]), 0.5 * (S0[1][1] + S1[1][1])],
    ];

    const logGauss = (x: number, y: number, mu: number[], S: number[][]) => {
      const det = Math.max(1e-9, S[0][0] * S[1][1] - S[0][1] * S[1][0]);
      const inv = [
        [S[1][1] / det, -S[0][1] / det],
        [-S[1][0] / det, S[0][0] / det],
      ];
      const dx = x - mu[0], dy = y - mu[1];
      const quad = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
      return -0.5 * (Math.log(det) + quad);
    };

    const post1 = (x: number, y: number) => {
      const A = modelType === 'lda' ? Spooled : S0;
      const B = modelType === 'lda' ? Spooled : S1;
      const l0 = logGauss(x, y, m0, A);
      const l1 = logGauss(x, y, m1, B);
      return sig(l1 - l0);
    };

    const grid = Array.from({ length: 30 }, (_, i) => -2.5 + (i / 29) * 5.0);
    const Z = grid.map((yVal) => grid.map((xVal) => post1(xVal, yVal)));

    const correct =
      c0.filter(([x, y]) => post1(x, y) < 0.5).length + c1.filter(([x, y]) => post1(x, y) >= 0.5).length;

    const pTraces: PlotTrace[] = [
      {
        type: 'heatmap',
        x: grid,
        y: grid,
        z: Z,
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, '#0f2942'],
          [0.5, '#1e3a5f'],
          [1, '#581c4c'],
        ],
        showscale: true,
        name: 'P(y=1|x)',
      },
      {
        type: 'contour',
        x: grid,
        y: grid,
        z: Z,
        contours: { start: 0.5, end: 0.5, size: 0.1, coloring: 'none' },
        line: { color: '#ffd173', width: 3 },
        showscale: false,
        name: 'Biên P = 0.5',
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: c0.map((r) => r[0]),
        y: c0.map((r) => r[1]),
        name: 'Lớp 0 (Σ₀)',
        marker: { size: 10, color: '#42d3e5', line: { color: '#fff', width: 1.5 } },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: c1.map((r) => r[0]),
        y: c1.map((r) => r[1]),
        name: 'Lớp 1 (Σ₁)',
        marker: { size: 10, color: '#f28db9', line: { color: '#fff', width: 1.5 } },
      },
    ];
    void mu0;
    return { traces: pTraces, acc: correct / (c0.length + c1.length) };
  }, [modelType, rotScale]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Giả định Ma trận Hiệp phương sai
          <select value={modelType} onChange={(e) => setModelType(e.target.value as 'lda' | 'qda')}>
            <option value="qda">QDA (Σ₀ ≠ Σ₁ → Biên bậc hai uốn cong)</option>
            <option value="lda">LDA (Dùng chung Σ → Biên đường thẳng)</option>
          </select>
        </label>
        <label>
          Độ xoay chéo cụm Lớp 1: <strong>{fmt(rotScale, 2)}</strong>
          <input type="range" min={-1.2} max={1.5} step={0.1} value={rotScale} onChange={(e) => setRotScale(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="x₁" yTitle="x₂" height={340} />
      <p className="sim-caption">
        Độ chính xác phân loại: <strong>{fmt(acc * 100, 1)}%</strong>. {modelType === 'lda'
          ? 'LDA ép hai lớp dùng chung Σ nên biên quyết định P = 0.5 luôn là đường thẳng.'
          : 'QDA cho phép từng lớp có ma trận Σₖ riêng biệt nên tạo biên parabol/elip ôm sát cấu trúc dữ liệu.'}
      </p>
    </div>
  );
}

/** Unit 4.2: Soft-Margin SVM Margin Width, C & Support Vectors */
function SimSvmMargin() {
  const [logC, setLogC] = useState(0.5);
  const [withOutlier, setWithOutlier] = useState(true);

  const { traces, marginWidth, svCount } = useMemo(() => {
    const basePts = [
      { x1: -1.6, x2: -0.9, y: -1 },
      { x1: -1.2, x2: -0.2, y: -1 },
      { x1: -0.8, x2: -1.2, y: -1 },
      { x1: -0.6, x2: -0.5, y: -1 },
      { x1: 0.6, x2: 0.5, y: 1 },
      { x1: 0.9, x2: 1.3, y: 1 },
      { x1: 1.4, x2: 0.7, y: 1 },
      { x1: 1.6, x2: 1.5, y: 1 },
    ];
    if (withOutlier) basePts.push({ x1: 0.3, x2: 0.25, y: -1 }); // Near-margin outlier

    const C = Math.pow(10, logC);
    let w1 = 0.5, w2 = 0.5, b = 0.0;
    for (let step = 1; step <= 600; step++) {
      const lr = 0.08 / (1 + 0.02 * step);
      let dw1 = w1, dw2 = w2, db = 0;
      for (const p of basePts) {
        if (p.y * (w1 * p.x1 + w2 * p.x2 + b) < 1.0) {
          dw1 -= C * p.y * p.x1;
          dw2 -= C * p.y * p.x2;
          db -= C * p.y;
        }
      }
      w1 -= lr * dw1;
      w2 -= lr * dw2;
      b -= lr * db;
    }

    const svMask = basePts.map((p) => p.y * (w1 * p.x1 + w2 * p.x2 + b) <= 1.08);
    const xs = [-2.0, 2.0];
    const lineY = (offset: number) =>
      Math.abs(w2) > 1e-4 ? xs.map((x) => (offset - b - w1 * x) / w2) : [0, 0];

    const cNeg = basePts.filter((p) => p.y === -1);
    const cPos = basePts.filter((p) => p.y === 1);
    const svs = basePts.filter((_, i) => svMask[i]);

    const pTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: cNeg.map((p) => p.x1),
        y: cNeg.map((p) => p.x2),
        name: 'Lớp −1',
        marker: { size: 10, color: '#42d3e5' },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: cPos.map((p) => p.x1),
        y: cPos.map((p) => p.x2),
        name: 'Lớp +1',
        marker: { size: 10, color: '#f28db9' },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: svs.map((p) => p.x1),
        y: svs.map((p) => p.x2),
        name: 'Support Vectors (αᵢ > 0)',
        marker: { size: 17, color: 'transparent', line: { color: '#ffd173', width: 2.5 } },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: xs,
        y: lineY(0),
        name: 'Biên quyết định wᵀx + b = 0',
        line: { color: '#e6eef8', width: 2.5 },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: xs,
        y: lineY(1),
        name: 'Lề +1',
        line: { color: '#84e0a8', width: 1.5, dash: 'dash' },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: xs,
        y: lineY(-1),
        name: 'Lề −1',
        line: { color: '#84e0a8', width: 1.5, dash: 'dash' },
      },
    ];

    const norm = Math.hypot(w1, w2) || 1e-6;
    return { traces: pTraces, marginWidth: 2.0 / norm, svCount: svs.length };
  }, [logC, withOutlier]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Phạt vi phạm biên log₁₀(C): <strong>{fmt(logC, 2)}</strong> (C = {Math.pow(10, logC).toFixed(2)})
          <input type="range" min={-1.0} max={2.0} step={0.1} value={logC} onChange={(e) => setLogC(+e.target.value)} />
        </label>
        <label>
          Nhiễu gần biên quyết định
          <select value={withOutlier ? 'yes' : 'no'} onChange={(e) => setWithOutlier(e.target.value === 'yes')}>
            <option value="yes">Có điểm ngoại lệ lớp −1 lấn sâu</option>
            <option value="no">Tách biệt tuyến tính sạch</option>
          </select>
        </label>
      </div>
      <Chart traces={traces} xTitle="x₁" yTitle="x₂" height={340} layout={{ xaxis: { range: [-2, 2] }, yaxis: { range: [-2, 2] } }} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Bề rộng hành lang lề (2 / ||w||₂)</span>
          <strong style={{ color: '#84e0a8' }}>{fmt(marginWidth, 3)}</strong>
        </div>
        <div className="metric-card">
          <span>Số điểm tựa (Support Vectors)</span>
          <strong style={{ color: '#ffd173' }}>{svCount} điểm</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 4.3: RBF Kernel Non-linear Boundary & Overfitting with Gamma */
function SimRbfKernel() {
  const [gamma, setGamma] = useState(1.2);
  const [logLam, setLogLam] = useState(-2.0);

  const { traces, acc } = useMemo(() => {
    const pts: { x1: number; x2: number; y: number }[] = [];
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const rIn = 0.5 + 0.1 * Math.sin(i * 3);
      pts.push({ x1: rIn * Math.cos(a), x2: rIn * Math.sin(a), y: 1 });
      const rOut = 1.45 + 0.15 * Math.cos(i * 2);
      pts.push({ x1: rOut * Math.cos(a + 0.15), x2: rOut * Math.sin(a + 0.15), y: -1 });
    }
    const n = pts.length;
    const lam = Math.pow(10, logLam);
    const kRbf = (ax: number, ay: number, bx: number, by: number) =>
      Math.exp(-gamma * ((ax - bx) ** 2 + (ay - by) ** 2));

    const K = pts.map((p, r) =>
      pts.map((q, c) => kRbf(p.x1, p.x2, q.x1, q.x2) + (r === c ? lam : 0)),
    );
    const alpha = solve(K, pts.map((p) => p.y));
    const predict = (x: number, y: number) => pts.reduce((s, p, i) => s + alpha[i] * kRbf(x, y, p.x1, p.x2), 0);

    const grid = Array.from({ length: 28 }, (_, i) => -2.0 + (i / 27) * 4.0);
    const Z = grid.map((yv) => grid.map((xv) => predict(xv, yv)));
    const ok = pts.filter((p) => (predict(p.x1, p.x2) >= 0 ? 1 : -1) === p.y).length;

    const pTraces: PlotTrace[] = [
      {
        type: 'heatmap',
        x: grid,
        y: grid,
        z: Z,
        zmin: -1.2,
        zmax: 1.2,
        colorscale: [
          [0, '#0f2942'],
          [0.5, '#1e293b'],
          [1, '#581c4c'],
        ],
        showscale: true,
        name: 'f(x)',
      },
      {
        type: 'contour',
        x: grid,
        y: grid,
        z: Z,
        contours: { start: 0, end: 0, size: 0.1, coloring: 'none' },
        line: { color: '#ffd173', width: 3 },
        showscale: false,
        name: 'Biên f(x) = 0',
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: pts.filter((p) => p.y === 1).map((p) => p.x1),
        y: pts.filter((p) => p.y === 1).map((p) => p.x2),
        name: 'Vòng trong (+1)',
        marker: { size: 9, color: '#f28db9', line: { color: '#fff', width: 1.5 } },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: pts.filter((p) => p.y === -1).map((p) => p.x1),
        y: pts.filter((p) => p.y === -1).map((p) => p.x2),
        name: 'Vòng ngoài (−1)',
        marker: { size: 9, color: '#42d3e5', line: { color: '#fff', width: 1.5 } },
      },
    ];
    return { traces: pTraces, acc: ok / n };
  }, [gamma, logLam]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Băng thông nghịch đảo γ: <strong>{fmt(gamma, 2)}</strong>
          <input type="range" min={0.1} max={8.0} step={0.2} value={gamma} onChange={(e) => setGamma(+e.target.value)} />
        </label>
        <label>
          Ridge log₁₀(λ): <strong>{fmt(logLam, 1)}</strong>
          <input type="range" min={-4} max={0.5} step={0.5} value={logLam} onChange={(e) => setLogLam(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="x₁" yTitle="x₂" height={340} />
      <p className="sim-caption">
        Train accuracy: <strong>{fmt(acc * 100, 1)}%</strong>. Quan sát khi γ quá lớn (&gt; 5), hàm nhân RBF chỉ bám vào từng hạt riêng lẻ quanh các điểm train thay vì tạo vành tròn mượt.
      </p>
    </div>
  );
}

/** Unit 5.1: Curse of Dimensionality */
function SimCurseDim() {
  const [eps, setEps] = useState(0.1);
  const [maxD, setMaxD] = useState(100);

  const { shellTraces, contrastTraces } = useMemo(() => {
    const dims = Array.from({ length: 16 }, (_, i) => Math.max(1, Math.round(1 + (i / 15) * (maxD - 1))));
    const shell = dims.map((d) => 1 - Math.pow(1 - eps, d));

    // Deterministic relative contrast (d_max - d_min) / d_min for uniform random vectors
    const contrast = dims.map((d) => {
      const dists: number[] = [];
      let s = 1337;
      const rnd = () => {
        s = (1664525 * s + 1013904223) >>> 0;
        return s / 4294967296;
      };
      for (let i = 0; i < 40; i++) {
        let sum2 = 0;
        for (let j = 0; j < d; j++) {
          const v = rnd() - 0.5;
          sum2 += v * v;
        }
        dists.push(Math.sqrt(sum2));
      }
      const dMin = Math.min(...dists);
      const dMax = Math.max(...dists);
      return dMin > 0 ? (dMax - dMin) / dMin : 0;
    });

    const sTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: dims,
        y: shell,
        name: `Tỷ lệ vỏ ngoài 1 − (1 − ${eps})^d`,
        line: { color: '#f28db9', width: 2.5 },
      },
    ];
    const cTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: dims,
        y: contrast,
        name: '(d_max − d_min) / d_min',
        line: { color: '#42d3e5', width: 2.5 },
      },
    ];
    return { shellTraces: sTraces, contrastTraces: cTraces };
  }, [eps, maxD]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ dày lớp vỏ ngoài ε: <strong>{fmt(eps, 2)}</strong>
          <input type="range" min={0.02} max={0.3} step={0.02} value={eps} onChange={(e) => setEps(+e.target.value)} />
        </label>
        <label>
          Số chiều tối đa d: <strong>{maxD}</strong>
          <input type="range" min={10} max={250} step={10} value={maxD} onChange={(e) => setMaxD(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={shellTraces} xTitle="Số chiều d" yTitle="Thể tích lớp vỏ ngoài" height={300} layout={{ yaxis: { range: [0, 1.05] } }} />
        <Chart traces={contrastTraces} xTitle="Số chiều d" yTitle="Độ tương phản khoảng cách" height={300} />
      </div>
    </div>
  );
}

/** Unit 5.2: CART Greedy Splitter */
function SimCartSplit() {
  const [thresh, setThresh] = useState(4.5);
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini');

  const samples = useMemo(
    () => [
      { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 3, y: 1 }, { x: 4, y: 0 }, { x: 5, y: 0 },
      { x: 6, y: 1 }, { x: 7, y: 1 }, { x: 8, y: 1 }, { x: 9, y: 0 }, { x: 10, y: 1 },
    ],
    [],
  );

  const { traces, gain, impL, impR, bestT } = useMemo(() => {
    const imp = (ys: number[]) => {
      if (!ys.length) return 0;
      const p1 = ys.filter((v) => v === 1).length / ys.length;
      const p0 = 1 - p1;
      if (criterion === 'gini') return 1 - p1 * p1 - p0 * p0;
      return -(p1 > 0 ? p1 * Math.log2(p1) : 0) - (p0 > 0 ? p0 * Math.log2(p0) : 0);
    };

    const allY = samples.map((s) => s.y);
    const parent = imp(allY);
    const mids = Array.from({ length: 9 }, (_, i) => i + 1.5);
    const gains = mids.map((t) => {
      const L = samples.filter((s) => s.x <= t).map((s) => s.y);
      const R = samples.filter((s) => s.x > t).map((s) => s.y);
      return parent - (L.length * imp(L) + R.length * imp(R)) / samples.length;
    });

    const curL = samples.filter((s) => s.x <= thresh).map((s) => s.y);
    const curR = samples.filter((s) => s.x > thresh).map((s) => s.y);
    const curGain = parent - (curL.length * imp(curL) + curR.length * imp(curR)) / samples.length;
    const maxIdx = gains.indexOf(Math.max(...gains));

    const pTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: mids.map((m) => `x ≤ ${m}`),
        y: gains,
        name: `Mức giảm ${criterion === 'gini' ? 'Gini' : 'Entropy'}`,
        marker: { color: mids.map((m) => (Math.abs(m - thresh) < 0.1 ? '#f28db9' : '#42d3e5')) },
      },
    ];
    return { traces: pTraces, gain: curGain, impL: imp(curL), impR: imp(curR), bestT: mids[maxIdx] };
  }, [thresh, criterion, samples]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Thước đo bất thuần
          <select value={criterion} onChange={(e) => setCriterion(e.target.value as 'gini' | 'entropy')}>
            <option value="gini">Gini Impurity</option>
            <option value="entropy">Shannon Entropy (Information Gain)</option>
          </select>
        </label>
        <label>
          Ngưỡng thử x ≤ <strong>{thresh}</strong> (Tối ưu tham lam: {bestT})
          <input type="range" min={1.5} max={9.5} step={1} value={thresh} onChange={(e) => setThresh(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Các ngưỡng trung điểm ứng viên" yTitle="Mức giảm bất thuần ΔI" height={300} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Bất thuần Nhánh Trái / Nhánh Phải</span>
          <strong style={{ color: '#84e0a8' }}>
            {fmt(impL, 3)} / {fmt(impR, 3)}
          </strong>
        </div>
        <div className="metric-card">
          <span>Độ giảm bất thuần có trọng số ΔI</span>
          <strong style={{ color: '#ffd173' }}>{fmt(gain, 4)}</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 5.3: Bagging vs Random Forest Variance Reduction Floor */
function SimBaggingVariance() {
  const [rhoRf, setRhoRf] = useState(0.18);
  const [sigma2, setSigma2] = useState(9.0);
  const [maxB, setMaxB] = useState(60);

  const { traces, varBag, varRf } = useMemo(() => {
    const rhoBag = 0.65;
    const bs = Array.from({ length: maxB }, (_, i) => i + 1);
    const yBag = bs.map((b) => rhoBag * sigma2 + ((1 - rhoBag) / b) * sigma2);
    const yRf = bs.map((b) => rhoRf * sigma2 + ((1 - rhoRf) / b) * sigma2);

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: bs, y: yBag, name: `Bagging (ρ = ${rhoBag})`, line: { color: '#f28db9', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: bs, y: yRf, name: `Random Forest (ρ = ${fmt(rhoRf, 2)})`, line: { color: '#42d3e5', width: 3 } },
      { type: 'scatter', mode: 'lines', x: [1, maxB], y: [rhoRf * sigma2, rhoRf * sigma2], name: 'Sàn giới hạn RF (ρσ²)', line: { color: '#84e0a8', width: 1.5, dash: 'dot' } },
    ];
    return { traces: pTraces, varBag: yBag[yBag.length - 1], varRf: yRf[yRf.length - 1] };
  }, [rhoRf, sigma2, maxB]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Tương quan cặp cây Random Forest ρ: <strong>{fmt(rhoRf, 2)}</strong>
          <input type="range" min={0.02} max={0.6} step={0.02} value={rhoRf} onChange={(e) => setRhoRf(+e.target.value)} />
        </label>
        <label>
          Phương sai từng cây đơn lẻ σ²: <strong>{fmt(sigma2, 1)}</strong>
          <input type="range" min={1} max={16} step={1} value={sigma2} onChange={(e) => setSigma2(+e.target.value)} />
        </label>
        <label>
          Số cây B: <strong>{maxB}</strong>
          <input type="range" min={10} max={120} step={10} value={maxB} onChange={(e) => setMaxB(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Số lượng cây B" yTitle="Phương sai tổ hợp Var(f̄)" height={310} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Var Bagging tại B = {maxB}</span>
          <strong style={{ color: '#f28db9' }}>{fmt(varBag, 3)}</strong>
        </div>
        <div className="metric-card">
          <span>Var Random Forest tại B = {maxB}</span>
          <strong style={{ color: '#42d3e5' }}>{fmt(varRf, 3)}</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 5.4: Stagewise Gradient Boosting with Depth-1 Stumps */
function SimGradientBoosting1D() {
  const [rounds, setRounds] = useState(6);
  const [lr, setLr] = useState(0.5);

  const { fitTraces, resTraces, trainMse } = useMemo(() => {
    const xs = Array.from({ length: 24 }, (_, i) => -1.5 + (i / 23) * 3.0);
    const ys = xs.map((x) => Math.sin(2.0 * x) + 0.15 * x);
    let F = xs.map(() => 0);

    // Candidate split thresholds
    const mids = xs.slice(0, -1).map((x, i) => 0.5 * (x + xs[i + 1]));

    for (let m = 0; m < rounds; m++) {
      const residuals = ys.map((y, i) => y - F[i]);
      let bestErr = Infinity;
      let bestPred = xs.map(() => 0);
      for (const t of mids) {
        const L = residuals.filter((_, i) => xs[i] <= t);
        const R = residuals.filter((_, i) => xs[i] > t);
        const mL = L.reduce((s, v) => s + v, 0) / L.length;
        const mR = R.reduce((s, v) => s + v, 0) / R.length;
        const sse = residuals.reduce((s, r, i) => s + (r - (xs[i] <= t ? mL : mR)) ** 2, 0);
        if (sse < bestErr) {
          bestErr = sse;
          bestPred = xs.map((x) => (x <= t ? mL : mR));
        }
      }
      F = F.map((v, i) => v + lr * bestPred[i]);
    }

    const finalRes = ys.map((y, i) => y - F[i]);
    const mse = finalRes.reduce((s, r) => s + r * r, 0) / ys.length;

    const fTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'markers', x: xs, y: ys, name: 'Dữ liệu y', marker: { size: 8, color: '#42d3e5' } },
      { type: 'scatter', mode: 'lines', x: xs, y: F, name: `GBDT F_${rounds}(x) (η=${lr})`, line: { color: '#ffd173', width: 3, shape: 'hv' } },
    ];
    const rTraces: PlotTrace[] = [
      { type: 'bar', x: xs.map((x) => x.toFixed(2)), y: finalRes, name: 'Pseudo-residual y − F(x)', marker: { color: '#f28db9' } },
    ];
    return { fitTraces: fTraces, resTraces: rTraces, trainMse: mse };
  }, [rounds, lr]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số vòng Boosting M: <strong>{rounds}</strong>
          <input type="range" min={1} max={25} value={rounds} onChange={(e) => setRounds(+e.target.value)} />
        </label>
        <label>
          Shrinkage / Learning rate η: <strong>{fmt(lr, 2)}</strong>
          <input type="range" min={0.1} max={1.0} step={0.1} value={lr} onChange={(e) => setLr(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={fitTraces} xTitle="x" yTitle="Dự đoán F_M(x)" height={310} />
        <Chart traces={resTraces} xTitle="Điểm mẫu x" yTitle="Phần dư còn lại rᵢ" height={310} layout={{ yaxis: { range: [-1.2, 1.2] } }} />
      </div>
      <p className="sim-caption">
        MSE trên tập huấn luyện sau M = {rounds} cây gốc: <strong>{fmt(trainMse, 5)}</strong>. Mỗi cây mới chỉ khớp phần dư còn sót lại sau các cây trước đó, giúp hạ nhanh Bias theo từng bước cộng dồn.
      </p>
    </div>
  );
}

export function GradSimulationM45({ id }: { id: GradSimulationId }) {
  if (id === 'm4-u1') return <SimGdaLdaQda />;
  if (id === 'm4-u2') return <SimSvmMargin />;
  if (id === 'm4-u3') return <SimRbfKernel />;
  if (id === 'm5-u1') return <SimCurseDim />;
  if (id === 'm5-u2') return <SimCartSplit />;
  if (id === 'm5-u3') return <SimBaggingVariance />;
  if (id === 'm5-u4') return <SimGradientBoosting1D />;
  return null;
}
