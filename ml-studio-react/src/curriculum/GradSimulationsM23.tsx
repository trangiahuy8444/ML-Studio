import { useMemo, useState } from 'react';
import { Chart } from '../components/Chart';
import { solve } from '../lib/domain';
import type { PlotTrace } from '../types';
import type { GradSimulationId } from './types';

const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : '∞');

function sig(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

function softplus(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

function softThresh(z: number, lam: number): number {
  return z > lam ? z - lam : z < -lam ? z + lam : 0;
}

function fitRidge2D(X: number[][], y: number[], lam: number): [number, number] {
  const n = X.length;
  const a00 = X.reduce((s, r) => s + r[0] * r[0], 0) + lam;
  const a01 = X.reduce((s, r) => s + r[0] * r[1], 0);
  const a11 = X.reduce((s, r) => s + r[1] * r[1], 0) + lam;
  const b0 = X.reduce((s, r, i) => s + r[0] * y[i], 0);
  const b1 = X.reduce((s, r, i) => s + r[1] * y[i], 0);
  void n;
  const sol = solve(
    [
      [a00, a01],
      [a01, a11],
    ],
    [b0, b1],
  );
  return [sol[0], sol[1]];
}

/** Unit 2.1: Đa cộng tuyến (Collinearity), Số điều kiện κ(XᵀX) & Độ bất ổn OLS */
function SimOlsCollinearity() {
  const [corr, setCorr] = useState(0.85);
  const [ridgeLam, setRidgeLam] = useState(0.0);

  const { traces, kappa, stdW1, meanW1, orthoErr } = useMemo(() => {
    const x1 = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    const z = [0.6, -0.8, 0.3, -0.4, 0.9, -0.5, -0.2, 0.1];
    const alpha = 1 - corr;
    const X = x1.map((v, i) => [v, corr * v + Math.sqrt(Math.max(1e-8, 1 - corr * corr)) * (0.2 + alpha) * z[i]]);

    // Compute XᵀX eigenvalues analytically for 2x2 symmetric matrix
    const g00 = X.reduce((s, r) => s + r[0] * r[0], 0) + ridgeLam;
    const g01 = X.reduce((s, r) => s + r[0] * r[1], 0);
    const g11 = X.reduce((s, r) => s + r[1] * r[1], 0) + ridgeLam;
    const tr = g00 + g11;
    const det = g00 * g11 - g01 * g01;
    const disc = Math.sqrt(Math.max(0, (g00 - g11) ** 2 + 4 * g01 * g01));
    const lmax = 0.5 * (tr + disc);
    const lmin = Math.max(1e-14, 0.5 * (tr - disc));
    const cond = lmax / lmin;

    // Simulate 16 noisy target draws around true w* = [2.0, 1.0]
    const w1s: number[] = [];
    const w2s: number[] = [];
    let resOrtho = 0;
    for (let m = 0; m < 16; m++) {
      const y = X.map((r, i) => 2.0 * r[0] + 1.0 * r[1] + 0.15 * Math.sin(m * 2.1 + i * 1.7));
      const [w1, w2] = fitRidge2D(X, y, Math.max(1e-10, ridgeLam));
      w1s.push(w1);
      w2s.push(w2);
      if (m === 0) {
        const res = X.map((r, i) => r[0] * w1 + r[1] * w2 - y[i]);
        const o1 = X.reduce((s, r, i) => s + r[0] * res[i], 0);
        const o2 = X.reduce((s, r, i) => s + r[1] * res[i], 0);
        resOrtho = Math.hypot(o1, o2);
      }
    }

    const mw1 = w1s.reduce((s, v) => s + v, 0) / w1s.length;
    const sw1 = Math.sqrt(w1s.reduce((s, v) => s + (v - mw1) ** 2, 0) / w1s.length);

    void det;
    const plotTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: w1s,
        y: w2s,
        name: '16 ước lượng ŵ⁽ᵐ⁾ khi nhiễu nhẹ',
        marker: { size: 9, color: '#42d3e5' },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: [2.0],
        y: [1.0],
        name: 'Tham số thật w* = [2, 1]',
        marker: { size: 13, symbol: 'star', color: '#ffd173' },
      },
    ];

    return { traces: plotTraces, kappa: cond, stdW1: sw1, meanW1: mw1, orthoErr: resOrtho };
  }, [corr, ridgeLam]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ tương quan hai cột ρ: <strong>{fmt(corr, 3)}</strong>
          <input type="range" min={0} max={0.998} step={0.002} value={corr} onChange={(e) => setCorr(+e.target.value)} />
        </label>
        <label>
          Phạt ổn định Ridge λ: <strong>{fmt(ridgeLam, 2)}</strong>
          <input type="range" min={0} max={2.0} step={0.05} value={ridgeLam} onChange={(e) => setRidgeLam(+e.target.value)} />
        </label>
      </div>

      <Chart
        traces={traces}
        xTitle="Trọng số ŵ₁"
        yTitle="Trọng số ŵ₂"
        height={320}
        layout={{ xaxis: { range: [-4, 8] }, yaxis: { range: [-5, 7] } }}
      />

      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Số điều kiện κ(XᵀX + λI)</span>
          <strong style={{ color: kappa > 500 ? '#f28db9' : '#84e0a8' }}>{kappa > 1e6 ? kappa.toExponential(1) : fmt(kappa, 1)}</strong>
        </div>
        <div className="metric-card">
          <span>Trung bình ŵ₁ / Độ lệch chuẩn σ(ŵ₁)</span>
          <strong style={{ color: '#42d3e5' }}>
            {fmt(meanW1, 2)} ± {fmt(stdW1, 3)}
          </strong>
        </div>
        <div className="metric-card">
          <span>Kiểm tra trực giao ||Xᵀr||₂ (λ=0)</span>
          <strong style={{ color: '#ffd173' }}>{orthoErr.toExponential(1)}</strong>
        </div>
      </div>
      <p className="sim-caption">
        Khi ρ → 0.998 và λ = 0, hai đặc trưng gần như cộng tuyến làm κ(XᵀX) tăng vọt và các ước lượng ŵ trượt dọc theo đường w₁ + w₂ ≈ const với phương sai rất lớn. Thêm một lượng nhỏ λ ổn định ngay phân tán hệ số.
      </p>
    </div>
  );
}

/** Unit 2.2: Regularization Path — Ridge (L2) vs Lasso (L1) Coordinate Descent */
function SimRegularizationPath() {
  const [logLam, setLogLam] = useState(-1.2);
  const [mode, setMode] = useState<'lasso' | 'ridge'>('lasso');

  const { pathTraces, barTraces, dfRidge, nonzeroCount } = useMemo(() => {
    // 4 standardized features on n=24 synthetic observations; true signal is w = [2.2, -1.3, 0, 0]
    const n = 24;
    const X = Array.from({ length: n }, (_, i) => {
      const a = Math.sin(i * 0.7);
      const b = Math.cos(i * 1.1);
      const c = 0.8 * a + 0.35 * Math.sin(i * 2.3);
      const d = Math.sin(i * 3.7);
      return [a, b, c, d];
    });
    const y = X.map((r, i) => 2.2 * r[0] - 1.3 * r[1] + 0.15 * Math.sin(i * 4.1));

    // Singular values of X (rough Gram eigenvalues for df)
    const gridLog = Array.from({ length: 25 }, (_, idx) => -3 + (idx / 24) * 4);
    const names = ['w₁ (Tín hiệu mạnh)', 'w₂ (Tín hiệu âm)', 'w₃ (Tương quan x₁)', 'w₄ (Nhiễu)'];
    const colors = ['#42d3e5', '#f28db9', '#ffd173', '#9fb2ca'];

    const solveAt = (lam: number, kind: 'lasso' | 'ridge') => {
      if (kind === 'ridge') {
        const A = Array.from({ length: 4 }, (_, r) =>
          Array.from({ length: 4 }, (_, c) => X.reduce((s, row) => s + row[r] * row[c], 0) + (r === c ? lam * n : 0)),
        );
        const rhs = Array.from({ length: 4 }, (_, r) => X.reduce((s, row, i) => s + row[r] * y[i], 0));
        return solve(A, rhs);
      }
      const w = [0, 0, 0, 0];
      for (let sweep = 0; sweep < 80; sweep++) {
        for (let j = 0; j < 4; j++) {
          let rho = 0;
          let zj = 0;
          for (let i = 0; i < n; i++) {
            let predNoJ = 0;
            for (let k = 0; k < 4; k++) if (k !== j) predNoJ += X[i][k] * w[k];
            rho += X[i][j] * (y[i] - predNoJ);
            zj += X[i][j] * X[i][j];
          }
          w[j] = zj > 0 ? softThresh(rho / n, lam) / (zj / n) : 0;
        }
      }
      return w;
    };

    const paths = [0, 1, 2, 3].map(() => [] as number[]);
    for (const lg of gridLog) {
      const w = solveAt(Math.pow(10, lg), mode);
      for (let j = 0; j < 4; j++) paths[j].push(w[j]);
    }

    const curLam = Math.pow(10, logLam);
    const curW = solveAt(curLam, mode);

    // Effective degrees of freedom of Ridge: tr(X(XᵀX + n*lam*I)^-1 Xᵀ)
    const Acur = Array.from({ length: 4 }, (_, r) =>
      Array.from({ length: 4 }, (_, c) => X.reduce((s, row) => s + row[r] * row[c], 0) + (r === c ? curLam * n : 0)),
    );
    let df = 0;
    for (let j = 0; j < 4; j++) {
      const ej = [0, 0, 0, 0];
      ej[j] = 1;
      const invCol = solve(Acur, ej);
      const gramCol = Array.from({ length: 4 }, (_, r) => X.reduce((s, row) => s + row[r] * row[j], 0));
      df += gramCol.reduce((s, v, k) => s + v * invCol[k], 0);
    }

    const pTraces: PlotTrace[] = paths.map((ys, j) => ({
      type: 'scatter',
      mode: 'lines',
      x: gridLog,
      y: ys,
      name: names[j],
      line: { color: colors[j], width: 2.5 },
    }));
    pTraces.push({
      type: 'scatter',
      mode: 'lines',
      x: [logLam, logLam],
      y: [-2.0, 2.5],
      name: 'λ đang chọn',
      line: { color: '#e6eef8', width: 1.5, dash: 'dot' },
    });

    const bTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: ['w₁', 'w₂', 'w₃', 'w₄'],
        y: curW,
        marker: { color: colors },
        text: curW.map((v) => fmt(v, 3)),
        textposition: 'auto',
      },
    ];

    return {
      pathTraces: pTraces,
      barTraces: bTraces,
      dfRidge: df,
      nonzeroCount: curW.filter((v) => Math.abs(v) > 1e-6).length,
    };
  }, [logLam, mode]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Phương pháp Regularization
          <select value={mode} onChange={(e) => setMode(e.target.value as 'lasso' | 'ridge')}>
            <option value="lasso">Lasso (L1 · Coordinate Descent)</option>
            <option value="ridge">Ridge (L2 · Closed Form)</option>
          </select>
        </label>
        <label>
          Cường độ phạt log₁₀(λ): <strong>{fmt(logLam, 2)}</strong> (λ = {Math.pow(10, logLam).toFixed(4)})
          <input type="range" min={-3} max={1} step={0.1} value={logLam} onChange={(e) => setLogLam(+e.target.value)} />
        </label>
      </div>

      <div className="grad-sim-grid">
        <Chart
          traces={pathTraces}
          xTitle="log₁₀(λ)"
          yTitle="Giá trị trọng số wⱼ"
          height={310}
          layout={{ yaxis: { range: [-1.8, 2.5] } }}
        />
        <Chart
          traces={barTraces}
          xTitle="Đặc trưng"
          yTitle="Hệ số tại λ hiện tại"
          height={310}
          layout={{ yaxis: { range: [-1.8, 2.5] } }}
        />
      </div>

      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Số trọng số khác 0 (Sparsity)</span>
          <strong style={{ color: '#42d3e5' }}>{nonzeroCount} / 4</strong>
        </div>
        <div className="metric-card">
          <span>Bậc tự do hiệu dụng Ridge df(λ)</span>
          <strong style={{ color: '#ffd173' }}>{fmt(dfRidge, 2)} / 4.00</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 2.3: 2D Logistic Regression Probability Field & Decision Threshold */
function SimLogisticBoundary() {
  const [w1, setW1] = useState(2.2);
  const [w2, setW2] = useState(1.6);
  const [b, setB] = useState(-0.3);
  const [thresh, setThresh] = useState(0.5);

  const data = useMemo(
    () => [
      { x1: -1.6, x2: -0.9, y: 0 },
      { x1: -1.2, x2: 0.4, y: 0 },
      { x1: -0.7, x2: -1.1, y: 0 },
      { x1: -0.4, x2: 0.2, y: 0 },
      { x1: 0.1, x2: -0.5, y: 0 },
      { x1: 0.3, x2: 1.2, y: 1 },
      { x1: 0.8, x2: 0.4, y: 1 },
      { x1: 1.2, x2: -0.2, y: 1 },
      { x1: 1.5, x2: 1.1, y: 1 },
      { x1: -0.1, x2: 1.4, y: 1 },
    ],
    [],
  );

  const fitMle = () => {
    let tw1 = w1;
    let tw2 = w2;
    let tb = b;
    for (let s = 0; s < 12; s++) {
      const Xa = data.map((d) => [d.x1, d.x2, 1]);
      const p = Xa.map((r) => sig(r[0] * tw1 + r[1] * tw2 + tb));
      const g = [0, 0, 0];
      const H = [
        [1e-3, 0, 0],
        [0, 1e-3, 0],
        [0, 0, 1e-3],
      ];
      for (let i = 0; i < data.length; i++) {
        const e = (p[i] - data[i].y) / data.length;
        const wt = (p[i] * (1 - p[i])) / data.length;
        for (let r = 0; r < 3; r++) {
          g[r] += e * Xa[i][r];
          for (let c = 0; c < 3; c++) H[r][c] += wt * Xa[i][r] * Xa[i][c];
        }
      }
      const step = solve(H, g);
      tw1 -= step[0];
      tw2 -= step[1];
      tb -= step[2];
    }
    setW1(+tw1.toFixed(2));
    setW2(+tw2.toFixed(2));
    setB(+tb.toFixed(2));
  };

  const { traces, bce, prec, rec, acc } = useMemo(() => {
    const grid = Array.from({ length: 28 }, (_, i) => -2.0 + (i / 27) * 4.0);
    const Z = grid.map((yVal) => grid.map((xVal) => sig(w1 * xVal + w2 * yVal + b)));

    let lossSum = 0;
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    for (const d of data) {
      const z = w1 * d.x1 + w2 * d.x2 + b;
      lossSum += softplus(z) - d.y * z;
      const pred = sig(z) >= thresh ? 1 : 0;
      if (pred === 1 && d.y === 1) tp++;
      else if (pred === 1 && d.y === 0) fp++;
      else if (pred === 0 && d.y === 0) tn++;
      else fn++;
    }

    const logitT = Math.log(thresh / (1 - thresh));
    const xsLine = [-2.0, 2.0];
    const ysLine =
      Math.abs(w2) > 1e-4 ? xsLine.map((x) => (logitT - b - w1 * x) / w2) : [0, 0];

    const c0 = data.filter((d) => d.y === 0);
    const c1 = data.filter((d) => d.y === 1);

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
        type: 'scatter',
        mode: 'markers',
        x: c0.map((d) => d.x1),
        y: c0.map((d) => d.x2),
        name: 'Lớp y = 0',
        marker: { size: 11, color: '#42d3e5', line: { color: '#fff', width: 1.5 } },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: c1.map((d) => d.x1),
        y: c1.map((d) => d.x2),
        name: 'Lớp y = 1',
        marker: { size: 11, color: '#f28db9', line: { color: '#fff', width: 1.5 } },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: xsLine,
        y: ysLine,
        name: `Ngưỡng P = ${thresh.toFixed(2)}`,
        line: { color: '#ffd173', width: 2.5, dash: 'dash' },
      },
    ];

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    return {
      traces: pTraces,
      bce: lossSum / data.length,
      prec: precision,
      rec: recall,
      acc: (tp + tn) / data.length,
    };
  }, [w1, w2, b, thresh, data]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          w₁: <strong>{fmt(w1, 2)}</strong>
          <input type="range" min={-4} max={4} step={0.1} value={w1} onChange={(e) => setW1(+e.target.value)} />
        </label>
        <label>
          w₂: <strong>{fmt(w2, 2)}</strong>
          <input type="range" min={-4} max={4} step={0.1} value={w2} onChange={(e) => setW2(+e.target.value)} />
        </label>
        <label>
          Bias b: <strong>{fmt(b, 2)}</strong>
          <input type="range" min={-3} max={3} step={0.1} value={b} onChange={(e) => setB(+e.target.value)} />
        </label>
        <label>
          Ngưỡng phân loại τ: <strong>{fmt(thresh, 2)}</strong>
          <input type="range" min={0.1} max={0.9} step={0.05} value={thresh} onChange={(e) => setThresh(+e.target.value)} />
        </label>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <button className="btn-outline" onClick={fitMle}>
          Khớp nghiệm Newton MLE tối ưu
        </button>
      </div>

      <Chart
        traces={traces}
        xTitle="Đặc trưng x₁"
        yTitle="Đặc trưng x₂"
        height={340}
        layout={{ xaxis: { range: [-2, 2] }, yaxis: { range: [-2, 2] } }}
      />

      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Binary Cross-Entropy (không phụ thuộc τ)</span>
          <strong style={{ color: '#42d3e5' }}>{fmt(bce, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Accuracy</span>
          <strong style={{ color: '#84e0a8' }}>{fmt(acc * 100, 1)}%</strong>
        </div>
        <div className="metric-card">
          <span>Precision / Recall</span>
          <strong style={{ color: '#ffd173' }}>
            {fmt(prec, 2)} / {fmt(rec, 2)}
          </strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 2.4: Poisson GLM vs OLS trên Dữ liệu đếm */
function SimPoissonGlm() {
  const [slope, setSlope] = useState(0.55);
  const [intercept, setIntercept] = useState(0.35);

  const countData = useMemo(
    () => [
      [-1.8, 0],
      [-1.4, 1],
      [-1.0, 0],
      [-0.5, 1],
      [0.0, 2],
      [0.5, 2],
      [1.0, 3],
      [1.4, 3],
      [1.8, 5],
      [2.2, 4],
      [2.6, 7],
      [3.0, 8],
    ],
    [],
  );

  const fitPoissonNewton = () => {
    let w = slope;
    let b = intercept;
    const n = countData.length;
    for (let it = 0; it < 15; it++) {
      let g0 = 0;
      let g1 = 0;
      let h00 = 1e-6;
      let h01 = 0;
      let h11 = 1e-6;
      for (const [x, y] of countData) {
        const mu = Math.exp(Math.max(-10, Math.min(10, w * x + b)));
        g0 += (mu - y) * x;
        g1 += mu - y;
        h00 += mu * x * x;
        h01 += mu * x;
        h11 += mu;
      }
      const step = solve(
        [
          [h00 / n, h01 / n],
          [h01 / n, h11 / n],
        ],
        [g0 / n, g1 / n],
      );
      w -= step[0];
      b -= step[1];
    }
    setSlope(+w.toFixed(2));
    setIntercept(+b.toFixed(2));
  };

  const { traces, poissonNll, negOls } = useMemo(() => {
    const xs = countData.map((r) => r[0]);
    const ys = countData.map((r) => r[1]);
    // Fit exact OLS
    const n = xs.length;
    const mx = xs.reduce((s, v) => s + v, 0) / n;
    const my = ys.reduce((s, v) => s + v, 0) / n;
    const wOls = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
    const bOls = my - wOls * mx;

    const grid = Array.from({ length: 61 }, (_, i) => -2.2 + (i / 60) * 5.6);
    const yPoisson = grid.map((x) => Math.exp(slope * x + intercept));
    const yOls = grid.map((x) => wOls * x + bOls);

    const nll = countData.reduce((s, [x, y]) => {
      const eta = slope * x + intercept;
      return s + Math.exp(eta) - y * eta;
    }, 0) / n;

    const plotTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: xs,
        y: ys,
        name: 'Số lần đếm thực tế y ∈ {0,1,2,...}',
        marker: { size: 9, color: '#84e0a8' },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: grid,
        y: yPoisson,
        name: 'Poisson GLM: μ = exp(wx + b)',
        line: { color: '#42d3e5', width: 3 },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: grid,
        y: yOls,
        name: 'OLS tuyến tính (bị âm khi x < −0.6)',
        line: { color: '#f28db9', width: 2, dash: 'dash' },
      },
    ];

    return { traces: plotTraces, poissonNll: nll, negOls: yOls[0] };
  }, [slope, intercept, countData]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Hệ số góc Poisson w: <strong>{fmt(slope, 2)}</strong>
          <input type="range" min={-0.2} max={1.0} step={0.05} value={slope} onChange={(e) => setSlope(+e.target.value)} />
        </label>
        <label>
          Bias b: <strong>{fmt(intercept, 2)}</strong>
          <input type="range" min={-0.5} max={1.2} step={0.05} value={intercept} onChange={(e) => setIntercept(+e.target.value)} />
        </label>
      </div>
      <div>
        <button className="btn-outline" onClick={fitPoissonNewton}>
          Khớp Poisson MLE tối ưu
        </button>
      </div>
      <Chart traces={traces} xTitle="Đặc trưng x" yTitle="Kỳ vọng số lần đếm E[y|x]" height={320} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Poisson Negative Log-Likelihood</span>
          <strong style={{ color: '#42d3e5' }}>{fmt(poissonNll, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Dự đoán OLS tại x = −2.2</span>
          <strong style={{ color: '#f28db9' }}>{fmt(negOls, 2)} (phi lý vì &lt; 0)</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 3.1: Condition Number κ = λ_max / λ_min & GD Zig-Zag Trajectory */
function SimConditionNumberContour() {
  const [kappa, setKappa] = useState(12);
  const [etaRatio, setEtaRatio] = useState(1.2); // eta = etaRatio / L

  const { traces, diverged, endLoss } = useMemo(() => {
    // f(w) = 0.5 * (kappa * w0^2 + w1^2) -> H = diag(kappa, 1), L = kappa, mu = 1
    const L = kappa;
    const eta = etaRatio / L;
    const grid = Array.from({ length: 35 }, (_, i) => -2.4 + (i / 34) * 4.8);
    const Z = grid.map((y) => grid.map((x) => 0.5 * (kappa * x * x + y * y)));

    const xs = [1.8];
    const ys = [1.8];
    let w0 = 1.8;
    let w1 = 1.8;
    for (let t = 0; t < 25; t++) {
      w0 -= eta * kappa * w0;
      w1 -= eta * 1.0 * w1;
      if (Math.abs(w0) > 50 || Math.abs(w1) > 50) break;
      xs.push(w0);
      ys.push(w1);
    }

    const pTraces: PlotTrace[] = [
      {
        type: 'contour',
        x: grid,
        y: grid,
        z: Z,
        colorscale: 'Viridis',
        showscale: false,
        contours: { coloring: 'lines' },
        name: 'Đường đồng mức f(w)',
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: xs,
        y: ys,
        name: 'Quỹ đạo Gradient Descent (25 bước)',
        line: { color: '#f28db9', width: 2.5 },
        marker: { size: 6, color: '#ffd173' },
      },
    ];
    const finLoss = 0.5 * (kappa * w0 * w0 + w1 * w1);
    return { traces: pTraces, diverged: etaRatio >= 2.0, endLoss: finLoss };
  }, [kappa, etaRatio]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số điều kiện κ = L/μ: <strong>{kappa}</strong>
          <input type="range" min={1} max={50} value={kappa} onChange={(e) => setKappa(+e.target.value)} />
        </label>
        <label>
          Tỷ lệ bước học η · L: <strong>{fmt(etaRatio, 2)}</strong> (Ngưỡng ổn định η · L &lt; 2.0)
          <input type="range" min={0.2} max={2.15} step={0.05} value={etaRatio} onChange={(e) => setEtaRatio(+e.target.value)} />
        </label>
      </div>
      <Chart
        traces={traces}
        xTitle="w₁ (Trục dốc L = κ)"
        yTitle="w₂ (Trục phẳng μ = 1)"
        height={340}
        layout={{ xaxis: { range: [-2.4, 2.4] }, yaxis: { range: [-2.4, 2.4], scaleanchor: 'x' } }}
      />
      <p className="sim-caption">
        {diverged
          ? '⚠️ PHÂN KỲ: Vì η ≥ 2/L, bước nhảy vượt qua thành thung lũng làm sai số nổ tăng theo cấp số nhân.'
          : `Giá trị f(w₂₅) = ${fmt(endLoss, 4)}. Khi κ = 1 (hình tròn), GD đi thẳng vào tâm chỉ trong 1 bước với η = 1/L; khi κ lớn, đường đi zic-zắc mất rất nhiều bước dọc trục w₂.`}
      </p>
    </div>
  );
}

/** Unit 3.2: GD Convergence Bounds & Armijo Backtracking */
function SimGdStepSize() {
  const [kappa, setKappa] = useState(20);
  const [etaMult, setEtaMult] = useState(1.0); // eta = etaMult / L

  const { traces, theoEnd, empEnd, armijoEnd } = useMemo(() => {
    const L = kappa;
    const mu = 1;
    const eta = etaMult / L;
    const steps = 35;

    // GD from (1, 1) on 0.5*(kappa*w0^2 + w1^2)
    let w0 = 1.0;
    let w1 = 1.0;
    const emp: number[] = [];
    const bound: number[] = [];
    const f0 = 0.5 * (kappa + 1);
    const rhoOpt = (kappa - mu) / (kappa + mu);

    for (let t = 0; t <= steps; t++) {
      const fVal = 0.5 * (kappa * w0 * w0 + w1 * w1);
      emp.push(Math.log10(Math.max(1e-16, Math.min(1e12, fVal))));
      bound.push(Math.log10(Math.max(1e-16, f0 * Math.pow(rhoOpt, 2 * t))));
      w0 -= eta * kappa * w0;
      w1 -= eta * w1;
    }

    // Armijo backtracking curve
    let a0 = 1.0;
    let a1 = 1.0;
    const arm: number[] = [];
    for (let t = 0; t <= steps; t++) {
      const fw = 0.5 * (kappa * a0 * a0 + a1 * a1);
      arm.push(Math.log10(Math.max(1e-16, fw)));
      const g0 = kappa * a0;
      const g1 = a1;
      const gn2 = g0 * g0 + g1 * g1;
      let stepEta = 1.0;
      while (stepEta > 1e-12 && 0.5 * (kappa * (a0 - stepEta * g0) ** 2 + (a1 - stepEta * g1) ** 2) > fw - 1e-4 * stepEta * gn2) {
        stepEta *= 0.5;
      }
      a0 -= stepEta * g0;
      a1 -= stepEta * g1;
    }

    const xs = Array.from({ length: steps + 1 }, (_, i) => i);
    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: xs, y: emp, name: `GD bước cố định η=${fmt(etaMult, 2)}/L`, line: { color: '#42d3e5', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: xs, y: bound, name: 'Chận lý thuyết (κ−1)/(κ+1) ở η*=2/(L+μ)', line: { color: '#ffd173', width: 2, dash: 'dash' } },
      { type: 'scatter', mode: 'lines', x: xs, y: arm, name: 'GD + Armijo Backtracking (tự dò L)', line: { color: '#84e0a8', width: 2.5 } },
    ];
    return { traces: pTraces, theoEnd: bound[steps], empEnd: emp[steps], armijoEnd: arm[steps] };
  }, [kappa, etaMult]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số điều kiện κ = L/μ: <strong>{kappa}</strong>
          <input type="range" min={2} max={60} value={kappa} onChange={(e) => setKappa(+e.target.value)} />
        </label>
        <label>
          Hệ số bước học η · L: <strong>{fmt(etaMult, 2)}</strong>
          <input type="range" min={0.1} max={2.1} step={0.05} value={etaMult} onChange={(e) => setEtaMult(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Số vòng lặp t" yTitle="log₁₀ f(wₜ)" height={320} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>log₁₀ f(w₃₅) GD cố định</span>
          <strong style={{ color: etaMult >= 2 ? '#f28db9' : '#42d3e5' }}>{fmt(empEnd, 2)}</strong>
        </div>
        <div className="metric-card">
          <span>Chận tối ưu lý thuyết</span>
          <strong style={{ color: '#ffd173' }}>{fmt(theoEnd, 2)}</strong>
        </div>
        <div className="metric-card">
          <span>log₁₀ f(w₃₅) Armijo Backtracking</span>
          <strong style={{ color: '#84e0a8' }}>{fmt(armijoEnd, 2)}</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 3.3: Optimizer Race — GD vs Momentum vs Nesterov vs Adam */
function SimOptimizerRace() {
  const [lrGd, setLrGd] = useState(0.032);
  const [betaMom, setBetaMom] = useState(0.85);
  const [lrAdam, setLrAdam] = useState(0.35);

  const { contourTraces, lossTraces } = useMemo(() => {
    // Ravine: f(x, y) = 25*x^2 + 0.5*y^2 -> L = 50, mu = 1, start = (-1.6, 2.8)
    const f = (x: number, y: number) => 25 * x * x + 0.5 * y * y;
    const grad = (x: number, y: number): [number, number] => [50 * x, 1.0 * y];
    const start: [number, number] = [-1.6, 2.8];
    const steps = 40;

    // 1. GD
    const gdX = [start[0]];
    const gdY = [start[1]];
    const gdL = [Math.log10(f(...start))];
    let gx = start[0];
    let gy = start[1];

    // 2. Heavy-ball Momentum
    const mX = [start[0]];
    const mY = [start[1]];
    const mL = [Math.log10(f(...start))];
    let mx = start[0];
    let my = start[1];
    let vmx = 0;
    let vmy = 0;

    // 3. Nesterov
    const nX = [start[0]];
    const nY = [start[1]];
    const nL = [Math.log10(f(...start))];
    let nx = start[0];
    let ny = start[1];
    let vnx = 0;
    let vny = 0;

    // 4. Adam
    const aX = [start[0]];
    const aY = [start[1]];
    const aL = [Math.log10(f(...start))];
    let ax = start[0];
    let ay = start[1];
    let amx = 0;
    let amy = 0;
    let avx = 0;
    let avy = 0;

    for (let t = 1; t <= steps; t++) {
      // GD
      const [dgx, dgy] = grad(gx, gy);
      gx = Math.max(-5, Math.min(5, gx - lrGd * dgx));
      gy = Math.max(-5, Math.min(5, gy - lrGd * dgy));
      gdX.push(gx);
      gdY.push(gy);
      gdL.push(Math.log10(Math.max(1e-12, f(gx, gy))));

      // Momentum
      const [dmx, dmy] = grad(mx, my);
      vmx = betaMom * vmx + dmx;
      vmy = betaMom * vmy + dmy;
      mx = Math.max(-5, Math.min(5, mx - lrGd * 0.5 * vmx));
      my = Math.max(-5, Math.min(5, my - lrGd * 0.5 * vmy));
      mX.push(mx);
      mY.push(my);
      mL.push(Math.log10(Math.max(1e-12, f(mx, my))));

      // Nesterov look-ahead
      const lx = nx - lrGd * 0.5 * betaMom * vnx;
      const ly = ny - lrGd * 0.5 * betaMom * vny;
      const [dnx, dny] = grad(lx, ly);
      vnx = betaMom * vnx + dnx;
      vny = betaMom * vny + dny;
      nx = Math.max(-5, Math.min(5, nx - lrGd * 0.5 * vnx));
      ny = Math.max(-5, Math.min(5, ny - lrGd * 0.5 * vny));
      nX.push(nx);
      nY.push(ny);
      nL.push(Math.log10(Math.max(1e-12, f(nx, ny))));

      // Adam
      const [dax, day] = grad(ax, ay);
      const b1 = 0.9;
      const b2 = 0.999;
      amx = b1 * amx + (1 - b1) * dax;
      amy = b1 * amy + (1 - b1) * day;
      avx = b2 * avx + (1 - b2) * dax * dax;
      avy = b2 * avy + (1 - b2) * day * day;
      const mhx = amx / (1 - Math.pow(b1, t));
      const mhy = amy / (1 - Math.pow(b1, t));
      const vhx = avx / (1 - Math.pow(b2, t));
      const vhy = avy / (1 - Math.pow(b2, t));
      ax -= (lrAdam * mhx) / (Math.sqrt(vhx) + 1e-8);
      ay -= (lrAdam * mhy) / (Math.sqrt(vhy) + 1e-8);
      aX.push(ax);
      aY.push(ay);
      aL.push(Math.log10(Math.max(1e-12, f(ax, ay))));
    }

    const xs = Array.from({ length: 30 }, (_, i) => -2.2 + (i / 29) * 4.4);
    const ys = Array.from({ length: 30 }, (_, i) => -1.5 + (i / 29) * 4.8);
    const Z = ys.map((y) => xs.map((x) => f(x, y)));

    const cTraces: PlotTrace[] = [
      { type: 'contour', x: xs, y: ys, z: Z, colorscale: 'Viridis', showscale: false, contours: { coloring: 'lines' }, name: 'f(w)' },
      { type: 'scatter', mode: 'lines', x: gdX, y: gdY, name: 'GD thường', line: { color: '#9fb2ca', width: 2 } },
      { type: 'scatter', mode: 'lines', x: mX, y: mY, name: 'Momentum', line: { color: '#ffd173', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: nX, y: nY, name: 'Nesterov (NAG)', line: { color: '#84e0a8', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: aX, y: aY, name: 'Adam', line: { color: '#f28db9', width: 2.5 } },
    ];

    const iters = Array.from({ length: steps + 1 }, (_, i) => i);
    const lTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: iters, y: gdL, name: 'GD thường', line: { color: '#9fb2ca', width: 2 } },
      { type: 'scatter', mode: 'lines', x: iters, y: mL, name: 'Momentum', line: { color: '#ffd173', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: iters, y: nL, name: 'Nesterov (NAG)', line: { color: '#84e0a8', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: iters, y: aL, name: 'Adam', line: { color: '#f28db9', width: 2.5 } },
    ];

    return { contourTraces: cTraces, lossTraces: lTraces };
  }, [lrGd, betaMom, lrAdam]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Learning rate GD / Momentum: <strong>{fmt(lrGd, 3)}</strong>
          <input type="range" min={0.005} max={0.039} step={0.002} value={lrGd} onChange={(e) => setLrGd(+e.target.value)} />
        </label>
        <label>
          Hệ số quán tính β: <strong>{fmt(betaMom, 2)}</strong>
          <input type="range" min={0.5} max={0.96} step={0.02} value={betaMom} onChange={(e) => setBetaMom(+e.target.value)} />
        </label>
        <label>
          Learning rate Adam: <strong>{fmt(lrAdam, 2)}</strong>
          <input type="range" min={0.05} max={0.6} step={0.05} value={lrAdam} onChange={(e) => setLrAdam(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={contourTraces} xTitle="w₁ (trục hẹp)" yTitle="w₂ (trục dài)" height={320} />
        <Chart traces={lossTraces} xTitle="Bước t" yTitle="log₁₀ f(wₜ)" height={320} />
      </div>
    </div>
  );
}

/** Unit 3.4: First-Order GD vs Second-Order Newton-Raphson on Ill-Conditioned Logistic Regression */
function SimNewtonVsGd() {
  const [scaleRatio, setScaleRatio] = useState(25);
  const [gdLr, setGdLr] = useState(0.08);

  const { traces, nwEnd, gdEnd } = useMemo(() => {
    // 20 observations where feature 1 is unscaled vs feature 2 scaled down by 1/scaleRatio
    const X = Array.from({ length: 20 }, (_, i) => {
      const a = -2 + (i / 19) * 4;
      const b = Math.sin(i * 1.9);
      return [a * scaleRatio, b / Math.max(1, scaleRatio * 0.2), 1];
    });
    const y = X.map((_, i) => (i >= 10 || (i >= 8 && Math.sin(i * 1.9) > 0) ? 1 : 0));
    const reg = 1e-3;

    const evalLog = (th: number[]) => {
      const g = [0, 0, 0];
      const H = [
        [reg, 0, 0],
        [0, reg, 0],
        [0, 0, reg],
      ];
      for (let i = 0; i < X.length; i++) {
        const z = Math.max(-30, Math.min(30, X[i][0] * th[0] + X[i][1] * th[1] + th[2]));
        const p = sig(z);
        const e = (p - y[i]) / X.length;
        const wt = (p * (1 - p)) / X.length;
        for (let r = 0; r < 3; r++) {
          g[r] += e * X[i][r];
          for (let c = 0; c < 3; c++) H[r][c] += wt * X[i][r] * X[i][c];
        }
      }
      for (let r = 0; r < 3; r++) g[r] += reg * th[r];
      return { g, H, gNorm: Math.hypot(...g) };
    };

    const maxIter = 12;
    const nwNorms: number[] = [];
    const gdNorms: number[] = [];
    let thN = [0, 0, 0];
    let thG = [0, 0, 0];

    for (let k = 0; k <= maxIter; k++) {
      const nInfo = evalLog(thN);
      const gInfo = evalLog(thG);
      nwNorms.push(Math.log10(Math.max(1e-15, nInfo.gNorm)));
      gdNorms.push(Math.log10(Math.max(1e-15, Math.min(1e6, gInfo.gNorm))));
      const delta = solve(nInfo.H, nInfo.g);
      thN = [thN[0] - delta[0], thN[1] - delta[1], thN[2] - delta[2]];
      const safeStep = gdLr / Math.max(1, scaleRatio * 0.5);
      thG = [thG[0] - safeStep * gInfo.g[0], thG[1] - safeStep * gInfo.g[1], thG[2] - safeStep * gInfo.g[2]];
    }

    const xs = Array.from({ length: maxIter + 1 }, (_, i) => i);
    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: xs, y: nwNorms, name: 'Newton–Raphson (Bậc 2 · Quadratic)', line: { color: '#42d3e5', width: 3 } },
      { type: 'scatter', mode: 'lines+markers', x: xs, y: gdNorms, name: 'Gradient Descent (Bậc 1)', line: { color: '#f28db9', width: 2.5 } },
    ];
    return { traces: pTraces, nwEnd: nwNorms[maxIter], gdEnd: gdNorms[maxIter] };
  }, [scaleRatio, gdLr]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ chênh thang đo đặc trưng: <strong>{scaleRatio}×</strong>
          <input type="range" min={1} max={60} value={scaleRatio} onChange={(e) => setScaleRatio(+e.target.value)} />
        </label>
        <label>
          Bước học cơ sở của GD: <strong>{fmt(gdLr, 2)}</strong>
          <input type="range" min={0.02} max={0.3} step={0.02} value={gdLr} onChange={(e) => setGdLr(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Số vòng lặp t" yTitle="log₁₀ ||∇J(θₜ)||₂" height={320} />
      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>log₁₀ ||∇J|| sau 12 bước Newton</span>
          <strong style={{ color: '#42d3e5' }}>{fmt(nwEnd, 2)} (≈ 10^{Math.round(nwEnd)})</strong>
        </div>
        <div className="metric-card">
          <span>log₁₀ ||∇J|| sau 12 bước GD</span>
          <strong style={{ color: '#f28db9' }}>{fmt(gdEnd, 2)}</strong>
        </div>
      </div>
      <p className="sim-caption">
        Nhờ nhân với nghịch đảo Hessian H⁻¹, quỹ đạo Newton hoàn toàn bất biến trước việc thay đổi thang đo đặc trưng từ 1× lên 60× — đạt chuẩn hội tụ máy chỉ sau ~5–7 bước trong khi GD bị kẹt nặng khi thang đo chênh lệch.
      </p>
    </div>
  );
}

export function GradSimulationM23({ id }: { id: GradSimulationId }) {
  if (id === 'm2-u1') return <SimOlsCollinearity />;
  if (id === 'm2-u2') return <SimRegularizationPath />;
  if (id === 'm2-u3') return <SimLogisticBoundary />;
  if (id === 'm2-u4') return <SimPoissonGlm />;
  if (id === 'm3-u1') return <SimConditionNumberContour />;
  if (id === 'm3-u2') return <SimGdStepSize />;
  if (id === 'm3-u3') return <SimOptimizerRace />;
  if (id === 'm3-u4') return <SimNewtonVsGd />;
  return null;
}
