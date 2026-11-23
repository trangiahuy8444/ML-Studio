import { useMemo, useState } from 'react';
import { Chart } from '../components/Chart';
import { solve } from '../lib/domain';
import type { PlotTrace } from '../types';
import type { GradSimulationId } from './types';

const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : 'NaN');

// Log-gamma via Lanczos approximation for Beta normalization
function logGamma(z: number): number {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betaPdf(theta: number, a: number, b: number): number {
  if (theta <= 0 || theta >= 1) return a === 1 && b === 1 ? 1 : 0;
  const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
  return Math.exp((a - 1) * Math.log(theta) + (b - 1) * Math.log(1 - theta) - logB);
}

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function normalPair(rnd: () => number): [number, number] {
  const u1 = Math.max(1e-12, rnd());
  const u2 = rnd();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

/** Unit 1.1: Phép chiếu trực giao, Tích vô hướng Xw + b & Kiểm chứng Gradient ma trận */
function SimLinearProjectionGrad() {
  const [w1, setW1] = useState(1.2);
  const [w2, setW2] = useState(0.8);
  const [b, setB] = useState(-0.4);
  const [focusRow, setFocusRow] = useState(0);

  const X = useMemo(
    () => [
      [1.0, 2.0],
      [2.5, 0.5],
      [-1.0, 2.2],
      [1.8, 1.5],
    ],
    [],
  );
  const y = useMemo(() => [2.2, 2.8, 0.6, 3.0], []);

  const stats = useMemo(() => {
    const w = [w1, w2];
    const yhat = X.map((row) => row[0] * w[0] + row[1] * w[1] + b);
    const err = yhat.map((p, i) => p - y[i]);
    const loss = err.reduce((s, e) => s + e * e, 0) / y.length;

    // Analytical gradient w.r.t. w and b: (2/n) X^T (Xw + b - y)
    const gw1 = (2 / y.length) * err.reduce((s, e, i) => s + e * X[i][0], 0);
    const gw2 = (2 / y.length) * err.reduce((s, e, i) => s + e * X[i][1], 0);
    const gb = (2 / y.length) * err.reduce((s, e) => s + e, 0);

    // Central finite differences
    const eps = 1e-6;
    const lossAt = (tw1: number, tw2: number, tb: number) =>
      X.reduce((s, row, i) => {
        const d = row[0] * tw1 + row[1] * tw2 + tb - y[i];
        return s + d * d;
      }, 0) / y.length;
    const numGw1 = (lossAt(w1 + eps, w2, b) - lossAt(w1 - eps, w2, b)) / (2 * eps);
    const numGw2 = (lossAt(w1, w2 + eps, b) - lossAt(w1, w2 - eps, b)) / (2 * eps);
    const numGb = (lossAt(w1, w2, b + eps) - lossAt(w1, w2, b - eps)) / (2 * eps);

    // Orthogonal projection of focused sample x_i onto w
    const xFocus = X[focusRow];
    const wNormSq = w[0] * w[0] + w[1] * w[1] || 1e-9;
    const scale = (xFocus[0] * w[0] + xFocus[1] * w[1]) / wNormSq;
    const proj = [scale * w[0], scale * w[1]];

    return { yhat, loss, gw1, gw2, gb, numGw1, numGw2, numGb, xFocus, proj, scale };
  }, [X, y, w1, w2, b, focusRow]);

  const traces: PlotTrace[] = [
    {
      type: 'scatter',
      mode: 'markers+text',
      x: X.map((r) => r[0]),
      y: X.map((r) => r[1]),
      text: X.map((_, i) => `x${i + 1}`),
      textposition: 'top center',
      name: 'Các vector dữ liệu xᵢ',
      marker: {
        size: X.map((_, i) => (i === focusRow ? 13 : 9)),
        color: X.map((_, i) => (i === focusRow ? '#ffd173' : '#42d3e5')),
      },
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      x: [0, w1],
      y: [0, w2],
      name: 'Vector tham số w',
      line: { color: '#f28db9', width: 3 },
      marker: { size: 7, color: '#f28db9' },
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      x: [0, stats.proj[0]],
      y: [0, stats.proj[1]],
      name: 'Hình chiếu proj_w(x)',
      line: { color: '#84e0a8', width: 3 },
      marker: { size: 8, color: '#84e0a8' },
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [stats.xFocus[0], stats.proj[0]],
      y: [stats.xFocus[1], stats.proj[1]],
      name: 'Thành phần trực giao x − proj',
      line: { color: '#9fb2ca', width: 1.5, dash: 'dot' },
    },
  ];

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Trọng số w₁: <strong>{fmt(w1, 2)}</strong>
          <input type="range" min={-2.5} max={2.5} step={0.1} value={w1} onChange={(e) => setW1(+e.target.value)} />
        </label>
        <label>
          Trọng số w₂: <strong>{fmt(w2, 2)}</strong>
          <input type="range" min={-2.5} max={2.5} step={0.1} value={w2} onChange={(e) => setW2(+e.target.value)} />
        </label>
        <label>
          Bias b: <strong>{fmt(b, 2)}</strong>
          <input type="range" min={-2} max={2} step={0.1} value={b} onChange={(e) => setB(+e.target.value)} />
        </label>
        <label>
          Mẫu đang chiếu
          <select value={focusRow} onChange={(e) => setFocusRow(+e.target.value)}>
            {X.map((row, idx) => (
              <option key={idx} value={idx}>
                x{idx + 1} = [{row.join(', ')}] (y = {y[idx]})
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grad-sim-grid">
        <div>
          <Chart
            traces={traces}
            xTitle="Đặc trưng x₁"
            yTitle="Đặc trưng x₂"
            height={320}
            layout={{
              xaxis: { range: [-3, 3.2], zeroline: true },
              yaxis: { range: [-2.8, 3.2], scaleanchor: 'x', zeroline: true },
            }}
          />
          <p className="sim-caption">
            Hình học: wᵀx = {fmt(stats.xFocus[0] * w1 + stats.xFocus[1] * w2, 3)} tỉ lệ thuận với độ dài hình chiếu trực giao của x
            {focusRow + 1} lên phương vector pháp tuyến w.
          </p>
        </div>

        <div className="grad-sim-metrics">
          <div className="metric-card">
            <span>MSE Loss: L(w, b)</span>
            <strong>{fmt(stats.loss, 5)}</strong>
          </div>
          <h4>Kiểm chứng Gradient giải tích vs. Sai phân hữu hạn</h4>
          <table className="grad-table">
            <thead>
              <tr>
                <th>Tham số</th>
                <th>Giải tích 2Xᵀ(Xw+b−y)/n</th>
                <th>Sai phân trung tâm (ε=1e-6)</th>
                <th>Sai lệch |Δ|</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>∂L/∂w₁</td>
                <td>{fmt(stats.gw1, 6)}</td>
                <td>{fmt(stats.numGw1, 6)}</td>
                <td>{Math.abs(stats.gw1 - stats.numGw1).toExponential(1)}</td>
              </tr>
              <tr>
                <td>∂L/∂w₂</td>
                <td>{fmt(stats.gw2, 6)}</td>
                <td>{fmt(stats.numGw2, 6)}</td>
                <td>{Math.abs(stats.gw2 - stats.numGw2).toExponential(1)}</td>
              </tr>
              <tr>
                <td>∂L/∂b</td>
                <td>{fmt(stats.gb, 6)}</td>
                <td>{fmt(stats.numGb, 6)}</td>
                <td>{Math.abs(stats.gb - stats.numGb).toExponential(1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/** Unit 1.2: Beta-Binomial Prior, Likelihood, Posterior & MLE vs MAP */
function SimBetaMap() {
  const [n, setN] = useState(12);
  const [k, setK] = useState(9);
  const [alpha, setAlpha] = useState(3);
  const [beta, setBeta] = useState(3);

  const safeK = Math.min(k, n);

  const { traces, mle, mapEst, postMean } = useMemo(() => {
    const grid = Array.from({ length: 121 }, (_, i) => i / 120);
    const priorY = grid.map((t) => betaPdf(t, alpha, beta));
    const postA = alpha + safeK;
    const postB = beta + (n - safeK);
    const postY = grid.map((t) => betaPdf(t, postA, postB));

    // Normalized likelihood scaled visually to match posterior peak for clarity
    const rawLik = grid.map((t) => Math.pow(Math.max(t, 1e-9), safeK) * Math.pow(Math.max(1 - t, 1e-9), n - safeK));
    const maxLik = Math.max(...rawLik) || 1;
    const maxPost = Math.max(...postY) || 1;
    const likY = rawLik.map((v) => (v / maxLik) * maxPost * 0.85);

    const mleVal = safeK / n;
    const denom = postA + postB - 2;
    const mapVal = denom > 0 ? (postA - 1) / denom : mleVal;
    const meanVal = postA / (postA + postB);

    const plotTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: grid, y: priorY, name: `Prior Beta(${alpha}, ${beta})`, line: { color: '#9fb2ca', width: 2, dash: 'dash' } },
      { type: 'scatter', mode: 'lines', x: grid, y: likY, name: `Likelihood (chuẩn hóa tỷ lệ)`, line: { color: '#ffd173', width: 2 } },
      {
        type: 'scatter',
        mode: 'lines',
        x: grid,
        y: postY,
        name: `Posterior Beta(${postA}, ${postB})`,
        fill: 'tozeroy',
        fillcolor: 'rgba(66, 211, 229, 0.14)',
        line: { color: '#42d3e5', width: 3 },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: [mleVal, mapVal, meanVal],
        y: [0, betaPdf(mapVal, postA, postB), betaPdf(meanVal, postA, postB)],
        name: 'Điểm ước lượng (MLE / MAP / Mean)',
        marker: { size: 10, color: ['#ffd173', '#f28db9', '#84e0a8'] },
      },
    ];

    return { traces: plotTraces, mle: mleVal, mapEst: mapVal, postMean: meanVal };
  }, [n, safeK, alpha, beta]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số phép thử n: <strong>{n}</strong>
          <input
            type="range"
            min={1}
            max={60}
            value={n}
            onChange={(e) => {
              const nextN = +e.target.value;
              setN(nextN);
              if (k > nextN) setK(nextN);
            }}
          />
        </label>
        <label>
          Số thành công k: <strong>{safeK}</strong>
          <input type="range" min={0} max={n} value={safeK} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label>
          Prior α: <strong>{alpha}</strong>
          <input type="range" min={1} max={12} value={alpha} onChange={(e) => setAlpha(+e.target.value)} />
        </label>
        <label>
          Prior β: <strong>{beta}</strong>
          <input type="range" min={1} max={12} value={beta} onChange={(e) => setBeta(+e.target.value)} />
        </label>
      </div>

      <Chart traces={traces} xTitle="Tham số θ ∈ [0, 1]" yTitle="Mật độ xác suất" height={330} />

      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>MLE (k / n)</span>
          <strong style={{ color: '#ffd173' }}>{fmt(mle, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>MAP (Mode hậu nghiệm)</span>
          <strong style={{ color: '#f28db9' }}>{fmt(mapEst, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Kỳ vọng hậu nghiệm E[θ | D]</span>
          <strong style={{ color: '#84e0a8' }}>{fmt(postMean, 4)}</strong>
        </div>
      </div>
      <p className="sim-caption">
        Quan sát: Khi α=β=1 (Tiên nghiệm đều), MAP trùng chính xác với MLE. Khi giữ nguyên tỷ lệ k/n và tăng n → 60, trọng lượng của Likelihood áp đảo Prior khiến MLE, MAP và Posterior Mean hội tụ về cùng một điểm.
      </p>
    </div>
  );
}

/** Unit 1.3: Monte Carlo Bias-Variance Decomposition Lab */
function SimBiasVariance() {
  const [degree, setDegree] = useState(3);
  const [noiseStd, setNoiseStd] = useState(0.3);
  const [nSamples, setNSamples] = useState(14);
  const [ridgeExp, setRidgeExp] = useState(-6); // lambda = 10^ridgeExp

  const { curveTraces, barTraces, biasSq, variance, irrNoise, totalMse } = useMemo(() => {
    const lambda = Math.pow(10, ridgeExp);
    const M = 24;
    const grid = Array.from({ length: 61 }, (_, i) => (i / 60) * 2 - 1);
    const fTrue = grid.map((x) => Math.sin(Math.PI * x));
    const polyRow = (x: number) => Array.from({ length: degree + 1 }, (_, p) => Math.pow(x, p));

    const rnd = seededRng(2026);
    const allPreds: number[][] = [];
    const sampleLastX: number[] = [];
    const sampleLastY: number[] = [];

    for (let m = 0; m < M; m++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const x = rnd() * 2 - 1;
        const [z] = normalPair(rnd);
        xs.push(x);
        ys.push(Math.sin(Math.PI * x) + noiseStd * z);
      }
      if (m === 0) {
        sampleLastX.push(...xs);
        sampleLastY.push(...ys);
      }
      // Build normal equations with Ridge regularization (penalize weights including j>=1, small stability term for j=0)
      const Phi = xs.map(polyRow);
      const d = degree + 1;
      const A = Array.from({ length: d }, (_, r) =>
        Array.from({ length: d }, (_, c) => {
          let sum = 0;
          for (let i = 0; i < nSamples; i++) sum += Phi[i][r] * Phi[i][c];
          if (r === c) sum += r === 0 ? 1e-8 : lambda;
          return sum;
        }),
      );
      const rhs = Array.from({ length: d }, (_, r) => {
        let sum = 0;
        for (let i = 0; i < nSamples; i++) sum += Phi[i][r] * ys[i];
        return sum;
      });
      const w = solve(A, rhs);
      const preds = grid.map((xg) => {
        const row = polyRow(xg);
        return row.reduce((s, v, j) => s + v * w[j], 0);
      });
      allPreds.push(preds);
    }

    const meanPred = grid.map((_, j) => allPreds.reduce((s, p) => s + p[j], 0) / M);
    const b2 = grid.reduce((s, _, j) => s + Math.pow(meanPred[j] - fTrue[j], 2), 0) / grid.length;
    const varEst =
      grid.reduce((s, _, j) => s + allPreds.reduce((sv, p) => sv + Math.pow(p[j] - meanPred[j], 2), 0) / M, 0) /
      grid.length;
    const sigma2 = noiseStd * noiseStd;
    const total = b2 + varEst + sigma2;

    const cTraces: PlotTrace[] = allPreds.slice(0, 15).map((p, idx) => ({
      type: 'scatter',
      mode: 'lines',
      x: grid,
      y: p,
      name: idx === 0 ? 'Các mô hình f̂⁽ᵐ⁾(x)' : 'Các mô hình khác',
      showlegend: idx === 0,
      line: { color: 'rgba(66, 211, 229, 0.25)', width: 1.4 },
    }));
    cTraces.push(
      {
        type: 'scatter',
        mode: 'lines',
        x: grid,
        y: fTrue,
        name: 'Hàm thật f(x) = sin(πx)',
        line: { color: '#84e0a8', width: 3 },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: grid,
        y: meanPred,
        name: 'Trung bình Monte Carlo E[f̂(x)]',
        line: { color: '#ffd173', width: 3, dash: 'dash' },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: sampleLastX,
        y: sampleLastY,
        name: '1 tập mẫu huấn luyện (n điểm)',
        marker: { size: 7, color: '#f28db9' },
      },
    );

    const bTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: ['Bias²', 'Variance', 'Nhiễu σ²', 'Tổng E[MSE]'],
        y: [b2, varEst, sigma2, total],
        marker: { color: ['#ffd173', '#f28db9', '#9fb2ca', '#42d3e5'] },
        text: [fmt(b2, 3), fmt(varEst, 3), fmt(sigma2, 3), fmt(total, 3)],
        textposition: 'auto',
      },
    ];

    return { curveTraces: cTraces, barTraces: bTraces, biasSq: b2, variance: varEst, irrNoise: sigma2, totalMse: total };
  }, [degree, noiseStd, nSamples, ridgeExp]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Bậc đa thức d: <strong>{degree}</strong>
          <input type="range" min={1} max={9} value={degree} onChange={(e) => setDegree(+e.target.value)} />
        </label>
        <label>
          Độ lệch chuẩn nhiễu σ: <strong>{fmt(noiseStd, 2)}</strong>
          <input type="range" min={0} max={0.6} step={0.05} value={noiseStd} onChange={(e) => setNoiseStd(+e.target.value)} />
        </label>
        <label>
          Cỡ mẫu train n: <strong>{nSamples}</strong>
          <input type="range" min={8} max={40} step={2} value={nSamples} onChange={(e) => setNSamples(+e.target.value)} />
        </label>
        <label>
          Ridge log₁₀(λ): <strong>{ridgeExp}</strong> (λ = {Math.pow(10, ridgeExp).toExponential(0)})
          <input type="range" min={-6} max={2} step={1} value={ridgeExp} onChange={(e) => setRidgeExp(+e.target.value)} />
        </label>
      </div>

      <div className="grad-sim-grid">
        <Chart
          traces={curveTraces}
          xTitle="x ∈ [-1, 1]"
          yTitle="f(x)"
          height={330}
          layout={{ yaxis: { range: [-2.0, 2.0] } }}
        />
        <Chart
          traces={barTraces}
          xTitle="Thành phần sai số"
          yTitle="Giá trị kỳ vọng trên [-1, 1]"
          height={330}
          layout={{ yaxis: { range: [0, Math.max(0.8, totalMse * 1.15)] } }}
        />
      </div>

      <div className="grad-sim-kpi-row">
        <div className="metric-card">
          <span>Bias²</span>
          <strong style={{ color: '#ffd173' }}>{fmt(biasSq, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Variance</span>
          <strong style={{ color: '#f28db9' }}>{fmt(variance, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Irreducible Noise σ²</span>
          <strong style={{ color: '#9fb2ca' }}>{fmt(irrNoise, 4)}</strong>
        </div>
        <div className="metric-card">
          <span>Tổng Expected MSE</span>
          <strong style={{ color: '#42d3e5' }}>{fmt(totalMse, 4)}</strong>
        </div>
      </div>
    </div>
  );
}

/** Unit 1.4: Generalization Bound (Hoeffding) & Data Leakage Simulator */
function SimGeneralizationLeakage() {
  const [logH, setLogH] = useState(4);
  const [delta, setDelta] = useState(0.05);
  const [leak, setLeak] = useState(false);
  const [outlierVal, setOutlierVal] = useState(45);

  const { boundTraces, leakTraces, muUsed, sdUsed, zTrainMean } = useMemo(() => {
    const ns = [50, 100, 200, 400, 800, 1500, 3000, 5000];
    const logSize = logH * Math.LN10;
    const epsilons = ns.map((n) => Math.sqrt((logSize + Math.log(2 / delta)) / (2 * n)));

    const bTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: ns,
        y: epsilons,
        name: `Chận tổng quát hóa ε (|H|=10^${logH}, δ=${delta})`,
        line: { color: '#42d3e5', width: 2.5 },
      },
    ];

    const train = [2.0, 3.0, 4.0, 5.0, 6.0];
    const val = [4.5, outlierVal];
    const fitPool = leak ? [...train, ...val] : train;
    const mu = fitPool.reduce((s, v) => s + v, 0) / fitPool.length;
    const sd = Math.sqrt(fitPool.reduce((s, v) => s + Math.pow(v - mu, 2), 0) / fitPool.length) || 1;
    const zTrain = train.map((v) => (v - mu) / sd);
    const zVal = val.map((v) => (v - mu) / sd);

    const lTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'markers',
        x: zTrain,
        y: train.map(() => 'Tập Train (Z)'),
        name: 'Train sau chuẩn hóa',
        marker: { size: 11, color: '#84e0a8' },
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: zVal,
        y: val.map(() => 'Tập Val (Z)'),
        name: 'Validation sau chuẩn hóa',
        marker: { size: 12, color: '#f28db9', symbol: 'diamond' },
      },
    ];

    return {
      boundTraces: bTraces,
      leakTraces: lTraces,
      muUsed: mu,
      sdUsed: sd,
      zTrainMean: zTrain.reduce((s, v) => s + v, 0) / zTrain.length,
    };
  }, [logH, delta, leak, outlierVal]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ lớn lớp giả thuyết log₁₀|H|: <strong>{logH}</strong> (|H| = 10^{logH})
          <input type="range" min={1} max={12} value={logH} onChange={(e) => setLogH(+e.target.value)} />
        </label>
        <label>
          Mức tin cậy 1 − δ: <strong>{Math.round((1 - delta) * 100)}%</strong> (δ = {delta})
          <select value={delta} onChange={(e) => setDelta(+e.target.value)}>
            <option value={0.1}>δ = 0.10 (Tin cậy 90%)</option>
            <option value={0.05}>δ = 0.05 (Tin cậy 95%)</option>
            <option value={0.01}>δ = 0.01 (Tin cậy 99%)</option>
          </select>
        </label>
        <label>
          Quy trình chuẩn hóa dữ liệu
          <select value={leak ? 'leak' : 'clean'} onChange={(e) => setLeak(e.target.value === 'leak')}>
            <option value="clean">Chuẩn (Fit scaler chỉ trên Train)</option>
            <option value="leak">Sai quy trình (Fit trên cả Train + Val)</option>
          </select>
        </label>
        <label>
          Giá trị ngoại lệ trong tập Validation: <strong>{outlierVal}</strong>
          <input type="range" min={10} max={100} value={outlierVal} onChange={(e) => setOutlierVal(+e.target.value)} />
        </label>
      </div>

      <div className="grad-sim-grid">
        <div>
          <Chart
            traces={boundTraces}
            xTitle="Cỡ mẫu huấn luyện n"
            yTitle="Chặn sai lệch |R(h) − R̂ₙ(h)| ≤ ε"
            height={300}
            layout={{ yaxis: { range: [0, 0.7] } }}
          />
          <p className="sim-caption">
            Hội tụ đều PAC: Sai lệch tổng quát hóa giảm theo O(1/√n) và chỉ tăng theo √log|H|.
          </p>
        </div>

        <div>
          <Chart
            traces={leakTraces}
            xTitle="Giá trị Z-score sau biến đổi"
            yTitle="Tập dữ liệu"
            height={300}
          />
          <p className="sim-caption">
            Scaler dùng: μ = {fmt(muUsed, 2)}, σ = {fmt(sdUsed, 2)}. Trung bình Z_train = {fmt(zTrainMean, 3)}{' '}
            {leak
              ? '⚠️ RÒ RỈ: Ngoại lệ ở tập Val đã làm lệch phân phối chuẩn hóa của tập Train!'
              : '✓ SẠCH: Z_train giữ đúng trung bình 0 bất kể ngoại lệ ở tập Val.'}
          </p>
        </div>
      </div>
    </div>
  );
}

export function GradSimulation({ id }: { id?: GradSimulationId }) {
  if (!id) return null;
  if (id === 'm1-u1') return <SimLinearProjectionGrad />;
  if (id === 'm1-u2') return <SimBetaMap />;
  if (id === 'm1-u3') return <SimBiasVariance />;
  if (id === 'm1-u4') return <SimGeneralizationLeakage />;
  return null;
}
