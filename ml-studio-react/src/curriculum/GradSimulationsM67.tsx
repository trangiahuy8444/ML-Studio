import { useMemo, useState } from 'react';
import { Chart } from '../components/Chart';
import type { PlotTrace } from '../types';
import type { GradSimulationId } from './types';

const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : 'NaN');

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

/** Jacobi eigensolver for small symmetric matrices (returns k smallest eigenvectors stacked column-wise) */
function jacobiSmallestEigvecs(A: number[][], k: number): number[][] {
  const n = A.length;
  const M = A.map((row) => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < 25; sweep++) {
    let maxOff = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const av = Math.abs(M[i][j]);
        if (av > maxOff) {
          maxOff = av;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < 1e-9) break;
    const app = M[p][p];
    const aqq = M[q][q];
    const apq = M[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    M[p][p] = app - t * apq;
    M[q][q] = aqq + t * apq;
    M[p][q] = 0;
    M[q][p] = 0;
    for (let r = 0; r < n; r++) {
      if (r !== p && r !== q) {
        const mrp = M[r][p];
        const mrq = M[r][q];
        M[r][p] = c * mrp - s * mrq;
        M[p][r] = M[r][p];
        M[r][q] = s * mrp + c * mrq;
        M[q][r] = M[r][q];
      }
    }
    for (let r = 0; r < n; r++) {
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => M[a][a] - M[b][b]);
  const cols = order.slice(0, k);
  return Array.from({ length: n }, (_, i) => cols.map((col) => V[i][col]));
}

function runKmeans2(pts: number[][], init0: number[], init1: number[], iters = 15): number[] {
  let c0 = [...init0];
  let c1 = [...init1];
  let labels = pts.map(() => 0);
  for (let it = 0; it < iters; it++) {
    labels = pts.map((p) => {
      const d0 = p.reduce((s, v, j) => s + (v - c0[j]) ** 2, 0);
      const d1 = p.reduce((s, v, j) => s + (v - c1[j]) ** 2, 0);
      return d0 <= d1 ? 0 : 1;
    });
    const g0 = pts.filter((_, i) => labels[i] === 0);
    const g1 = pts.filter((_, i) => labels[i] === 1);
    if (g0.length > 0) c0 = c0.map((_, j) => g0.reduce((s, p) => s + p[j], 0) / g0.length);
    if (g1.length > 0) c1 = c1.map((_, j) => g1.reduce((s, p) => s + p[j], 0) / g1.length);
  }
  return labels;
}

/** Unit 6.1: K-means vs Spectral Clustering trên cụm phi lồi (2 vòng đồng tâm) */
function SimKmeansSpectral() {
  const [sigmaRbf, setSigmaRbf] = useState(0.55);
  const [noise, setNoise] = useState(0.09);

  const { kmTraces, spTraces, kmAcc, spAcc } = useMemo(() => {
    const rnd = seededRng(2026);
    const nPerRing = 18;
    const pts: number[][] = [];
    const truth: number[] = [];
    for (let i = 0; i < nPerRing; i++) {
      const th = (2 * Math.PI * i) / nPerRing;
      const [nx, ny] = normalPair(rnd);
      pts.push([1.0 * Math.cos(th) + noise * nx, 1.0 * Math.sin(th) + noise * ny]);
      truth.push(0);
    }
    for (let i = 0; i < nPerRing; i++) {
      const th = (2 * Math.PI * i) / nPerRing + 0.1;
      const [nx, ny] = normalPair(rnd);
      pts.push([2.6 * Math.cos(th) + noise * nx, 2.6 * Math.sin(th) + noise * ny]);
      truth.push(1);
    }

    const kmLabels = runKmeans2(pts, pts[0], pts[9]);
    const n = pts.length;
    const denom = 2 * sigmaRbf * sigmaRbf;
    const W = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        if (i === j) return 0;
        const d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
        return Math.exp(-d2 / denom);
      }),
    );
    const deg = W.map((r) => r.reduce((s, v) => s + v, 0));
    const dInvSqrt = deg.map((d) => (d > 1e-12 ? 1 / Math.sqrt(d) : 0));
    const Lsym = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - dInvSqrt[i] * W[i][j] * dInvSqrt[j]),
    );
    const V = jacobiSmallestEigvecs(Lsym, 2);
    const U = V.map((row) => {
      const norm = Math.hypot(row[0], row[1]) || 1;
      return [row[0] / norm, row[1] / norm];
    });
    const spLabels = runKmeans2(U, U[0], U[nPerRing]);

    const bestAcc = (pred: number[]) => {
      const matchSame = pred.filter((p, i) => p === truth[i]).length / n;
      return Math.max(matchSame, 1 - matchSame);
    };

    const mkTraces = (labels: number[]): PlotTrace[] => [0, 1].map((c) => {
      const idx = labels.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
      return {
        type: 'scatter',
        mode: 'markers',
        x: idx.map((i) => pts[i][0]),
        y: idx.map((i) => pts[i][1]),
        name: `Cụm ${c}`,
        marker: { size: 9, color: c === 0 ? '#42d3e5' : '#ff9b6a' },
      };
    });

    return {
      kmTraces: mkTraces(kmLabels),
      spTraces: mkTraces(spLabels),
      kmAcc: bestAcc(kmLabels),
      spAcc: bestAcc(spLabels),
    };
  }, [sigmaRbf, noise]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Băng thông RBF σ: <strong>{fmt(sigmaRbf, 2)}</strong>
          <input type="range" min={0.25} max={1.5} step={0.05} value={sigmaRbf} onChange={(e) => setSigmaRbf(+e.target.value)} />
        </label>
        <label>
          Nhiễu vòng đồng tâm: <strong>{fmt(noise, 2)}</strong>
          <input type="range" min={0.02} max={0.35} step={0.02} value={noise} onChange={(e) => setNoise(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={kmTraces} xTitle="x₁ (K-means: cắt phẳng Voronoi)" yTitle="x₂" height={300} layout={{ yaxis: { scaleanchor: 'x' } }} />
        <Chart traces={spTraces} xTitle="x₁ (Spectral Clustering L_sym)" yTitle="x₂" height={300} layout={{ yaxis: { scaleanchor: 'x' } }} />
      </div>
      <p className="sim-caption">
        Độ chính xác phân cụm: K-means = <strong>{fmt(kmAcc * 100, 1)}%</strong> (thất bại vì chia nửa mặt phẳng) vs Spectral Clustering ={' '}
        <strong>{fmt(spAcc * 100, 1)}%</strong>. Nếu tăng σ quá lớn (&gt; 1.1), các cạnh nối xuyên hai vòng xuất hiện làm phổ bị trộn lẫn.
      </p>
    </div>
  );
}

/** Unit 6.2: KDE Bandwidth Bias-Variance & 2-Component GMM Density */
function SimKdeGmm() {
  const [bw, setBw] = useState(0.5);

  const { traces, gmmLl, kdePeakErr } = useMemo(() => {
    const rnd = seededRng(88);
    const samples: number[] = [];
    for (let i = 0; i < 42; i++) {
      const [z] = normalPair(rnd);
      if (i < 17) samples.push(-1.6 + 0.55 * z);
      else samples.push(1.8 + 0.75 * z);
    }

    const xs = Array.from({ length: 160 }, (_, i) => -4.5 + (i / 159) * 9.0);
    const normPdf = (x: number, mu: number, sd: number) =>
      Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (Math.sqrt(2 * Math.PI) * sd);

    const trueY = xs.map((x) => 0.4 * normPdf(x, -1.6, 0.55) + 0.6 * normPdf(x, 1.8, 0.75));
    const kdeY = xs.map((x) => samples.reduce((s, xi) => s + normPdf(x, xi, bw), 0) / samples.length);

    // Compute empirical 2-component GMM likelihood at true parameters
    const ll = samples.reduce((s, xi) => {
      const l0 = Math.log(0.4 * normPdf(xi, -1.6, 0.55) + 0.6 * normPdf(xi, 1.8, 0.75) + 1e-30);
      return s + l0;
    }, 0);

    const mse = trueY.reduce((s, y, i) => s + (y - kdeY[i]) ** 2, 0) / xs.length;

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: xs, y: trueY, name: 'Mật độ thật GMM (2 đỉnh)', line: { color: '#42d3e5', width: 2.5, dash: 'dot' } },
      { type: 'scatter', mode: 'lines', x: xs, y: kdeY, name: `KDE Gaussian (h = ${bw.toFixed(2)})`, line: { color: '#ff9b6a', width: 3 } },
      { type: 'scatter', mode: 'markers', x: samples, y: samples.map(() => 0.008), name: 'Mẫu dữ liệu N=42', marker: { size: 7, color: '#a8a7ff', symbol: 'line-ns-open' } },
    ];
    return { traces: pTraces, gmmLl: ll, kdePeakErr: mse };
  }, [bw]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Băng thông KDE (h): <strong>{fmt(bw, 2)}</strong>
          <input type="range" min={0.1} max={1.6} step={0.05} value={bw} onChange={(e) => setBw(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="x" yTitle="Mật độ xác suất p̂(x)" height={320} />
      <p className="sim-caption">
        MSE giữa KDE và phân phối thật: <strong>{fmt(kdePeakErr, 5)}</strong> | Tổng Log-Likelihood GMM thật = <strong>{fmt(gmmLl, 2)}</strong>.{' '}
        {bw < 0.25
          ? 'h quá nhỏ → nhiều gai nhọn theo từng điểm (High Variance / Overfitting).'
          : bw > 0.95
            ? 'h quá lớn → làm phẳng mất hai đỉnh (High Bias / Undersmoothing ngược lại).'
            : 'Vùng h cân bằng: khôi phục rõ cấu trúc hai đỉnh bimodal.'}
      </p>
    </div>
  );
}

/** Unit 6.3: Thuật toán EM & Chặn dưới ELBO theo Bất đẳng thức Jensen */
function SimEmJensen() {
  const [stepIdx, setStepIdx] = useState(1);

  const { curveTraces, histTraces, curMu, curLl } = useMemo(() => {
    const data = [-2.2, -1.9, -1.7, -1.5, -1.4, 2.3, 2.5, 2.7, 2.8, 3.1];
    const sd = 0.65;
    const mu0 = -1.75; // fixed well-identified mode 1
    const normPdf = (x: number, m: number) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (Math.sqrt(2 * Math.PI) * sd);

    const logLik = (m2: number) =>
      data.reduce((s, x) => s + Math.log(0.5 * normPdf(x, mu0) + 0.5 * normPdf(x, m2) + 1e-30), 0);

    // Run 7 EM steps starting from poor mu2 = 0.2
    const musHist = [0.2];
    const llsHist = [logLik(0.2)];
    for (let t = 0; t < 7; t++) {
      const mCur = musHist[musHist.length - 1];
      const gammas = data.map((x) => {
        const p0 = 0.5 * normPdf(x, mu0);
        const p1 = 0.5 * normPdf(x, mCur);
        return p1 / (p0 + p1 + 1e-30);
      });
      const sumG = gammas.reduce((s, g) => s + g, 0);
      const mNext = data.reduce((s, x, i) => s + gammas[i] * x, 0) / (sumG || 1);
      musHist.push(mNext);
      llsHist.push(logLik(mNext));
    }

    const mAt = musHist[stepIdx];
    const gammasAt = data.map((x) => {
      const p0 = 0.5 * normPdf(x, mu0);
      const p1 = 0.5 * normPdf(x, mAt);
      return p1 / (p0 + p1 + 1e-30);
    });

    const elboAt = (m2: number) =>
      data.reduce((s, x, i) => {
        const q1 = Math.max(1e-12, Math.min(1 - 1e-12, gammasAt[i]));
        const q0 = 1 - q1;
        const term0 = q0 * Math.log((0.5 * normPdf(x, mu0)) / q0);
        const term1 = q1 * Math.log((0.5 * normPdf(x, m2) + 1e-30) / q1);
        return s + term0 + term1;
      }, 0);

    const grid = Array.from({ length: 110 }, (_, i) => -0.5 + (i / 109) * 4.2);
    const llVals = grid.map(logLik);
    const elboVals = grid.map(elboAt);
    const nextMu = musHist[Math.min(stepIdx + 1, musHist.length - 1)];

    const cTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: grid, y: llVals, name: 'Log-Likelihood ℓ(μ₂)', line: { color: '#42d3e5', width: 3 } },
      { type: 'scatter', mode: 'lines', x: grid, y: elboVals, name: `ELBO(q⁽${stepIdx}⁾, μ₂) [Jensen Lower Bound]`, line: { color: '#ffd173', width: 2.5, dash: 'dash' } },
      { type: 'scatter', mode: 'markers', x: [mAt], y: [logLik(mAt)], name: `Điểm tiếp xúc θ⁽${stepIdx}⁾`, marker: { size: 11, color: '#ff9b6a' } },
      { type: 'scatter', mode: 'markers', x: [nextMu], y: [logLik(nextMu)], name: `M-step θ⁽${stepIdx + 1}⁾`, marker: { size: 11, color: '#a8a7ff', symbol: 'diamond' } },
    ];

    const hTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: llsHist.map((_, i) => i), y: llsHist, name: 'ℓ(θ⁽ᵗ⁾) theo vòng lặp EM', line: { color: '#42d3e5', width: 2.5 }, marker: { size: 8, color: '#ffd173' } },
    ];

    return { curveTraces: cTraces, histTraces: hTraces, curMu: mAt, curLl: llsHist[stepIdx] };
  }, [stepIdx]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Vòng lặp EM (t): <strong>{stepIdx}</strong>
          <input type="range" min={0} max={6} step={1} value={stepIdx} onChange={(e) => setStepIdx(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={curveTraces} xTitle="Tham số μ₂" yTitle="Hàm mục tiêu" height={310} />
        <Chart traces={histTraces} xTitle="Số vòng lặp EM (t)" yTitle="Log-Likelihood ℓ(θ⁽ᵗ⁾)" height={310} />
      </div>
      <p className="sim-caption">
        Tại bước t = {stepIdx}: μ₂ = <strong>{fmt(curMu, 3)}</strong>, ℓ(θ) = <strong>{fmt(curLl, 3)}</strong>. Chặn dưới vàng ELBO luôn chạm khít đường cong ℓ(θ) tại θ⁽ᵗ⁾ (KL = 0 ở E-step), nên tối đa hóa ELBO ở M-step chắc chắn kéo ℓ(θ) đi lên đơn điệu.
      </p>
    </div>
  );
}

/** Unit 6.4: PCA via SVD, Orthogonal Projections & Variance Ratio */
function SimPcaSvd() {
  const [rho, setRho] = useState(0.85);
  const [ratioY, setRatioY] = useState(0.6);

  const { traces, varRatio, reconMse } = useMemo(() => {
    const rnd = seededRng(51);
    const pts: number[][] = [];
    const sd1 = 1.6;
    const sd2 = 1.6 * ratioY;
    for (let i = 0; i < 45; i++) {
      const [z1, z2] = normalPair(rnd);
      const x1 = sd1 * z1;
      const x2 = sd2 * (rho * z1 + Math.sqrt(Math.max(0.01, 1 - rho * rho)) * z2);
      pts.push([x1, x2]);
    }
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const xc = pts.map((p) => [p[0] - mx, p[1] - my]);

    // 2x2 covariance matrix
    let cxx = 0, cxy = 0, cyy = 0;
    for (const [x, y] of xc) {
      cxx += x * x;
      cxy += x * y;
      cyy += y * y;
    }
    cxx /= xc.length;
    cxy /= xc.length;
    cyy /= xc.length;

    const tr = cxx + cyy;
    const disc = Math.sqrt((cxx - cyy) ** 2 + 4 * cxy * cxy);
    const lam1 = 0.5 * (tr + disc);
    const lam2 = Math.max(0, 0.5 * (tr - disc));
    const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
    const u1 = [Math.cos(angle), Math.sin(angle)];
    const u2 = [-Math.sin(angle), Math.cos(angle)];

    const projs = xc.map(([x, y]) => {
      const t = x * u1[0] + y * u1[1];
      return [t * u1[0], t * u1[1]];
    });
    const mse = xc.reduce((s, [x, y], i) => s + (x - projs[i][0]) ** 2 + (y - projs[i][1]) ** 2, 0) / xc.length;

    const resLinesX: (number | null)[] = [];
    const resLinesY: (number | null)[] = [];
    for (let i = 0; i < xc.length; i += 2) {
      resLinesX.push(xc[i][0], projs[i][0], null);
      resLinesY.push(xc[i][1], projs[i][1], null);
    }

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: resLinesX, y: resLinesY, name: 'Sai số chiếu trực giao', line: { color: 'rgba(255,255,255,0.2)', width: 1 }, connectgaps: false },
      { type: 'scatter', mode: 'markers', x: xc.map((p) => p[0]), y: xc.map((p) => p[1]), name: 'Dữ liệu đã tâm hóa X̃', marker: { size: 7, color: '#42d3e5' } },
      { type: 'scatter', mode: 'markers', x: projs.map((p) => p[0]), y: projs.map((p) => p[1]), name: 'Tái tạo hạng 1 trên PC1', marker: { size: 6, color: '#ffd173' } },
      {
        type: 'scatter',
        mode: 'lines',
        x: [-3.2 * u1[0], 3.2 * u1[0]],
        y: [-3.2 * u1[1], 3.2 * u1[1]],
        name: 'Trục chính PC1 (u₁)',
        line: { color: '#ff9b6a', width: 3 },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: [-1.8 * u2[0], 1.8 * u2[0]],
        y: [-1.8 * u2[1], 1.8 * u2[1]],
        name: 'Trục chính PC2 (u₂)',
        line: { color: '#a8a7ff', width: 2, dash: 'dash' },
      },
    ];
    return { traces: pTraces, varRatio: lam1 / (lam1 + lam2 + 1e-12), reconMse: mse };
  }, [rho, ratioY]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Tương quan ρ giữa x₁ và x₂: <strong>{fmt(rho, 2)}</strong>
          <input type="range" min={-0.95} max={0.95} step={0.05} value={rho} onChange={(e) => setRho(+e.target.value)} />
        </label>
        <label>
          Tỷ lệ biên độ σ₂ / σ₁: <strong>{fmt(ratioY, 2)}</strong>
          <input type="range" min={0.2} max={1.2} step={0.05} value={ratioY} onChange={(e) => setRatioY(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="x̃₁" yTitle="x̃₂" height={330} layout={{ yaxis: { scaleanchor: 'x' } }} />
      <p className="sim-caption">
        Tỷ lệ phương sai giữ lại trên PC1 (k = 1): <strong>{fmt(varRatio * 100, 1)}%</strong> | MSE tái tạo trực giao ={' '}
        <strong>{fmt(reconMse, 4)}</strong> (chính bằng trị riêng bị loại λ₂). Khi ρ gần 0 và σ₂ ≈ σ₁, PCA mất hướng ưu tiên rõ rệt.
      </p>
    </div>
  );
}

/** Unit 7.1: Không gian Lọc ảnh 2D (Original vs Gaussian / Median / Sobel Edge) */
function SimImageFilter() {
  const [filterKind, setFilterKind] = useState<'orig' | 'box' | 'median' | 'sobel'>('sobel');
  const [noiseAmp, setNoiseAmp] = useState(12);

  const { heatTraces, profileTraces } = useMemo(() => {
    const H = 12;
    const W = 12;
    const rnd = seededRng(701);
    const img: number[][] = Array.from({ length: H }, (_, r) =>
      Array.from({ length: W }, (_, c) => {
        const base = r >= 3 && r <= 8 && c >= 4 && c <= 9 ? 200 : 40;
        const gNoise = (rnd() - 0.5) * 2 * noiseAmp;
        let val = base + gNoise;
        if ((r === 2 && c === 3) || (r === 7 && c === 1)) val = 250; // salt impulse noise
        return Math.max(0, Math.min(255, Math.round(val)));
      }),
    );

    const getPx = (r: number, c: number) => img[Math.max(0, Math.min(H - 1, r))][Math.max(0, Math.min(W - 1, c))];
    const filtered: number[][] = Array.from({ length: H }, (_, r) =>
      Array.from({ length: W }, (_, c) => {
        if (filterKind === 'orig') return img[r][c];
        const patch: number[] = [];
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) patch.push(getPx(r + dr, c + dc));
        }
        if (filterKind === 'box') return patch.reduce((s, v) => s + v, 0) / 9;
        if (filterKind === 'median') {
          patch.sort((a, b) => a - b);
          return patch[4];
        }
        // Sobel horizontal gradient |Gx|
        const gx = -patch[0] + patch[2] - 2 * patch[3] + 2 * patch[5] - patch[6] + patch[8];
        return Math.min(255, Math.abs(gx) / 2);
      }),
    );

    const hTraces: PlotTrace[] = [
      {
        type: 'heatmap',
        z: filtered,
        zmin: 0,
        zmax: 255,
        colorscale: 'Greys',
        reversescale: true,
        showscale: true,
      },
    ];
    const cols = Array.from({ length: W }, (_, c) => c);
    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: cols, y: img[5], name: 'Hàng ảnh gốc (y=5)', line: { color: '#42d3e5', width: 2, dash: 'dot' } },
      { type: 'scatter', mode: 'lines+markers', x: cols, y: filtered[5], name: `Sau lọc (${filterKind})`, line: { color: '#ff9b6a', width: 3 } },
    ];
    return { heatTraces: hTraces, profileTraces: pTraces };
  }, [filterKind, noiseAmp]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Bộ lọc 3×3:
          <select value={filterKind} onChange={(e) => setFilterKind(e.target.value as 'orig' | 'box' | 'median' | 'sobel')}>
            <option value="orig">Ảnh gốc (có nhiễu xung)</option>
            <option value="box">Box Blur 3×3 (Thông thấp)</option>
            <option value="median">Median 3×3 (Khử muối-tiêu)</option>
            <option value="sobel">Sobel |Gₓ| (Dò biên đứng)</option>
          </select>
        </label>
        <label>
          Nhiễu nền ±: <strong>{noiseAmp}</strong>
          <input type="range" min={0} max={45} step={3} value={noiseAmp} onChange={(e) => setNoiseAmp(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={heatTraces} xTitle="Cột pixel x" yTitle="Hàng pixel y" height={300} layout={{ yaxis: { autorange: 'reversed' } }} />
        <Chart traces={profileTraces} xTitle="Tọa độ cột x (tại y=5)" yTitle="Cường độ sáng [0–255]" height={300} layout={{ yaxis: { range: [0, 265] } }} />
      </div>
      <p className="sim-caption">
        Box Blur làm mờ biên và kéo nhiễu xung (250) loang ra lân cận, trong khi Median 3×3 xóa sạch nhiễu muối tiêu mà vẫn giữ sắc nét bờ bước nhảy cạnh vuông.
      </p>
    </div>
  );
}

/** Unit 7.2: Universal Approximation bằng ReLU MLP 1 tầng ẩn rộng H */
function SimMlpUniversal() {
  const [width, setWidth] = useState(12);

  const { traces, mse } = useMemo(() => {
    const xs = Array.from({ length: 80 }, (_, i) => -3.0 + (i / 79) * 6.0);
    const target = (x: number) => Math.sin(2.0 * x) + 0.4 * Math.cos(4.0 * x);
    const ys = xs.map(target);

    // Place (width) hinge knots uniformly across [-3, 3] and solve ridge for output weights
    const knots = Array.from({ length: width }, (_, k) => -2.8 + ((k + 0.5) / width) * 5.6);
    const cols = width + 2; // 1 + x + ReLU(x - knot_k)
    const phi = xs.map((x) => [1, x, ...knots.map((t) => Math.max(0, x - t))]);

    // Normal equations (Phi^T Phi + 1e-4 I) w = Phi^T y solved by Gaussian elimination
    const A = Array.from({ length: cols }, (_, r) =>
      Array.from({ length: cols }, (_, c) =>
        phi.reduce((s, row) => s + row[r] * row[c], 0) + (r === c ? 1e-3 : 0),
      ),
    );
    const b = Array.from({ length: cols }, (_, r) => phi.reduce((s, row, i) => s + row[r] * ys[i], 0));

    // Solve A w = b
    for (let i = 0; i < cols; i++) {
      let piv = i;
      for (let r = i + 1; r < cols; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
      [A[i], A[piv]] = [A[piv], A[i]];
      [b[i], b[piv]] = [b[piv], b[i]];
      const div = A[i][i] || 1e-12;
      for (let r = i + 1; r < cols; r++) {
        const fac = A[r][i] / div;
        for (let c = i; c < cols; c++) A[r][c] -= fac * A[i][c];
        b[r] -= fac * b[i];
      }
    }
    const w = new Array(cols).fill(0);
    for (let i = cols - 1; i >= 0; i--) {
      let s = b[i];
      for (let c = i + 1; c < cols; c++) s -= A[i][c] * w[c];
      w[i] = s / (A[i][i] || 1e-12);
    }

    const preds = phi.map((row) => row.reduce((s, v, j) => s + v * w[j], 0));
    const err = preds.reduce((s, p, i) => s + (p - ys[i]) ** 2, 0) / ys.length;

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: xs, y: ys, name: 'Hàm mục tiêu trơn f(x)', line: { color: '#42d3e5', width: 2.5, dash: 'dot' } },
      { type: 'scatter', mode: 'lines', x: xs, y: preds, name: `ReLU MLP (${width} nơ-ron ẩn)`, line: { color: '#38bdf8', width: 3 } },
    ];
    return { traces: pTraces, mse: err };
  }, [width]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số nơ-ron ReLU ẩn (H): <strong>{width}</strong>
          <input type="range" min={2} max={40} step={2} value={width} onChange={(e) => setWidth(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="x" yTitle="y = f̂(x)" height={310} />
      <p className="sim-caption">
        MSE xấp xỉ: <strong>{fmt(mse, 6)}</strong>. Mỗi nơ-ron ReLU bổ sung thêm một điểm gãy tuyến tính (piecewise linear knot); khi H tăng đủ lớn, mạng 1 tầng ẩn bám sát bất kỳ hàm liên tục nào trên [-3, 3].
      </p>
    </div>
  );
}

/** Unit 7.3: Bảo toàn phương sai theo độ sâu (Small vs Xavier vs He Init vs ResNet+Norm) */
function SimDeepSignalProp() {
  const [depth, setDepth] = useState(24);
  const [scaleMult, setScaleMult] = useState(1.0);

  const { traces, endHe, endXavier } = useMemo(() => {
    const layers = Array.from({ length: depth + 1 }, (_, l) => l);
    // For ReLU: Var(a_l) = (n * Var(w) / 2) * Var(a_{l-1}) = g^2 * Var(a_{l-1})
    const stdTrace = (factorPerLayer: number) =>
      layers.map((l) => Math.min(50, Math.pow(factorPerLayer * scaleMult, l / 2)));

    const xavier = stdTrace(0.5); // Xavier uses 1/n -> factor 1/2 after ReLU
    const he = stdTrace(1.0); // He uses 2/n -> factor 1.0 after ReLU
    const large = stdTrace(1.35); // Over-scaled init -> explodes
    const resnet = layers.map((l) => Math.sqrt(1 + 0.08 * l)); // bounded growth normalized by BN/LN

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: layers, y: xavier, name: 'Xavier/Small Init (Var = 1/n, tắt dần)', line: { color: '#f28db9', width: 2.5 } },
      { type: 'scatter', mode: 'lines+markers', x: layers, y: he, name: 'Kaiming He Init (Var = 2/n, ổn định ~1)', line: { color: '#42d3e5', width: 3 } },
      { type: 'scatter', mode: 'lines+markers', x: layers, y: large, name: 'Khởi tạo quá lớn (Bùng nổ tín hiệu)', line: { color: '#ff9b6a', width: 2, dash: 'dash' } },
      { type: 'scatter', mode: 'lines', x: layers, y: resnet, name: 'Residual Skip + Norm', line: { color: '#a8a7ff', width: 2.5, dash: 'dot' } },
    ];
    return { traces: pTraces, endHe: he[he.length - 1], endXavier: xavier[xavier.length - 1] };
  }, [depth, scaleMult]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ sâu mạng L: <strong>{depth}</strong> tầng
          <input type="range" min={6} max={36} step={2} value={depth} onChange={(e) => setDepth(+e.target.value)} />
        </label>
        <label>
          Hệ số nhân độ lệch chuẩn trọng số: <strong>{fmt(scaleMult, 2)}×</strong>
          <input type="range" min={0.75} max={1.3} step={0.05} value={scaleMult} onChange={(e) => setScaleMult(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Chỉ số tầng l" yTitle="Độ lệch chuẩn kích hoạt Std(a⁽ˡ⁾)" height={310} layout={{ yaxis: { range: [0, 6] } }} />
      <p className="sim-caption">
        Tại tầng l = {depth}: Std(Xavier trên ReLU) = <strong>{fmt(endXavier, 5)}</strong> (triệt tiêu theo (1/√2)^L) vs Std(Kaiming He) ={' '}
        <strong>{fmt(endHe, 3)}</strong>. Chỉ cần lệch chuẩn trọng số khỏi giá trị tới hạn một lượng nhỏ cũng khuếch đại theo hàm mũ độ sâu!
      </p>
    </div>
  );
}

/** Unit 7.4: Receptive Field tích lũy qua các tầng Conv2D */
function SimReceptiveField() {
  const [layers, setLayers] = useState(5);
  const [kernel, setKernel] = useState(3);
  const [dilation, setDilation] = useState(1);
  const [stride, setStride] = useState(1);

  const { rfTraces, sizeTraces, finalRf, finalSize } = useMemo(() => {
    const xs = Array.from({ length: layers + 1 }, (_, i) => i);
    let rfStd = 1;
    let jumpStd = 1;
    let rfUser = 1;
    let jumpUser = 1;
    let sizeUser = 224;

    const rfsStd = [1];
    const rfsUser = [1];
    const sizes = [224];

    for (let l = 1; l <= layers; l++) {
      rfStd = rfStd + (3 - 1) * jumpStd;
      jumpStd = jumpStd * 1;
      rfsStd.push(rfStd);

      rfUser = rfUser + dilation * (kernel - 1) * jumpUser;
      jumpUser = jumpUser * stride;
      rfsUser.push(rfUser);

      // Assume padding p = floor(d*(k-1)/2) ("same" modulo stride)
      const p = Math.floor((dilation * (kernel - 1)) / 2);
      sizeUser = Math.max(1, Math.floor((sizeUser + 2 * p - dilation * (kernel - 1) - 1) / stride) + 1);
      sizes.push(sizeUser);
    }

    const rTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: xs, y: rfsStd, name: 'Baseline 3×3 (s=1, d=1)', line: { color: '#a8a7ff', width: 2.5, dash: 'dot' } },
      { type: 'scatter', mode: 'lines+markers', x: xs, y: rfsUser, name: `Cấu hình (k=${kernel}, s=${stride}, d=${dilation})`, line: { color: '#38bdf8', width: 3 } },
    ];
    const sTraces: PlotTrace[] = [
      { type: 'bar', x: xs.map((l) => `L${l}`), y: sizes, name: 'Độ phân giải không gian H_out', marker: { color: '#42d3e5' } },
    ];
    return { rfTraces: rTraces, sizeTraces: sTraces, finalRf: rfsUser[rfsUser.length - 1], finalSize: sizes[sizes.length - 1] };
  }, [layers, kernel, dilation, stride]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số tầng Conv: <strong>{layers}</strong>
          <input type="range" min={1} max={10} value={layers} onChange={(e) => setLayers(+e.target.value)} />
        </label>
        <label>
          Kernel k: <strong>{kernel}</strong>
          <input type="range" min={3} max={7} step={2} value={kernel} onChange={(e) => setKernel(+e.target.value)} />
        </label>
        <label>
          Stride s: <strong>{stride}</strong>
          <input type="range" min={1} max={2} step={1} value={stride} onChange={(e) => setStride(+e.target.value)} />
        </label>
        <label>
          Dilation d: <strong>{dilation}</strong>
          <input type="range" min={1} max={3} step={1} value={dilation} onChange={(e) => setDilation(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={rfTraces} xTitle="Tầng tích chập l" yTitle="Trường tiếp nhận (RF pixels)" height={300} />
        <Chart traces={sizeTraces} xTitle="Tầng mạng" yTitle="Kích thước Feature Map (từ 224×224)" height={300} />
      </div>
      <p className="sim-caption">
        Sau {layers} tầng: Receptive Field đạt <strong>{finalRf}×{finalRf}</strong> pixel với bản đồ đặc trưng còn{' '}
        <strong>{finalSize}×{finalSize}</strong>. Dilation d &gt; 1 mở rộng RF rất nhanh mà không làm mất độ phân giải không gian.
      </p>
    </div>
  );
}

export function GradSimulationM67({ id }: { id: GradSimulationId }) {
  if (id === 'm6-u1') return <SimKmeansSpectral />;
  if (id === 'm6-u2') return <SimKdeGmm />;
  if (id === 'm6-u3') return <SimEmJensen />;
  if (id === 'm6-u4') return <SimPcaSvd />;
  if (id === 'm7-u1') return <SimImageFilter />;
  if (id === 'm7-u2') return <SimMlpUniversal />;
  if (id === 'm7-u3') return <SimDeepSignalProp />;
  if (id === 'm7-u4') return <SimReceptiveField />;
  return null;
}
