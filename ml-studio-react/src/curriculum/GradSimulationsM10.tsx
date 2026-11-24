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

/** Marsaglia-Tsang Gamma sampler for Thompson Sampling Beta(a, b) */
function sampleNormal(rnd: () => number): number {
  const u1 = Math.max(1e-12, rnd());
  const u2 = rnd();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sampleBeta(alpha: number, beta: number, rnd: () => number): number {
  // Fast integer/positive approximation using Gamma sum when small or normal approximation for larger params
  const sampleGamma = (k: number): number => {
    if (k < 1) {
      const u = Math.max(1e-12, rnd());
      return sampleGamma(k + 1) * Math.pow(u, 1 / k);
    }
    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (let iter = 0; iter < 50; iter++) {
      let x = 0;
      let v = 0;
      do {
        x = sampleNormal(rnd);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.max(1e-12, rnd());
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
    return d;
  };
  const x = sampleGamma(alpha);
  const y = sampleGamma(beta);
  return x / (x + y + 1e-12);
}

/** Unit 10.1: Multi-Armed Bandits Cumulative Regret (ε-greedy vs UCB1 vs Thompson) */
function SimBanditsRegret() {
  const [eps, setEps] = useState(0.12);
  const [ucbC, setUcbC] = useState(1.4);

  const { regretTraces, pullsTraces, finalEps, finalUcb, finalTs } = useMemo(() => {
    const mus = [0.25, 0.45, 0.62, 0.48, 0.78]; // optimal arm = index 4 (0.78)
    const muStar = 0.78;
    const K = mus.length;
    const T = 350;

    const runEps = () => {
      const rnd = seededRng(101);
      const Q = new Array(K).fill(0);
      const N = new Array(K).fill(0);
      const reg: number[] = [];
      let totalReg = 0;
      for (let t = 1; t <= T; t++) {
        let a = 0;
        if (rnd() < eps) a = Math.floor(rnd() * K);
        else a = Q.indexOf(Math.max(...Q));
        const r = rnd() < mus[a] ? 1 : 0;
        N[a] += 1;
        Q[a] += (r - Q[a]) / N[a];
        totalReg += muStar - mus[a];
        reg.push(totalReg);
      }
      return { reg, N };
    };

    const runUcb = () => {
      const rnd = seededRng(101);
      const Q = new Array(K).fill(0);
      const N = new Array(K).fill(0);
      const reg: number[] = [];
      let totalReg = 0;
      for (let t = 1; t <= T; t++) {
        const unvis = N.findIndex((n) => n === 0);
        let a = unvis;
        if (a < 0) {
          const scores = Q.map((q, i) => q + ucbC * Math.sqrt(Math.log(t) / N[i]));
          a = scores.indexOf(Math.max(...scores));
        }
        const r = rnd() < mus[a] ? 1 : 0;
        N[a] += 1;
        Q[a] += (r - Q[a]) / N[a];
        totalReg += muStar - mus[a];
        reg.push(totalReg);
      }
      return { reg, N };
    };

    const runTs = () => {
      const rnd = seededRng(101);
      const al = new Array(K).fill(1);
      const be = new Array(K).fill(1);
      const N = new Array(K).fill(0);
      const reg: number[] = [];
      let totalReg = 0;
      for (let t = 1; t <= T; t++) {
        const samples = al.map((aVal, i) => sampleBeta(aVal, be[i], rnd));
        const a = samples.indexOf(Math.max(...samples));
        const r = rnd() < mus[a] ? 1 : 0;
        al[a] += r;
        be[a] += 1 - r;
        N[a] += 1;
        totalReg += muStar - mus[a];
        reg.push(totalReg);
      }
      return { reg, N };
    };

    const resEps = runEps();
    const resUcb = runUcb();
    const resTs = runTs();
    const xs = Array.from({ length: T }, (_, i) => i + 1);

    const rTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: xs, y: resEps.reg, name: `ε-greedy (ε=${eps.toFixed(2)})`, line: { color: '#ff9b6a', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: xs, y: resUcb.reg, name: `UCB1 (c=${ucbC.toFixed(1)})`, line: { color: '#42d3e5', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: xs, y: resTs.reg, name: 'Thompson Sampling Beta(1,1)', line: { color: '#2dd4bf', width: 3 } },
    ];
    const arms = mus.map((m, i) => `Arm ${i} (μ=${m})`);
    const pTraces: PlotTrace[] = [
      { type: 'bar', x: arms, y: resEps.N, name: 'ε-greedy', marker: { color: '#ff9b6a' } },
      { type: 'bar', x: arms, y: resUcb.N, name: 'UCB1', marker: { color: '#42d3e5' } },
      { type: 'bar', x: arms, y: resTs.N, name: 'Thompson', marker: { color: '#2dd4bf' } },
    ];
    return {
      regretTraces: rTraces,
      pullsTraces: pTraces,
      finalEps: resEps.reg[T - 1],
      finalUcb: resUcb.reg[T - 1],
      finalTs: resTs.reg[T - 1],
    };
  }, [eps, ucbC]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Tỷ lệ thám hiểm ε (ε-greedy): <strong>{fmt(eps, 2)}</strong>
          <input type="range" min={0.0} max={0.4} step={0.02} value={eps} onChange={(e) => setEps(+e.target.value)} />
        </label>
        <label>
          Hệ số lạc quan c (UCB1): <strong>{fmt(ucbC, 1)}</strong>
          <input type="range" min={0.2} max={3.0} step={0.2} value={ucbC} onChange={(e) => setUcbC(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={regretTraces} xTitle="Số bước tương tác t" yTitle="Hối tiếc tích lũy L_t" height={300} />
        <Chart traces={pullsTraces} xTitle="Các tay máy (Tối ưu = Arm 4)" yTitle="Số lần được chọn sau 350 bước" height={300} layout={{ barmode: 'group' }} />
      </div>
      <p className="sim-caption">
        Regret tại t = 350: ε-greedy = <strong>{fmt(finalEps, 1)}</strong> (tăng tuyến tính nếu ε cố định &gt; 0) vs UCB1 ={' '}
        <strong>{fmt(finalUcb, 1)}</strong> vs Thompson Sampling = <strong>{fmt(finalTs, 1)}</strong> (tăng dưới tuyến tính O(log T)).
      </p>
    </div>
  );
}

/** Unit 10.2: Hội tụ Ánh xạ co Banach γ^k của Toán tử Bellman trong Value Iteration */
function SimBellmanContraction() {
  const [gamma, setGamma] = useState(0.85);
  const [v0Offset, setV0Offset] = useState(4.0);

  const { valTraces, errTraces, stepsToConv } = useMemo(() => {
    // 2 non-terminal states + terminal state 2 (V2=0):
    // S0 -> a0 -> S1 (r=1.0) or stay S0 (r=0.2)
    // S1 -> a0 -> 0.8*S2(5.0) + 0.2*S0(-1.0) or a1 -> S2(2.0)
    const vStar1 = (3.8 + 0.2 * (-1 + gamma * 1.0)) / (1 - 0.2 * gamma * gamma);
    const vStar0 = 1.0 + gamma * vStar1;

    let v0 = v0Offset;
    let v1 = -v0Offset * 0.5;
    const histV0 = [v0];
    const histV1 = [v1];
    const empErr = [Math.max(Math.abs(v0 - vStar0), Math.abs(v1 - vStar1))];
    const banachBound = [empErr[0]];
    const steps = 24;
    let convStep = steps;

    for (let k = 1; k <= steps; k++) {
      const nv0 = Math.max(1.0 + gamma * v1, 0.2 + gamma * v0);
      const nv1 = Math.max(0.8 * 5.0 + 0.2 * (-1.0 + gamma * v0), 2.0);
      v0 = nv0;
      v1 = nv1;
      histV0.push(v0);
      histV1.push(v1);
      const err = Math.max(Math.abs(v0 - vStar0), Math.abs(v1 - vStar1));
      empErr.push(err);
      banachBound.push(banachBound[0] * Math.pow(gamma, k));
      if (convStep === steps && err < 1e-2) convStep = k;
    }

    const ks = Array.from({ length: steps + 1 }, (_, i) => i);
    const vTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: ks, y: histV0, name: 'V_k(S0)', line: { color: '#42d3e5', width: 2.5 } },
      { type: 'scatter', mode: 'lines+markers', x: ks, y: histV1, name: 'V_k(S1)', line: { color: '#fb7185', width: 2.5 } },
      { type: 'scatter', mode: 'lines', x: [0, steps], y: [vStar0, vStar0], name: 'V*(S0)', line: { color: '#42d3e5', dash: 'dot' } },
      { type: 'scatter', mode: 'lines', x: [0, steps], y: [vStar1, vStar1], name: 'V*(S1)', line: { color: '#fb7185', dash: 'dot' } },
    ];
    const eTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: ks, y: empErr, name: 'Sai số thực ‖V_k − V*‖_∞', line: { color: '#ffd173', width: 3 } },
      { type: 'scatter', mode: 'lines', x: ks, y: banachBound, name: 'Chặn Banach γᵏ‖V₀ − V*‖_∞', line: { color: '#a8a7ff', width: 2, dash: 'dash' } },
    ];
    return { valTraces: vTraces, errTraces: eTraces, stepsToConv: convStep };
  }, [gamma, v0Offset]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Hệ số chiết khấu γ: <strong>{fmt(gamma, 2)}</strong>
          <input type="range" min={0.4} max={0.96} step={0.04} value={gamma} onChange={(e) => setGamma(+e.target.value)} />
        </label>
        <label>
          Độ lệch khởi tạo V₀: <strong>{fmt(v0Offset, 1)}</strong>
          <input type="range" min={-8.0} max={8.0} step={1.0} value={v0Offset} onChange={(e) => setV0Offset(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={valTraces} xTitle="Vòng lặp Value Iteration (k)" yTitle="Giá trị trạng thái V_k(s)" height={300} />
        <Chart traces={errTraces} xTitle="Vòng lặp k" yTitle="Khoảng cách tới điểm bất động V*" height={300} />
      </div>
      <p className="sim-caption">
        Sai số giảm xuống dưới 10⁻² sau <strong>{stepsToConv}</strong> bước lặp. Khi γ tiến sát 1.0, tốc độ co γᵏ chậm đi rõ rệt đúng theo định lý ánh xạ co Banach.
      </p>
    </div>
  );
}

/** Unit 10.3: Cliff Walking — On-Policy SARSA (An toàn) vs Off-Policy Q-Learning (Thám hiểm sát vực) */
function SimSarsaVsQLearning() {
  const [eps, setEps] = useState(0.15);

  const { traces, sarsaRet, qRet } = useMemo(() => {
    const epsValues = Array.from({ length: 18 }, (_, i) => 0.02 + i * 0.02);
    // Safe upper path takes ~17 steps (return -17 - occasional long detour);
    // Q-Learning optimal path takes 13 steps right along cliff edge (3 steps adjacent to cliff where random slip -> -100!)
    const sarsaCurve = epsValues.map((e) => -17.5 - 6.0 * e);
    const qCurve = epsValues.map((e) => -13.0 * (1 - e) - 3.0 * e * 35.0);

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines+markers', x: epsValues, y: sarsaCurve, name: 'SARSA (On-Policy: đi đường vòng an toàn)', line: { color: '#2dd4bf', width: 3 } },
      { type: 'scatter', mode: 'lines+markers', x: epsValues, y: qCurve, name: 'Q-Learning (Off-Policy: đi sát vực rơi khi ε > 0)', line: { color: '#fb7185', width: 3 } },
      { type: 'scatter', mode: 'markers', x: [eps, eps], y: [-17.5 - 6.0 * eps, -13.0 * (1 - eps) - 3.0 * eps * 35.0], name: 'ε đang chọn', marker: { size: 12, color: '#ffd173' } },
    ];
    return {
      traces: pTraces,
      sarsaRet: -17.5 - 6.0 * eps,
      qRet: -13.0 * (1 - eps) - 3.0 * eps * 35.0,
    };
  }, [eps]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Xác suất chọn hành động ngẫu nhiên ε (lúc huấn luyện): <strong>{fmt(eps, 2)}</strong>
          <input type="range" min={0.02} max={0.34} step={0.02} value={eps} onChange={(e) => setEps(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Tỷ lệ thám hiểm ε khi thu thập dữ liệu" yTitle="Tổng thưởng trung bình mỗi tập (Online Return)" height={310} />
      <p className="sim-caption">
        Tại ε = {fmt(eps, 2)}: Thưởng trực tuyến của SARSA = <strong>{fmt(sarsaRet, 1)}</strong> vs Q-Learning = <strong>{fmt(qRet, 1)}</strong>. Q-Learning hội tụ về đường tối ưu ngắn nhất (khi tắt ε = 0 ở test), nhưng trong quá trình học online lại trả giá nặng vì tự nhào xuống vực!
      </p>
    </div>
  );
}

/** Unit 10.4: Hàm mục tiêu Cắt xén PPO (Asymmetric Clipping L^CLIP) */
function SimPpoClip() {
  const [adv, setAdv] = useState(1.0);
  const [clipEps, setClipEps] = useState(0.2);

  const { traces } = useMemo(() => {
    const ratios = Array.from({ length: 120 }, (_, i) => 0.4 + (i / 119) * 1.4);
    const unclipped = ratios.map((r) => r * adv);
    const clipped = ratios.map((r) => Math.max(1 - clipEps, Math.min(1 + clipEps, r)) * adv);
    const ppoMin = ratios.map((_, i) => Math.min(unclipped[i], clipped[i]));

    const pTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: ratios, y: unclipped, name: 'Không cắt r·Â', line: { color: '#64748b', width: 2, dash: 'dot' } },
      { type: 'scatter', mode: 'lines', x: ratios, y: clipped, name: 'Chặn clip(r, 1±ε)·Â', line: { color: '#ffd173', width: 2, dash: 'dash' } },
      { type: 'scatter', mode: 'lines', x: ratios, y: ppoMin, name: 'Mục tiêu PPO: min(r·Â, clip·Â)', line: { color: '#fb7185', width: 3.5 } },
      { type: 'scatter', mode: 'markers', x: [1.0], y: [adv], name: 'Chính sách cũ (r = 1)', marker: { size: 10, color: '#42d3e5' } },
    ];
    return { traces: pTraces };
  }, [adv, clipEps]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Lợi thế Advantage Â: <strong>{adv >= 0 ? `+${fmt(adv, 1)}` : fmt(adv, 1)}</strong>
          <input type="range" min={-2.0} max={2.0} step={0.5} value={adv} onChange={(e) => setAdv(+e.target.value)} />
        </label>
        <label>
          Ngưỡng cắt ε (Clip epsilon): <strong>{fmt(clipEps, 2)}</strong>
          <input type="range" min={0.05} max={0.4} step={0.05} value={clipEps} onChange={(e) => setClipEps(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Tỷ số xác suất chính sách r(θ) = π_θ / π_old" yTitle="Giá trị hàm mục tiêu L(r)" height={310} />
      <p className="sim-caption">
        {adv > 0
          ? `Khi Â > 0 (hành động tốt): PPO cắt ngang mức thưởng tối đa tại r ≥ ${(1 + clipEps).toFixed(2)}, nhưng vẫn giữ dốc phạt nếu r giảm xuống quá thấp.`
          : adv < 0
            ? `Khi Â < 0 (hành động xấu): PPO cắt sàn tại r ≤ ${(1 - clipEps).toFixed(2)}, nhưng giữ nguyên dốc phạt mạnh nếu chính sách lỡ tăng xác suất r > ${(1 + clipEps).toFixed(2)}.`
            : 'Khi Â = 0: hành động đúng bằng trung bình Baseline nên gradient bằng 0.'}
      </p>
    </div>
  );
}

/** Unit 10.5: Calibration (Reliability Diagram + ECE dưới Temperature Scaling) & Ablation CI */
function SimCalibrationAblation() {
  const [tempScale, setTempScale] = useState(1.0);

  const { relTraces, ablTraces, eceVal, logLossVal } = useMemo(() => {
    const rawLogits = [-2.4, -1.6, -0.9, -0.35, 0.3, 0.95, 1.7, 2.5];
    const empiricalAcc = [0.16, 0.28, 0.38, 0.44, 0.57, 0.65, 0.76, 0.84];
    const binCounts = [20, 22, 25, 28, 30, 32, 26, 20];
    const totalN = binCounts.reduce((s, c) => s + c, 0);

    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
    const preds = rawLogits.map((z) => sigmoid(z / tempScale));

    const ece = preds.reduce((s, p, i) => s + (binCounts[i] / totalN) * Math.abs(empiricalAcc[i] - p), 0);
    const nll = -preds.reduce(
      (s, p, i) => s + (binCounts[i] / totalN) * (empiricalAcc[i] * Math.log(p + 1e-12) + (1 - empiricalAcc[i]) * Math.log(1 - p + 1e-12)),
      0,
    );

    const rTraces: PlotTrace[] = [
      { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], name: 'Hiệu chuẩn hoàn hảo (y = x)', line: { color: '#64748b', width: 2, dash: 'dash' } },
      { type: 'scatter', mode: 'lines+markers', x: preds, y: empiricalAcc, name: `Độ tin cậy thực nghiệm (T=${tempScale.toFixed(2)})`, line: { color: '#42d3e5', width: 3 }, marker: { size: 9, color: '#fb7185' } },
    ];

    const ablVariants = ['Baseline', '+Pre-LN', '+Cosine LR', '+Full (TempScale)'];
    const ablMeans = [82.4, 84.9, 86.1, 87.3];
    const ablErrs = [1.4, 1.1, 0.9, 0.7];
    const aTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: ablVariants,
        y: ablMeans,
        error_y: { type: 'data', array: ablErrs, visible: true, color: '#ffd173' },
        name: 'Accuracy ± 95% CI (5 seeds)',
        marker: { color: ['#64748b', '#a8a7ff', '#42d3e5', '#2dd4bf'] },
      },
    ];
    return { relTraces: rTraces, ablTraces: aTraces, eceVal: ece, logLossVal: nll };
  }, [tempScale]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Temperature Scaling hậu xử lý (T): <strong>{fmt(tempScale, 2)}</strong>
          <input type="range" min={0.5} max={2.6} step={0.1} value={tempScale} onChange={(e) => setTempScale(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={relTraces} xTitle="Xác suất dự đoán trung bình conf(B_m)" yTitle="Tần suất đúng thực nghiệm acc(B_m)" height={300} layout={{ xaxis: { range: [0, 1] }, yaxis: { range: [0, 1] } }} />
        <Chart traces={ablTraces} xTitle="Cấu hình nghiên cứu cắt giảm (Ablation)" yTitle="Mean Accuracy % (± 95% CI)" height={300} layout={{ yaxis: { range: [78, 91] } }} />
      </div>
      <p className="sim-caption">
        Calibration ECE = <strong>{fmt(eceVal, 4)}</strong> | Validation NLL = <strong>{fmt(logLossVal, 4)}</strong>. Điều chỉnh Temperature T ≈ 1.4 trên tập Validation kéo đường cong sát đường chéo lý tưởng mà không làm đổi bất kỳ nhãn argmax nào!
      </p>
    </div>
  );
}

export function GradSimulationM10({ id }: { id: GradSimulationId }) {
  if (id === 'm10-u1') return <SimBanditsRegret />;
  if (id === 'm10-u2') return <SimBellmanContraction />;
  if (id === 'm10-u3') return <SimSarsaVsQLearning />;
  if (id === 'm10-u4') return <SimPpoClip />;
  if (id === 'm10-u5') return <SimCalibrationAblation />;
  return null;
}
