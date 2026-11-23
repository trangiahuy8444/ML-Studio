import { useMemo, useState } from 'react';
import { Chart } from '../components/Chart';
import type { PlotTrace } from '../types';
import type { GradSimulationId } from './types';

const fmt = (v: number, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : 'NaN');

function rowSoftmax(scores: number[]): number[] {
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - m));
  const sum = exps.reduce((acc, v) => acc + v, 0) || 1;
  return exps.map((v) => v / sum);
}

/** Unit 8.1: Không gian Word2Vec & Phép loại suy Vector (King - Man + Woman ≈ Queen) */
function SimWord2VecAnalogy() {
  const [shift, setShift] = useState(0.0);

  const { traces, simQueen, simApple } = useMemo(() => {
    const man = [0.2, 0.2];
    const woman = [0.25, 1.35];
    const king = [1.85, 0.35];
    const queenTrue = [1.9 + shift, 1.5 - 0.35 * shift];
    const apple = [-1.1, -0.7];
    const analogy = [king[0] - man[0] + woman[0], king[1] - man[1] + woman[1]];

    const cos = (a: number[], b: number[]) => {
      const na = Math.hypot(a[0], a[1]) || 1;
      const nb = Math.hypot(b[0], b[1]) || 1;
      return (a[0] * b[0] + a[1] * b[1]) / (na * nb);
    };

    const pTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: [man[0], woman[0], null, king[0], analogy[0]],
        y: [man[1], woman[1], null, king[1], analogy[1]],
        name: 'Vector giới tính (woman − man)',
        line: { color: '#c084fc', width: 2.5, dash: 'dash' },
        connectgaps: false,
      },
      {
        type: 'scatter',
        mode: 'markers+text',
        x: [man[0], woman[0], king[0], queenTrue[0], apple[0]],
        y: [man[1], woman[1], king[1], queenTrue[1], apple[1]],
        text: ['man', 'woman', 'king', 'queen', 'apple'],
        textposition: 'top center',
        name: 'Embedding từ vựng',
        marker: { size: 11, color: ['#42d3e5', '#42d3e5', '#ffd173', '#ff9b6a', '#8c9fb8'] },
      },
      {
        type: 'scatter',
        mode: 'markers+text',
        x: [analogy[0]],
        y: [analogy[1]],
        text: ['king − man + woman'],
        textposition: 'bottom center',
        name: 'Điểm truy vấn loại suy',
        marker: { size: 13, color: '#2dd4bf', symbol: 'star' },
      },
    ];

    return {
      traces: pTraces,
      simQueen: cos(analogy, queenTrue),
      simApple: cos(analogy, apple),
    };
  }, [shift]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ lệch ngữ nghĩa của "queen" (nhiễu ngữ liệu): <strong>{fmt(shift, 2)}</strong>
          <input type="range" min={-1.0} max={1.0} step={0.1} value={shift} onChange={(e) => setShift(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Thành phần ngữ nghĩa 1 (Hoàng gia)" yTitle="Thành phần ngữ nghĩa 2 (Giới tính)" height={320} />
      <p className="sim-caption">
        Cosine(king − man + woman, queen) = <strong>{fmt(simQueen, 4)}</strong> vs Cosine(..., apple) ={' '}
        <strong>{fmt(simApple, 4)}</strong>. Negative Sampling tổ chức các mối quan hệ đồng xuất hiện thành các dịch chuyển tuyến tính song song trong không gian dày.
      </p>
    </div>
  );
}

/** Unit 8.2: Triệt tiêu/Bùng nổ Gradient trong Vanilla RNN vs Constant Error Carousel của LSTM */
function SimLstmVanishing() {
  const [sigmaRnn, setSigmaRnn] = useState(0.78);
  const [forgetGate, setForgetGate] = useState(0.96);

  const { traces, endRnn, endLstm } = useMemo(() => {
    const steps = Array.from({ length: 30 }, (_, k) => k);
    const rnnCurve = steps.map((k) => Math.min(20, Math.pow(sigmaRnn * 0.92, k)));
    const lstmCurve = steps.map((k) => Math.pow(forgetGate, k));

    const pTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: steps,
        y: rnnCurve,
        name: `Vanilla RNN (σ(W_hh)=${sigmaRnn.toFixed(2)})`,
        line: { color: '#ff9b6a', width: 2.5 },
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: steps,
        y: lstmCurve,
        name: `LSTM Cell Highway (f_t=${forgetGate.toFixed(2)})`,
        line: { color: '#42d3e5', width: 3 },
      },
    ];
    return {
      traces: pTraces,
      endRnn: rnnCurve[rnnCurve.length - 1],
      endLstm: lstmCurve[lstmCurve.length - 1],
    };
  }, [sigmaRnn, forgetGate]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ khuếch đại ma trận RNN σ(W_hh): <strong>{fmt(sigmaRnn, 2)}</strong>
          <input type="range" min={0.5} max={1.25} step={0.03} value={sigmaRnn} onChange={(e) => setSigmaRnn(+e.target.value)} />
        </label>
        <label>
          Cổng quên LSTM f_t: <strong>{fmt(forgetGate, 2)}</strong>
          <input type="range" min={0.5} max={0.99} step={0.01} value={forgetGate} onChange={(e) => setForgetGate(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Khoảng cách bước thời gian Δt = T − k" yTitle="Độ lớn gradient tương đối ‖∂c_T/∂c_k‖" height={310} layout={{ yaxis: { range: [0, 3.2] } }} />
      <p className="sim-caption">
        Sau Δt = 29 bước: Vanilla RNN còn <strong>{fmt(endRnn, 5)}</strong> ({sigmaRnn > 1.09 ? 'bùng nổ!' : 'tắt dần về 0'}) vs LSTM giữ được{' '}
        <strong>{fmt(endLstm, 3)}</strong> nhờ cập nhật cộng tính c_t = f_t ⊙ c_{'{t-1}'} + i_t ⊙ g_t.
      </p>
    </div>
  );
}

/** Unit 8.3: Bản đồ căn chỉnh Bahdanau Attention giữa câu Anh → Việt */
function SimBahdanauAlign() {
  const [sharpness, setSharpness] = useState(2.2);
  const [targetIdx, setTargetIdx] = useState(2);

  const { heatTraces, barTraces, srcTokens, tgtTokens } = useMemo(() => {
    const src = ['I', 'study', 'machine', 'learning', 'today'];
    const tgt = ['Hôm_nay', 'tôi', 'học', 'máy_học'];
    // Raw affinity logits showing word order swap (today -> Hôm_nay, machine+learning -> máy_học)
    const rawLogits: number[][] = [
      [0.0, 0.1, 0.2, 0.2, 2.5], // Hôm_nay <- today
      [2.4, 0.3, 0.1, 0.1, 0.2], // tôi <- I
      [0.3, 2.3, 0.4, 0.6, 0.1], // học <- study
      [0.1, 0.4, 2.1, 2.2, 0.1], // máy_học <- machine, learning
    ];
    const alphas = rawLogits.map((row) => rowSoftmax(row.map((v) => v * sharpness)));

    const hTraces: PlotTrace[] = [
      {
        type: 'heatmap',
        x: src,
        y: tgt,
        z: alphas,
        zmin: 0,
        zmax: 1,
        colorscale: 'Viridis',
        showscale: true,
      },
    ];
    const bTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: src,
        y: alphas[targetIdx],
        name: `α cho từ đích "${tgt[targetIdx]}"`,
        marker: { color: '#c084fc' },
      },
    ];
    return { heatTraces: hTraces, barTraces: bTraces, srcTokens: src, tgtTokens: tgt };
  }, [sharpness, targetIdx]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Từ đích đang sinh:
          <select value={targetIdx} onChange={(e) => setTargetIdx(+e.target.value)}>
            {tgtTokens.map((tok, i) => (
              <option key={tok} value={i}>
                t={i}: "{tok}"
              </option>
            ))}
          </select>
        </label>
        <label>
          Biên độ v_a (Độ nhọn Attention): <strong>{fmt(sharpness, 1)}×</strong>
          <input type="range" min={0.4} max={4.0} step={0.2} value={sharpness} onChange={(e) => setSharpness(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={heatTraces} xTitle={`Từ nguồn Encoder (${srcTokens.length} tokens)`} yTitle="Từ đích Decoder" height={300} layout={{ yaxis: { autorange: 'reversed' } }} />
        <Chart traces={barTraces} xTitle="Từ nguồn" yTitle="Trọng số α_{t,i}" height={300} layout={{ yaxis: { range: [0, 1.05] } }} />
      </div>
      <p className="sim-caption">
        Attention tự học phép đảo trật tự từ giữa hai ngôn ngữ (như "today" ở cuối câu nguồn căn chỉnh với "Hôm_nay" ở đầu câu đích; "máy_học" trải trọng số lên cả "machine" và "learning").
      </p>
    </div>
  );
}

/** Unit 8.4: Transformer Self-Attention — Causal Mask & Vai trò chia √d_k */
function SimTransformerAttention() {
  const [causal, setCausal] = useState(true);
  const [scaleBySqrtDk, setScaleBySqrtDk] = useState(true);
  const [dk, setDk] = useState(64);

  const { traces, entropy0 } = useMemo(() => {
    const tokens = ['Học', 'máy', 'hiện', 'đại', '2026'];
    const baseDots: number[][] = [
      [1.2, 0.8, 0.1, 0.2, -0.3],
      [0.9, 1.5, 0.4, 0.5, 0.2],
      [0.2, 0.6, 1.4, 0.9, 0.3],
      [0.1, 0.5, 1.0, 1.6, 0.7],
      [-0.2, 0.3, 0.4, 0.9, 1.7],
    ];
    const mult = scaleBySqrtDk ? 1.0 : Math.sqrt(dk) / 2.2;
    const weights = baseDots.map((row, i) => {
      const masked = row.map((v, j) => (causal && j > i ? -1e9 : v * mult));
      return rowSoftmax(masked);
    });
    const lastRow = weights[weights.length - 1];
    const ent = -lastRow.reduce((s, p) => (p > 1e-12 ? s + p * Math.log2(p) : s), 0);

    const pTraces: PlotTrace[] = [
      {
        type: 'heatmap',
        x: tokens,
        y: tokens,
        z: weights,
        zmin: 0,
        zmax: 1,
        colorscale: 'Plasma',
        showscale: true,
      },
    ];
    return { traces: pTraces, entropy0: ent };
  }, [causal, scaleBySqrtDk, dk]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
          Bật Causal Mask (GPT Decoder — cấm nhìn tương lai)
        </label>
        <label style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <input type="checkbox" checked={scaleBySqrtDk} onChange={(e) => setScaleBySqrtDk(e.target.checked)} />
          Chia √d_k (Tắt để thấy bão hòa Softmax)
        </label>
        <label>
          Chiều d_k: <strong>{dk}</strong>
          <input type="range" min={8} max={256} step={8} value={dk} onChange={(e) => setDk(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Key tokens (k)" yTitle="Query tokens (q)" height={310} layout={{ yaxis: { autorange: 'reversed' } }} />
      <p className="sim-caption">
        Entropy phân phối chú ý ở token cuối = <strong>{fmt(entropy0, 3)} bits</strong>.{' '}
        {!scaleBySqrtDk && dk >= 64
          ? '⚠️ Bão hòa Softmax! Không chia √d_k khiến tích vô hướng phóng đại, đẩy xác suất dồn cục gần 1 và triệt tiêu gradient.'
          : causal
            ? 'Tam giác trên nghiêm ngặt bị triệt tiêu (= 0) đảm bảo tính nhân quả tự hồi quy.'
            : 'Bidirectional (BERT/PhoBERT): mỗi token đều tham chiếu tự do cả ngữ cảnh trái và phải.'}
      </p>
    </div>
  );
}

/** Unit 8.5: Giải mã LLM (Temperature & Top-k) + RAG Cosine Similarity */
function SimLlmDecodingRag() {
  const [temp, setTemp] = useState(0.8);
  const [topK, setTopK] = useState(5);

  const { decTraces, ragTraces, entropy } = useMemo(() => {
    const vocab = ['mô_hình', 'thuật_toán', 'mạng', 'dữ_liệu', 'hàm', 'tham_số', 'biến', 'nhiễu'];
    const rawLogits = [3.1, 2.7, 2.1, 1.8, 1.0, 0.4, -0.2, -1.0];
    const scaled = rawLogits.map((z) => z / temp);
    const keepMask = scaled.map((_, idx) => idx < topK);
    const masked = scaled.map((v, idx) => (keepMask[idx] ? v : -1e9));
    const probs = rowSoftmax(masked).map((p, idx) => (keepMask[idx] ? p : 0));
    const ent = -probs.reduce((s, p) => (p > 1e-12 ? s + p * Math.log2(p) : s), 0);

    const dTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: vocab,
        y: probs,
        name: `P(token) (T=${temp.toFixed(1)}, top-${topK})`,
        marker: { color: probs.map((_, i) => (i < topK ? '#c084fc' : 'rgba(255,255,255,0.12)')) },
      },
    ];
    const docs = ['Doc0: Transformer PE', 'Doc1: PhoBERT Vi', 'Doc2: Word2Vec SGNS', 'Doc3: K-means', 'Doc4: XGBoost'];
    const sims = [0.91, 0.86, 0.62, 0.18, 0.12];
    const rTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: docs,
        y: sims,
        name: 'Cosine similarity với câu hỏi NLP',
        marker: { color: sims.map((s) => (s > 0.7 ? '#2dd4bf' : '#64748b')) },
      },
    ];
    return { decTraces: dTraces, ragTraces: rTraces, entropy: ent };
  }, [temp, topK]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Nhiệt độ giải mã (Temperature T): <strong>{fmt(temp, 1)}</strong>
          <input type="range" min={0.2} max={2.0} step={0.1} value={temp} onChange={(e) => setTemp(+e.target.value)} />
        </label>
        <label>
          Bộ lọc Top-k: <strong>{topK}</strong> từ
          <input type="range" min={1} max={8} step={1} value={topK} onChange={(e) => setTopK(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={decTraces} xTitle="Từ ứng viên tiếp theo" yTitle="Xác suất sinh P(w)" height={300} layout={{ yaxis: { range: [0, 1] } }} />
        <Chart traces={ragTraces} xTitle="Tài liệu RAG trong Vector Store" yTitle="Cosine Similarity" height={300} layout={{ yaxis: { range: [0, 1] } }} />
      </div>
      <p className="sim-caption">
        Entropy sinh token: <strong>{fmt(entropy, 3)} bits</strong>. Temperature thấp + Top-k hẹp giúp câu trả lời bám chắc thực tế trong bài toán RAG, trong khi Temperature cao tăng sáng tạo.
      </p>
    </div>
  );
}

/** Unit 9.1: Phổ Đồ thị & Hiện tượng Over-smoothing (Năng lượng Dirichlet giảm theo độ sâu) */
function SimGraphLaplacian() {
  const [depth, setDepth] = useState(6);
  const [useResidual, setUseResidual] = useState(false);

  const { energyTraces, stateTraces, finalEnergy } = useMemo(() => {
    // 6-node barbell-like graph: triangle (0,1,2) connected via (2,3) to triangle (3,4,5)
    const A: number[][] = [
      [1, 1, 1, 0, 0, 0],
      [1, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 1],
      [0, 0, 0, 1, 1, 1],
    ];
    const deg = A.map((r) => r.reduce((s, v) => s + v, 0));
    const Ahat = A.map((row, i) => row.map((v, j) => (v > 0 ? v / Math.sqrt(deg[i] * deg[j]) : 0)));

    let h = [-1.5, -1.2, -0.6, 0.7, 1.3, 1.6];
    const dirichlet = (vec: number[]) => {
      let e = 0;
      for (let i = 0; i < 6; i++) {
        for (let j = i + 1; j < 6; j++) {
          if (i !== j && A[i][j] > 0) e += (vec[i] - vec[j]) ** 2;
        }
      }
      return e;
    };

    const e0 = dirichlet(h) || 1;
    const energies = [1.0];
    for (let l = 1; l <= depth; l++) {
      const next = Ahat.map((row) => row.reduce((s, w, j) => s + w * h[j], 0));
      h = useResidual ? next.map((v, i) => 0.65 * v + 0.35 * h[i]) : next;
      energies.push(dirichlet(h) / e0);
    }

    const xs = Array.from({ length: depth + 1 }, (_, i) => i);
    const eTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: xs,
        y: energies,
        name: useResidual ? 'GCN + Residual Skip' : 'GCN thuần (Over-smoothing)',
        line: { color: useResidual ? '#2dd4bf' : '#ff9b6a', width: 3 },
      },
    ];
    const sTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: ['N0 (Cụm A)', 'N1 (Cụm A)', 'N2 (Cầu A)', 'N3 (Cầu B)', 'N4 (Cụm B)', 'N5 (Cụm B)'],
        y: h,
        name: `Giá trị nhúng tại tầng L=${depth}`,
        marker: { color: ['#42d3e5', '#42d3e5', '#42d3e5', '#c084fc', '#c084fc', '#c084fc'] },
      },
    ];
    return { energyTraces: eTraces, stateTraces: sTraces, finalEnergy: energies[energies.length - 1] };
  }, [depth, useResidual]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số tầng GCN L: <strong>{depth}</strong>
          <input type="range" min={0} max={15} step={1} value={depth} onChange={(e) => setDepth(+e.target.value)} />
        </label>
        <label style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <input type="checkbox" checked={useResidual} onChange={(e) => setUseResidual(e.target.checked)} />
          Thêm Residual Connection (Giữ tín hiệu riêng từng cụm)
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={energyTraces} xTitle="Độ sâu tầng GCN (L)" yTitle="Năng lượng Dirichlet chuẩn hóa" height={300} layout={{ yaxis: { range: [0, 1.05] } }} />
        <Chart traces={stateTraces} xTitle="Các nút trên đồ thị 2 cụm" yTitle="Đặc trưng H^(L)" height={300} layout={{ yaxis: { range: [-1.8, 1.8] } }} />
      </div>
      <p className="sim-caption">
        Năng lượng Dirichlet còn lại tại L = {depth}: <strong>{fmt(finalEnergy * 100, 1)}%</strong>. Không có kết nối phần dư, các lớp GCN liên tiếp làm mượt toàn bộ tín hiệu qua cầu (N2–N3), khiến hai cụm mất dần ranh giới phân tách.
      </p>
    </div>
  );
}

/** Unit 9.2: Khung Message Passing (MPNN) & Trường tiếp nhận k-hop trên đồ thị dây chuyền */
function SimMpnnReceptive() {
  const [hops, setHops] = useState(2);

  const { traces, activeCount } = useMemo(() => {
    const nodes = ['A', 'B', 'C', 'D', 'E', 'F'];
    const n = nodes.length;
    // Undirected path A-B-C-D-E-F with self-loops, initial impulse at node E (index 4)
    let state = [0.0, 0.0, 0.0, 0.0, 8.0, 0.0];
    for (let h = 0; h < hops; h++) {
      const next = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        const nbrs = [i];
        if (i > 0) nbrs.push(i - 1);
        if (i < n - 1) nbrs.push(i + 1);
        next[i] = nbrs.reduce((s, j) => s + state[j], 0) / nbrs.length;
      }
      state = next;
    }

    const pTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: nodes,
        y: state,
        name: `Trạng thái nút sau ${hops}-hop Mean Aggregation`,
        marker: { color: state.map((v) => (v > 0.01 ? '#2dd4bf' : '#475569')) },
      },
    ];
    return { traces: pTraces, activeCount: state.filter((v) => v > 1e-6).length };
  }, [hops]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Số bước Message Passing (k-hop): <strong>{hops}</strong>
          <input type="range" min={0} max={5} step={1} value={hops} onChange={(e) => setHops(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Đồ thị dây chuyền A — B — C — D — E — F (Xung ban đầu tại E = 8)" yTitle="Giá trị trạng thái nút" height={300} layout={{ yaxis: { range: [0, 8.5] } }} />
      <p className="sim-caption">
        Số nút đã nhận tín hiệu từ nút E = <strong>{activeCount}/6</strong>. Với mỗi lớp MPNN đồng bộ, trường tiếp nhận (receptive field) lan rộng thêm đúng 1 bước cạnh trên đồ thị (cùng cơ chế với phòng lab GNN A—B—C).
      </p>
    </div>
  );
}

/** Unit 9.3: So sánh GCN (Trọng số bậc cố định) vs GAT (Chú ý thích nghi đặc trưng) */
function SimGcnVsGat() {
  const [attnGain, setAttnGain] = useState(3.0);

  const { weightTraces, projTraces, gatCos, gcnCos } = useMemo(() => {
    const names = ['N0 (Tâm)', 'N1 (Cùng lớp)', 'N2 (Cùng lớp)', 'N3 (Nhiễu khác lớp)', 'N4 (Nhiễu khác lớp)'];
    const feats = [
      [1.0, 0.8],
      [1.1, 0.7],
      [0.9, 1.0],
      [-1.0, -0.9],
      [-0.8, -1.1],
    ];
    const wGcn = feats.map(() => 1 / feats.length);
    // Feature similarity logit for GAT relative to center node N0
    const logits = feats.map(([x, y]) => attnGain * (x * feats[0][0] + y * feats[0][1]));
    const wGat = rowSoftmax(logits);

    const agg = (w: number[]) => [
      w.reduce((s, wi, i) => s + wi * feats[i][0], 0),
      w.reduce((s, wi, i) => s + wi * feats[i][1], 0),
    ];
    const hGcn = agg(wGcn);
    const hGat = agg(wGat);
    const cos = (v: number[]) => {
      const nv = Math.hypot(v[0], v[1]) || 1;
      const n0 = Math.hypot(feats[0][0], feats[0][1]) || 1;
      return (v[0] * feats[0][0] + v[1] * feats[0][1]) / (nv * n0);
    };

    const wTraces: PlotTrace[] = [
      { type: 'bar', x: names, y: wGcn, name: 'GCN (Cố định theo bậc)', marker: { color: '#64748b' } },
      { type: 'bar', x: names, y: wGat, name: `GAT (Masked Softmax α_0j)`, marker: { color: '#2dd4bf' } },
    ];
    const pTraces: PlotTrace[] = [
      {
        type: 'scatter',
        mode: 'markers+text',
        x: feats.map((f) => f[0]),
        y: feats.map((f) => f[1]),
        text: ['N0', 'N1', 'N2', 'N3 (nhiễu)', 'N4 (nhiễu)'],
        textposition: 'top center',
        name: 'Đặc trưng lân cận',
        marker: { size: 9, color: ['#ffd173', '#42d3e5', '#42d3e5', '#ff9b6a', '#ff9b6a'] },
      },
      {
        type: 'scatter',
        mode: 'markers+text',
        x: [hGcn[0], hGat[0]],
        y: [hGcn[1], hGat[1]],
        text: ['Tổng hợp GCN', 'Tổng hợp GAT'],
        textposition: 'bottom right',
        name: 'Kết quả sau Message Passing',
        marker: { size: 12, color: ['#94a3b8', '#2dd4bf'], symbol: 'diamond' },
      },
    ];
    return { weightTraces: wTraces, projTraces: pTraces, gatCos: cos(hGat), gcnCos: cos(hGcn) };
  }, [attnGain]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Độ nhạy Attention của GAT (α_ij): <strong>{fmt(attnGain, 1)}</strong>
          <input type="range" min={0.0} max={5.0} step={0.25} value={attnGain} onChange={(e) => setAttnGain(+e.target.value)} />
        </label>
      </div>
      <div className="grad-sim-grid">
        <Chart traces={weightTraces} xTitle="Tập lân cận N(0) ∪ {0}" yTitle="Trọng số tổng hợp" height={300} layout={{ barmode: 'group', yaxis: { range: [0, 0.7] } }} />
        <Chart traces={projTraces} xTitle="Đặc trưng 1" yTitle="Đặc trưng 2" height={300} layout={{ yaxis: { scaleanchor: 'x' } }} />
      </div>
      <p className="sim-caption">
        Cosine giữ hướng đúng của N0: GCN = <strong>{fmt(gcnCos, 3)}</strong> (bị 2 nút nhiễu kéo lệch về gần gốc 0) vs GAT ={' '}
        <strong>{fmt(gatCos, 3)}</strong> (tự hạ trọng số nút nhiễu về sát 0).
      </p>
    </div>
  );
}

/** Unit 9.4: Rò rỉ thời gian (Random Split) vs Temporal Split hợp lệ trên Đồ thị động */
function SimTemporalLeakage() {
  const [futureDrift, setFutureDrift] = useState(0.35);

  const { traces, randomAuc, temporalAuc } = useMemo(() => {
    const splits = ['ROC-AUC Báo cáo', 'Hiệu năng thực trên luồng mới'];
    const rApparent = Math.min(0.99, 0.91 + 0.15 * futureDrift);
    const rReal = Math.max(0.52, 0.86 - 0.55 * futureDrift);
    const tApparent = Math.max(0.55, 0.88 - 0.3 * futureDrift);
    const tReal = Math.max(0.54, 0.87 - 0.32 * futureDrift);

    const pTraces: PlotTrace[] = [
      {
        type: 'bar',
        x: splits,
        y: [rApparent, rReal],
        name: 'Random Edge Split (Rò rỉ tương lai!)',
        marker: { color: '#ff9b6a' },
      },
      {
        type: 'bar',
        x: splits,
        y: [tApparent, tReal],
        name: 'Temporal Split t_train < t_test (Chuẩn)',
        marker: { color: '#2dd4bf' },
      },
    ];
    return { traces: pTraces, randomAuc: rApparent, temporalAuc: tReal };
  }, [futureDrift]);

  return (
    <div className="grad-sim-box">
      <div className="grad-sim-controls">
        <label>
          Tốc độ dịch phân phối theo thời gian (Concept Drift): <strong>{fmt(futureDrift, 2)}</strong>
          <input type="range" min={0.05} max={0.6} step={0.05} value={futureDrift} onChange={(e) => setFutureDrift(+e.target.value)} />
        </label>
      </div>
      <Chart traces={traces} xTitle="Phương thức đánh giá" yTitle="ROC-AUC Link Prediction" height={300} layout={{ barmode: 'group', yaxis: { range: [0.4, 1.02] } }} />
      <p className="sim-caption">
        Random Split báo cáo ảo <strong>{fmt(randomAuc, 3)}</strong> do lén nhìn các cạnh tương lai làm ngữ cảnh, trong khi đánh giá Temporal chuẩn phản ánh trung thực khả năng tổng quát <strong>{fmt(temporalAuc, 3)}</strong> khi triển khai thực tế.
      </p>
    </div>
  );
}

export function GradSimulationM89({ id }: { id: GradSimulationId }) {
  if (id === 'm8-u1') return <SimWord2VecAnalogy />;
  if (id === 'm8-u2') return <SimLstmVanishing />;
  if (id === 'm8-u3') return <SimBahdanauAlign />;
  if (id === 'm8-u4') return <SimTransformerAttention />;
  if (id === 'm8-u5') return <SimLlmDecodingRag />;
  if (id === 'm9-u1') return <SimGraphLaplacian />;
  if (id === 'm9-u2') return <SimMpnnReceptive />;
  if (id === 'm9-u3') return <SimGcnVsGat />;
  if (id === 'm9-u4') return <SimTemporalLeakage />;
  return null;
}
