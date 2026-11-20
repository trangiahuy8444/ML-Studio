import {SimulationStudio,SimulationPanel,SimulationCode} from './SimulationStudio';
import {Transport,usePlayback} from './ObservationControls';
import {useCallback, useState} from 'react';
import {adjacency, graphTrace} from '../lib/graphTrace';
import type {Aggregator, Edge} from '../lib/graphTrace';
import {Notebook} from './Notebook';
import graphBook from '../data/gnn-foundations.json';
import projectBook from '../data/gnn-project.json';
import results from '../data/gnn-results.json';
import type {NotebookData} from '../types';

const labels = ['A','B','C','D','E'];
const positions = [[52,98],[155,54],[262,98],[255,192],[90,194]];
const candidateEdges: Edge[] = labels.flatMap((_,i) => labels.slice(i+1).map((__,j) => [i,i+j+1] as const));
const keyOf = ([i,j]: Edge) => `${i}-${j}`;
const initialEdges = ['0-1','1-2','2-3','3-4'];
const initialFeatures = [2,4,8,6,1];
const fmt = (n: number) => n.toLocaleString('vi-VN',{maximumFractionDigits:4});
const lessons = [
  {title:'Dữ liệu đồ thị', ref:'§1.1 và §2.3', question:'Nếu thêm một trích dẫn, ô nào trong ma trận kề thay đổi?',
    task:'Thêm cạnh A–C và giải thích vì sao hai ô đối xứng đổi. So sánh dự đoán nhãn một nút, một cạnh và toàn đồ thị.'},
  {title:'Truyền thông điệp', ref:'Ch. 4 và §11.1', question:'Sau bao nhiêu lượt thì C có thể ảnh hưởng đến A?',
    task:'Giữ A–B–C, chọn mean và bật self-loop. Tính tay đầu ra của A ở lượt 1 và lượt 2 rồi mở bảng từng đóng góp.'},
  {title:'GCN', ref:'§5.1.3', question:'Vì sao hai hàng xóm có bậc khác nhau mang hệ số khác nhau?',
    task:'Tính tay một ô của D⁻¹ᐟ²(A+I)D⁻¹ᐟ², rồi so với mean trên cùng đồ thị. Thử nhiều lượt và theo dõi các biểu diễn có thể tiến gần nhau.'},
  {title:'GAT', ref:'§7.1', question:'Thêm một hàng xóm có đổi trọng số của hàng xóm cũ không?',
    task:'Chọn A, thêm cạnh A–C và kiểm tra tổng các hệ số trên đúng tập hàng xóm. Attention mô tả phép tính này, chưa chứng minh cạnh gây ra dự đoán.'},
  {title:'Dự án phân loại nút', ref:'Ch. 15', question:'Cạnh trích dẫn có giúp mô hình dự đoán chủ đề bài báo?',
    task:'So MLP, GCN và GCN chỉ dùng self-loop. Viết một kết luận dựa vào macro-F1 và chỉ ra một nhóm lỗi.'},
  {title:'Đồ thị động', ref:'§9.1, §9.4 và §12.3–12.4', question:'Ta được nhìn những cạnh nào khi dự đoán liên kết của tháng sau?',
    task:'Vẽ trục thời gian, định nghĩa cạnh dương/âm và chỉ ra hai cách rò rỉ. Đổi mốc quan sát để kiểm tra thiết kế của em.'},
];

const foundationNotebook = graphBook as unknown as NotebookData;
const projectNotebook = projectBook as unknown as NotebookData;

type GraphInput = {edges: string[]; features: number[]};
function GraphLab({lesson, input, update}: {lesson: number; input: GraphInput; update: (value: GraphInput) => void}) {
  const {edges, features} = input;
  const [node,setNode] = useState(0);
  const [loops,setLoops] = useState(true);

  const [mean,setMean] = useState(true);
  const [guess,setGuess] = useState('');
  const player = usePlayback(5,`${edges.join(',')}|${features.join(',')}|${loops}|${mean}`);
  const round = player.index;
  const mode: Aggregator = lesson===3 ? 'gat' : lesson===2 ? 'gcn' : mean ? 'mean' : 'sum';
  const weight = lesson>=2 ? .8 : 1;
  const selectedEdges = candidateEdges.filter(edge => edges.includes(keyOf(edge)));
  const original = adjacency(labels.length,selectedEdges,false);
  const effective = adjacency(labels.length,selectedEdges,lesson>0 && loops);
  let state = features;
  for(let i=1;i<round;i++) state = graphTrace(effective,state,mode,weight).output;
  const trace = graphTrace(effective,state,mode,weight);
  const chosen = trace.rows[node];
  const reset = () => {player.seek(0);setGuess('');};
  const predicted=guess.trim()?Number(guess.trim().replace(',','.')):NaN;
  const feedback=round===1&&Number.isFinite(predicted)?`Dự đoán lệch ${fmt(Math.abs(predicted-chosen.output))} so với phép tính.`:'';
  function toggle(edge: Edge) {
    const key = keyOf(edge);
    update({...input, edges: edges.includes(key) ? edges.filter(value => value!==key) : [...edges,key]});
    reset();
  }

  const aggregate=mode==='gat'?'score = LeakyReLU(a @ concat(Wh_i, Wh_j)); alpha = softmax(score)':mode==='gcn'?'alpha = 1 / sqrt(degree[i] * degree[j])':mode==='mean'?'alpha = 1 / len(neighbors)':'alpha = 1';
  return <SimulationStudio title={lesson===0?'Cấu trúc đồ thị':'Dữ liệu đi qua các nút'}>
    {lesson>0&&<Transport player={player} count={5}/>}
    <div className="gnn-lab">
    <div className="gnn-graph">
      <svg viewBox="0 0 315 240" role="img" aria-label="Sơ đồ năm bài báo và các cạnh trích dẫn đang bật">
        {selectedEdges.map(([i,j]) => <line key={keyOf([i,j])} x1={positions[i][0]} y1={positions[i][1]} x2={positions[j][0]} y2={positions[j][1]} className={i===node||j===node?'selected':''}/>)}
        {positions.map(([x,y],i) => <g key={i}>
          <circle cx={x} cy={y} r="23" className={i===node?'selected':''}/><text x={x} y={y+5}>{labels[i]}</text>
          <text x={x} y={y+39} className="gnn-value">{fmt(round ? trace.output[i] : features[i])}</text>
        </g>)}
      </svg>
      <p>Trích dẫn giả lập, biểu diễn bằng cạnh vô hướng. Số dưới nút là {round ? `đầu ra lượt ${round}` : 'đặc trưng đầu vào'}.</p>
      <fieldset className="gnn-edges"><legend>Bật hoặc tắt cạnh</legend>{candidateEdges.map(edge =>
        <label key={keyOf(edge)}><input type="checkbox" checked={edges.includes(keyOf(edge))} onChange={()=>toggle(edge)}/>{labels[edge[0]]}–{labels[edge[1]]}</label>)}</fieldset>
    </div>
    <div className="gnn-work">
      <div className="gnn-controls">
        <label>Nút quan sát <select value={node} onChange={event=>{setNode(Number(event.target.value));setGuess('');}}>{labels.map((label,i)=><option key={label} value={i}>{label}</option>)}</select></label>
        <label>Đặc trưng nút {labels[node]} <input type="range" min={0} max={10} value={features[node]} onChange={event=>{const value=Number(event.target.value);update({...input, features: features.map((v,i)=>i===node?value:v)});reset();}}/><output>{features[node]}</output></label>
      </div>
      <table className="gnn-matrix">
        <caption>{lesson===0?'Ma trận kề A':loops?'Ma trận dùng để truyền tin (A + I)':'Ma trận dùng để truyền tin (A)'}</caption>
        <thead><tr><th scope="col">Nút</th>{labels.map(label=><th key={label} scope="col">{label}</th>)}<th scope="col">Bậc</th></tr></thead>
        <tbody>{(lesson===0?original:effective).map((row,i)=><tr key={i} className={i===node?'selected':''}><th scope="row">{labels[i]}</th>{row.map((v,j)=><td key={j}>{v}</td>)}<td>{trace.degrees[i]}</td></tr>)}</tbody>
      </table>
      {lesson===0 ? <p>Hàng xóm của {labels[node]}: {chosen.messages.map(m=>labels[m.source]).join(', ')||'chưa có'}.</p> : <>
        <div className="gnn-controls">
          {lesson===1 && <label>Tổng hợp <select value={mean?'mean':'sum'} onChange={e=>{setMean(e.target.value==='mean');reset();}}><option value="mean">Trung bình</option><option value="sum">Tổng</option></select></label>}
          <label><input type="checkbox" checked={loops} onChange={e=>{setLoops(e.target.checked);reset();}}/> Bật self-loop</label>

        </div>
        {lesson===2 && <p>Hệ số j → i bằng 1/√(dᵢdⱼ). Phép chiếu dùng W = 0,8 đặt sẵn.</p>}
        {lesson===3 && <p>Chiếu với W = 0,8. Điểm cạnh = LeakyReLU(0,6 Whᵢ − 0,3 Whⱼ), sau đó softmax trên các hàng xóm của i. Tham số đặt sẵn.</p>}
        {round===0 && <label className="gnn-guess">Em đoán đầu ra lượt 1 của nút {labels[node]} <input inputMode="decimal" value={guess} onChange={e=>setGuess(e.target.value)} placeholder="VD 3,5"/></label>}
        {round>0 && <div className="gnn-result" aria-live="polite">
          <h3>Nút {labels[node]} ở lượt {round}: {fmt(chosen.output)}</h3>
          {feedback && <p>{feedback}</p>}
          <p>Các nút cùng đọc trạng thái của lượt trước. Nút cô lập khi tắt self-loop nhận tổng bằng 0.</p>
          <table className="gnn-matrix gnn-messages">
            <caption>Từng đóng góp vào nút {labels[node]}</caption>
            <thead><tr><th scope="col">Nguồn</th><th scope="col">Giá trị trước</th>{mode==='gat'&&<th scope="col">Điểm</th>}<th scope="col">Hệ số</th><th scope="col">Đóng góp</th></tr></thead>
            <tbody>{chosen.messages.map(m=><tr key={m.source}><th scope="row">{labels[m.source]}</th><td>{fmt(state[m.source])}</td>{mode==='gat'&&<td>{fmt(m.score)}</td>}<td>{fmt(m.coefficient)}</td><td>{fmt(m.contribution)}</td></tr>)}</tbody>
          </table>
        </div>}
      </>}
    </div>
  </div>
  {lesson>0&&<div className="simulation-studio-grid"><SimulationPanel title="Python tương ứng">
    <SimulationCode lines={['neighbors = adjacency[i].nonzero()[0]',aggregate,'h_next[i] = sum(alpha * W * h[neighbors])','h = h_next.copy()']} active={round>0?3:-1}
      detail={round>0?`Lượt ${round} · ${chosen.messages.length} đóng góp vào nút ${labels[node]} · Đầu ra ${fmt(chosen.output)}`:'Bấm Từng bước để cập nhật đồng thời các nút.'}/>
  </SimulationPanel></div>}
  </SimulationStudio>;
}

function ResearchTimeline() {
  const [cutoff,setCutoff] = useState(1);
  const months = ['Tháng 1','Tháng 2','Tháng 3','Tháng 4'];
  return <SimulationStudio title="Mốc quan sát và cửa sổ dự đoán">
    <label>Mốc quan sát <select value={cutoff} onChange={e=>setCutoff(Number(e.target.value))}>{months.slice(0,3).map((m,i)=><option key={m} value={i}>{m}</option>)}</select></label>
    <ol className="gnn-timeline">{months.map((m,i)=><li key={m}><strong>{m}</strong><span>{i<=cutoff?'Được quan sát':i===cutoff+1?'Cửa sổ dự đoán':'Chưa dùng'}</span></li>)}</ol>
    <p>Mục tiêu ví dụ: dự đoán cạnh mới xuất hiện trong tháng kế tiếp. Với đích này, chỉ coi là âm những cặp đủ điều kiện ở mốc quan sát và không xuất hiện trong cửa sổ đích.</p>
    <details><summary>Xem hai lỗi thường gặp</summary><p>Tính embedding từ đồ thị chứa cạnh tương lai. Chọn mẫu âm bằng cách nhìn qua cửa sổ đích sang các tháng sau.</p></details>
  </SimulationStudio>;
}

export function GraphLearning({onBusy,openDay}: {onBusy:(busy:boolean)=>void;openDay:(day:number)=>void}) {
  const [lesson,setLesson] = useState(0);
  const [input, setInput] = useState<GraphInput>({edges:initialEdges,features:initialFeatures});
  const [busy, setBusy] = useState(false);
  const handleBusy = useCallback((value: boolean) => {setBusy(value); onBusy(value);},[onBusy]);
  const current = lessons[lesson];
  return <>
    <header className="page-heading gnn-heading">
      <span className="eyebrow">Học tiếp sau Ngày 30</span>
      <h1>Mạng nơ-ron đồ thị</h1>
      <p>Nếu hai bài báo giống nhau về nội dung nhưng được trích dẫn bởi hai nhóm khác nhau, mô hình có thể học được gì từ các mối liên hệ đó?</p>
      <a href="https://drive.google.com/file/d/1gVj-naMcospKctRAnVqaKRySssQAo2LS/view?usp=drivesdk" target="_blank" rel="noreferrer">Xem sách của Liu và Zhou</a>
    </header>
    <nav className="gnn-lessons" aria-label="Các buổi GNN">{lessons.map((item,index)=><button className="btn-outline" key={item.title} aria-current={index===lesson?'step':undefined} disabled={busy} onClick={()=>setLesson(index)}>{index+1}. {item.title}</button>)}</nav>
    <section className="panel gnn-session">
      <h2>{current.question}</h2><p>Đọc {current.ref} trong sách. {current.task}</p>
      <details><summary>Ôn lại kiến thức đã học</summary><div className="gnn-controls">{[[3,'Ma trận'],[19,'Forward'],[20,'Backprop'],[22,'CNN'],[24,'Embedding'],[25,'Attention']].map(([id,title])=><button key={id} className="btn-outline" onClick={()=>openDay(Number(id))}>{title}</button>)}</div></details>
    </section>
    {lesson<4 ? <GraphLab key={lesson} lesson={lesson} input={input} update={setInput}/> : lesson===4 ?
      <section className="panel">
        <h2>Kết quả chạy trên ba seed</h2><p>Dữ liệu giả lập. Test gồm các đồ thị tách biệt. Các mô hình dùng cùng kiến trúc đầu phân loại và cùng phân hoạch dữ liệu trong mỗi seed.</p>
        <div className="table-wrap"><table>
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Macro-F1 test</th><th scope="col">Accuracy test</th></tr></thead>
          <tbody>{results.summary.map(row=><tr key={row.model}><th scope="row">{row.model}</th><td>{fmt(row.macro_f1)}</td><td>{fmt(row.accuracy)}</td></tr>)}</tbody>
        </table></div>
        <details><summary>Ma trận nhầm lẫn theo seed</summary>
          {results.runs.map(run=><p key={run.seed}>Seed {run.seed}: {run.models.map(model=>`${model.model} [${model.confusion.map(row=>row.join(', ')).join(' / ')}]`).join(' · ')}</p>)}
          <p>Hàng là lớp đúng, cột là lớp dự đoán.</p>
        </details>
      </section> : <ResearchTimeline/>}
    {lesson<5 && <Notebook key={lesson<4?'gnn-foundations':'gnn-project'} book={lesson<4?foundationNotebook:projectNotebook}
      bookId={lesson<4?'gnn-foundations':'gnn-project'} title={lesson<4?'GNN · Tính tay và đối chiếu bằng NumPy':'GNN · Dự án phân loại nút'} onBusy={handleBusy}/>}
  </>;
}
