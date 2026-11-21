import {useMemo,useState} from 'react';
import type {Route} from '../types';
import {Bars,LineChart,ProjectFrame,Slider,Stat,fmt,pct} from './ui';
import type {StepDef} from './ui';

// 14 bài báo giả lập. topic 1 = thị giác máy tính, 0 = xử lý ngôn ngữ.
// feature = tỉ lệ từ khóa về ảnh trong tóm tắt; vài bài cố tình gây nhiễu (bài 4, 9, 12).
const NODES:{title:string;topic:0|1;feature:number;x:number;y:number;known?:boolean}[]=[
  {title:'CNN nhận diện ảnh',topic:1,feature:.86,x:90,y:80,known:true},
  {title:'Phát hiện vật thể',topic:1,feature:.78,x:200,y:50},
  {title:'Phân đoạn ảnh y tế',topic:1,feature:.81,x:160,y:160},
  {title:'Tăng cường dữ liệu ảnh',topic:1,feature:.72,x:60,y:200},
  {title:'Chú thích ảnh tự động',topic:1,feature:.41,x:270,y:150},
  {title:'Ước lượng độ sâu',topic:1,feature:.69,x:130,y:260,known:true},
  {title:'Nhận diện khuôn mặt',topic:1,feature:.75,x:250,y:250},
  {title:'Dịch máy Attention',topic:0,feature:.18,x:470,y:70,known:true},
  {title:'Word2Vec tiếng Việt',topic:0,feature:.12,x:560,y:140},
  {title:'Hỏi đáp trên văn bản',topic:0,feature:.55,x:420,y:170},
  {title:'Phân tích cảm xúc',topic:0,feature:.22,x:540,y:250},
  {title:'Tóm tắt văn bản',topic:0,feature:.27,x:440,y:270,known:true},
  {title:'Nhận dạng tiếng nói',topic:0,feature:.58,x:620,y:60},
  {title:'Mô hình ngôn ngữ lớn',topic:0,feature:.15,x:640,y:220},
];
const EDGES:[number,number][]=[[0,1],[0,2],[0,3],[1,2],[1,4],[2,5],[3,5],[5,6],[2,6],[4,6],[7,8],[7,9],[8,10],[9,11],[10,11],[7,12],[8,13],[10,13],[12,13],[9,10],[4,9]];
const NOISE:[number,number][]=[[6,11],[3,12],[1,8],[5,10],[0,13],[2,9],[6,12],[3,7],[4,10],[4,12],[9,6],[12,2]];
const TOPIC=['Ngôn ngữ','Thị giác'],COL=['#f28db9','#42d3e5'];
const unknown=NODES.map((_,i)=>i).filter(i=>!NODES[i].known);

function propagate(edges:[number,number][],layers:number){
  const nb=NODES.map((_,i)=>[i]);edges.forEach(([a,b])=>{nb[a].push(b);nb[b].push(a)});
  const hist=[NODES.map(n=>n.feature)];
  for(let l=0;l<layers;l++){const h=hist[l];hist.push(nb.map(list=>list.reduce((s,j)=>s+h[j],0)/list.length))}
  return hist;
}
const accOf=(h:number[],thr=.5)=>unknown.filter(i=>((h[i]>thr)?1:0)===NODES[i].topic).length/unknown.length;

function Graph({values,edges,extra=[],guess,onNode,showValues=true}:{values?:number[];edges:[number,number][];extra?:[number,number][];guess?:(0|1|null)[];onNode?:(i:number)=>void;showValues?:boolean}){
  return <svg className="gnn-svg" viewBox="0 0 700 320">
    {edges.map(([a,b],i)=><line key={i} x1={NODES[a].x} y1={NODES[a].y} x2={NODES[b].x} y2={NODES[b].y} stroke="var(--line-strong)" strokeWidth={2}/>)}
    {extra.map(([a,b],i)=><line key={'n'+i} x1={NODES[a].x} y1={NODES[a].y} x2={NODES[b].x} y2={NODES[b].y} stroke="#ffd173" strokeWidth={2} strokeDasharray="5 4"/>)}
    {NODES.map((n,i)=>{const v=values?.[i];const fill=n.known?COL[n.topic]:guess?(guess[i]===null?'var(--surface-3)':COL[guess[i]!]):v!==undefined?`color-mix(in srgb, ${COL[1]} ${Math.round(v*100)}%, ${COL[0]})`:'var(--surface-3)';
      return <g key={i} className={onNode&&!n.known?'gnn-node click':'gnn-node'} onClick={()=>!n.known&&onNode?.(i)}>
        <circle cx={n.x} cy={n.y} r={n.known?19:16} fill={fill} stroke={n.known?'#fff':'var(--bg)'} strokeWidth={n.known?3:2}/>
        <text x={n.x} y={n.y+4} textAnchor="middle" className="gnn-num">{showValues&&v!==undefined?fmt(v):i+1}</text>
        <text x={n.x} y={n.y+33} textAnchor="middle" className="gnn-title">{n.title}</text></g>})}
  </svg>;
}

export function GnnProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [step,setStep]=useState(0);
  const [guess,setGuess]=useState<(0|1|null)[]>(()=>NODES.map(()=>null));
  const [revealed,setRevealed]=useState(false);
  const [thr,setThr]=useState(.5),[layers,setLayers]=useState(1),[noise,setNoise]=useState(0),[layers4,setLayers4]=useState(2);
  const clean=useMemo(()=>propagate(EDGES,40),[]);
  const noisy=useMemo(()=>propagate([...EDGES,...NOISE.slice(0,noise)],8),[noise]);
  const myAcc=unknown.filter(i=>guess[i]===NODES[i].topic).length;
  const byLayer=clean.map(h=>accOf(h));
  const noiseCurve=useMemo(()=>NOISE.map((_,k)=>accOf(propagate([...EDGES,...NOISE.slice(0,k+1)],2)[2])),[]);
  const gap=(h:number[])=>{const m=(t:0|1)=>{const ids=NODES.map((_,i)=>i).filter(i=>NODES[i].topic===t);return ids.reduce((s,i)=>s+h[i],0)/ids.length};return m(1)-m(0)};

  const steps:StepDef[]=[
    {label:'Em đoán chủ đề',title:'Chỉ biết chủ đề của 4 bài. Đoán 10 bài còn lại.',
      intro:<p>Mỗi vòng tròn là một bài báo, đường nối là trích dẫn. Bốn bài viền trắng đã biết chủ đề. <b>Bấm vào</b> các bài còn lại để đoán: bấm lần lượt đổi giữa Thị giác, Ngôn ngữ, chưa đoán.</p>,
      controls:<><div className="stat-row"><Stat label="Đã đoán" value={`${unknown.filter(i=>guess[i]!==null).length}/10`}/>{revealed&&<Stat label="Em đúng" value={`${myAcc}/10`}/>}</div>
        <button className="btn-outline" onClick={()=>setRevealed(r=>!r)}>{revealed?'Ẩn đáp án':'Xem đáp án'}</button>
        {revealed&&<p className="small-note">Đáp án: {unknown.map(i=>`${i+1}: ${TOPIC[NODES[i].topic]}`).join(' · ')}</p>}</>,
      visual:<><Graph edges={EDGES} guess={guess} showValues={false} onNode={i=>setGuess(g=>g.map((v,j)=>j===i?(v===null?1:v===1?0:null):v))}/><p className="chart-caption"><span style={{color:COL[1]}}>●</span> Thị giác · <span style={{color:COL[0]}}>●</span> Ngôn ngữ · xám: chưa đoán</p></>,
      takeaway:'Em đoán bằng cách nhìn hàng xóm: bài trích dẫn nhiều bài thị giác thì có lẽ cũng là thị giác. Đó chính là giả định đồng loại (homophily) mà GNN dựa vào.'},
    {label:'Chỉ nhìn nội dung',title:'Bỏ qua trích dẫn, chỉ đọc tóm tắt.',
      intro:<p>Mỗi bài có một con số: tỉ lệ từ khóa về ảnh trong tóm tắt. Lớn hơn ngưỡng thì đoán Thị giác. Màu nút là con số đó, từ hồng (0) sang xanh (1).</p>,
      controls:<><Slider label="Ngưỡng" value={thr} min={.1} max={.9} step={.02} onChange={setThr} format={v=>fmt(v)}/>
        <div className="stat-row"><Stat label="Đúng (10 bài chưa biết)" value={pct(accOf(clean[0],thr))}/></div>
        <p className="small-note">"Chú thích ảnh tự động" dùng nhiều từ ngôn ngữ; "Hỏi đáp" và "Tiếng nói" có nhiều từ về tín hiệu. Chỉ nhìn nội dung thì khó phân loại đúng các bài này.</p></>,
      visual:<Graph edges={EDGES} values={clean[0]}/>,
      takeaway:'Không ngưỡng nào phân loại đúng cả 10 bài, vì nội dung riêng của vài bài gây hiểu nhầm. Cần thêm thông tin từ bên ngoài bài báo.'},
    {label:'Truyền thông điệp',title:'Mỗi bài lấy trung bình với hàng xóm.',
      intro:<p>Một <b>lớp</b> GNN: mỗi nút thay giá trị của mình bằng trung bình của chính nó và các hàng xóm. Lặp nhiều lớp thì thông tin lan xa hơn. Số trên nút là giá trị sau từng lớp.</p>,
      controls:<><Slider label="Số lớp" value={layers} min={0} max={40} onChange={setLayers}/>
        <div className="stat-row"><Stat label="Đúng" value={pct(byLayer[layers])} tone={byLayer[layers]===1?'good':undefined}/><Stat label="Khoảng cách 2 chủ đề" value={fmt(gap(clean[layers]))} tone={gap(clean[layers])<.12?'warn':undefined}/></div></>,
      visual:<><Graph edges={EDGES} values={clean[layers]}/><LineChart height={140} xLabel="Số lớp" yLabel="Độ chính xác" series={[{name:'Độ chính xác',values:byLayer,color:'#84e0a8'},{name:'Khoảng cách giữa 2 chủ đề',values:clean.map(gap),color:'#ffd173',dashed:true}]}/></>,
      takeaway:`Chỉ 1 lớp đã sửa được các bài gây nhiễu nhờ hàng xóm. Càng nhiều lớp, giá trị mọi nút càng co về gần nhau: khoảng cách giữa hai chủ đề từ ${fmt(gap(clean[0]))} còn ${fmt(gap(clean[40]))} sau 40 lớp (over-smoothing). Đồ thị nhỏ này vẫn tách được, nhưng với GNN thật các nút trở nên gần như không phân biệt được và độ chính xác giảm, nên người ta thường chỉ dùng 2–3 lớp.`},
    {label:'Cạnh nhiễu',title:'Nếu trích dẫn chéo chủ đề thì sao?',
      intro:<p>Thêm các trích dẫn nối hai chủ đề (đường vàng nét đứt). Giả định "hàng xóm giống mình" yếu dần.</p>,
      controls:<><Slider label="Số cạnh nhiễu" value={noise} min={0} max={NOISE.length} onChange={setNoise}/><Slider label="Số lớp" value={layers4} min={0} max={4} onChange={setLayers4}/>
        <div className="stat-row"><Stat label="Đúng" value={pct(accOf(noisy[layers4]))}/></div></>,
      visual:<><Graph edges={EDGES} extra={NOISE.slice(0,noise)} values={noisy[layers4]}/><LineChart height={140} xLabel="Số cạnh nhiễu − 1" yLabel="Đúng (2 lớp)" series={[{name:'2 lớp',values:noiseCurve,color:'#f28db9'}]} yMin={0}/></>,
      takeaway:'GNN chỉ tốt khi cấu trúc đồ thị thật sự mang thông tin. Với đồ thị dị loại (hàng xóm thường khác mình), cần kiến trúc khác hoặc trọng số cạnh như GAT.'},
    {label:'Tổng kết',title:'Từ phép trung bình đến GNN thật.',
      intro:<p>GCN làm đúng điều em vừa thấy, chỉ thêm một ma trận trọng số học được và hàm kích hoạt sau mỗi lần lấy trung bình: <b>h′ = σ(W · trung bình(h của mình và hàng xóm))</b>.</p>,
      controls:<><p className="small-note">Học tiếp với dữ liệu trích dẫn thật và notebook chạy PyTorch trong trang GNN.</p><button className="btn-primary" onClick={()=>navigate({kind:'gnn'})}>Mở trang Mạng nơ-ron đồ thị</button></>,
      visual:<Bars rows={[{label:'Chỉ nội dung',value:accOf(clean[0])},{label:'1 lớp',value:byLayer[1]},{label:'2 lớp',value:byLayer[2]},{label:'2 lớp + 6 cạnh nhiễu',value:noiseCurve[5],color:'var(--gold)'},{label:'2 lớp + 12 cạnh nhiễu',value:noiseCurve.at(-1)!,color:'var(--pink)'}]} max={1} format={v=>pct(v)}/>,
      takeaway:'Quan hệ giữa các mẫu là một nguồn thông tin riêng. Khi dữ liệu có dạng mạng lưới (trích dẫn, bạn bè, phân tử), hãy nghĩ tới việc để mô hình đọc cả các cạnh.'},
  ];
  return <ProjectFrame id="gnn" steps={steps} step={step} setStep={setStep} navigate={navigate}/>;
}
