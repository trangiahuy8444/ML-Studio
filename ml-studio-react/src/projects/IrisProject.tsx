import {useMemo,useState} from 'react';
import type {ReactNode} from 'react';
import type {Route} from '../types';
import iris from '../track-py/data/iris2d.json';
import {LineChart,PALETTE,Plot,ProjectFrame,Segmented,Slider,Stat,fmt,pct,rng,shuffled} from './ui';
import type {Scale,StepDef,ReverseDef} from './ui';

type Row={f:[number,number];y:number};
const SPECIES=['setosa','versicolor','virginica'],COLORS=[PALETTE[0],PALETTE[2],PALETTE[1]];
const FEAT=['Dài cánh hoa (cm)','Rộng cánh hoa (cm)'];
const rows:Row[]=(iris.points as number[][]).map(p=>({f:[p[0],p[1]],y:p[2]}));
const order=shuffled(rows.map((_,i)=>i),11),trainIdx=order.slice(0,105),testIdx=order.slice(105);
const X:[number,number]=[0.6,7.2],Y:[number,number]=[0,2.7];

const counts=(idx:number[],data:Row[])=>{const c=[0,0,0];for(const i of idx)c[data[i].y]++;return c};
const gini=(c:number[])=>{const n=c.reduce((a,b)=>a+b,0);return n?1-c.reduce((s,v)=>s+(v/n)**2,0):0};
const argmax=(c:number[])=>c.indexOf(Math.max(...c));

interface Node {idx:number[];counts:number[];depth:number;feature?:number;thr?:number;left?:Node;right?:Node}
function bestSplit(idx:number[],data:Row[]){
  let best={gain:0,feature:-1,thr:0};const g=gini(counts(idx,data));
  for(const f of [0,1]){const vals=[...new Set(idx.map(i=>data[i].f[f]))].sort((a,b)=>a-b);
    for(let j=0;j<vals.length-1;j++){const t=(vals[j]+vals[j+1])/2,L=idx.filter(i=>data[i].f[f]<=t),R=idx.filter(i=>data[i].f[f]>t);
      const gain=g-(L.length*gini(counts(L,data))+R.length*gini(counts(R,data)))/idx.length;if(gain>best.gain+1e-12)best={gain,feature:f,thr:t}}}
  return best;
}
function grow(idx:number[],data:Row[],maxDepth:number,depth=0):Node{
  const node:Node={idx,counts:counts(idx,data),depth};
  if(depth>=maxDepth||gini(node.counts)===0)return node;
  const s=bestSplit(idx,data);if(s.feature<0)return node;
  node.feature=s.feature;node.thr=s.thr;
  node.left=grow(idx.filter(i=>data[i].f[s.feature]<=s.thr),data,maxDepth,depth+1);
  node.right=grow(idx.filter(i=>data[i].f[s.feature]>s.thr),data,maxDepth,depth+1);
  return node;
}
function predict(n:Node,f:[number,number],path:Node[]=[]):{y:number;path:Node[]}{path.push(n);if(n.feature===undefined||!n.left||!n.right)return {y:argmax(n.counts),path};return predict(f[n.feature]<=n.thr!?n.left:n.right,f,path)}
const accuracy=(n:Node,idx:number[],data:Row[])=>idx.filter(i=>predict(n,data[i].f).y===data[i].y).length/idx.length;
const leaves=(n:Node):number=>n.left&&n.right?leaves(n.left)+leaves(n.right):1;

function Regions({node,box,sx,sy}:{node:Node;box:[number,number,number,number];sx:Scale;sy:Scale}){
  const [x0,x1,y0,y1]=box;
  if(node.feature===undefined||!node.left||!node.right)return <rect x={sx(x0)} y={sy(y1)} width={sx(x1)-sx(x0)} height={sy(y0)-sy(y1)} fill={COLORS[argmax(node.counts)]} opacity={.13} stroke="var(--line-strong)" strokeWidth={.5}/>;
  const t=node.thr!;
  return node.feature===0?<><Regions node={node.left} box={[x0,Math.min(t,x1),y0,y1]} sx={sx} sy={sy}/><Regions node={node.right} box={[Math.max(t,x0),x1,y0,y1]} sx={sx} sy={sy}/></>
    :<><Regions node={node.left} box={[x0,x1,y0,Math.min(t,y1)]} sx={sx} sy={sy}/><Regions node={node.right} box={[x0,x1,Math.max(t,y0),y1]} sx={sx} sy={sy}/></>;
}
function Dots({data,idx,sx,sy,r=3.6}:{data:Row[];idx:number[];sx:Scale;sy:Scale;r?:number}){
  // Rung nhẹ để các bông hoa trùng số đo không chồng khít lên nhau.
  const j=rng(3);return <>{idx.map(i=>{const dx=(j()-.5)*.06,dy=(j()-.5)*.04;return <circle key={i} cx={sx(data[i].f[0]+dx)} cy={sy(data[i].f[1]+dy)} r={r} fill={COLORS[data[i].y]} fillOpacity={.85}/>})}</>;
}
function TreeView({node,active}:{node:Node;active?:Set<Node>}){
  const leaf=node.feature===undefined||!node.left||!node.right;
  return <li className={active?.has(node)?'on':''}>
    <div className="tree-node">{leaf?<><b style={{color:COLORS[argmax(node.counts)]}}>{SPECIES[argmax(node.counts)]}</b><small>{node.counts.join(' / ')} · Gini {fmt(gini(node.counts))}</small></>
      :<><span>{node.feature===0?'Dài cánh':'Rộng cánh'} ≤ {fmt(node.thr!)}?</span><small>{node.counts.join(' / ')} · Gini {fmt(gini(node.counts))}</small></>}</div>
    {!leaf&&<ul><TreeView node={node.left!} active={active}/><TreeView node={node.right!} active={active}/></ul>}
  </li>;
}
function GroupBars({c,title}:{c:number[];title:string}){const n=c.reduce((a,b)=>a+b,0);
  return <div className="group-bars"><strong>{title} · {n} bông</strong><div className="stack">{c.map((v,i)=>v?<i key={i} style={{flex:v,background:COLORS[i]}} title={`${SPECIES[i]}: ${v}`}>{v}</i>:null)}</div><small>Gini {fmt(gini(c))}</small></div>}

export function IrisProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [f1,setF1]=useState(0),[t1,setT1]=useState(3.2);
  const [f2,setF2]=useState(1),[t2,setT2]=useState(1.5);
  const [depth,setDepth]=useState(2),[noise,setNoise]=useState(false);
  const [q,setQ]=useState<[number,number]>([4.9,1.6]);
  const all=rows.map((_,i)=>i);
  // Bước 1–2: cây em tự đặt
  const L1=all.filter(i=>rows[i].f[f1]<=t1),R1=all.filter(i=>rows[i].f[f1]>t1);
  const mixed=gini(counts(L1,rows))>gini(counts(R1,rows))?'left':'right',M=mixed==='left'?L1:R1;
  const mine:Node=useMemo(()=>{const leaf=(idx:number[],d:number):Node=>({idx,counts:counts(idx,rows),depth:d});
    const L2=M.filter(i=>rows[i].f[f2]<=t2),R2=M.filter(i=>rows[i].f[f2]>t2);
    const sub:Node={...leaf(M,1),feature:f2,thr:t2,left:leaf(L2,2),right:leaf(R2,2)};
    return {...leaf(all,0),feature:f1,thr:t1,left:mixed==='left'?sub:leaf(L1,1),right:mixed==='right'?sub:leaf(R1,1)};
  },[f1,t1,f2,t2])// eslint-disable-line react-hooks/exhaustive-deps
  const w1=(L1.length*gini(counts(L1,rows))+R1.length*gini(counts(R1,rows)))/rows.length;
  // Bước 3: máy quét mọi ngưỡng ở gốc
  const scan=useMemo(()=>[0,1].map(f=>{const vals=[...new Set(rows.map(r=>r.f[f]))].sort((a,b)=>a-b);return vals.slice(0,-1).map((v,j)=>{const t=(v+vals[j+1])/2,L=all.filter(i=>rows[i].f[f]<=t),R=all.filter(i=>rows[i].f[f]>t);return {t,g:gini([50,50,50])-(L.length*gini(counts(L,rows))+R.length*gini(counts(R,rows)))/150}})}),[]);// eslint-disable-line react-hooks/exhaustive-deps
  const auto=useMemo(()=>grow(all,rows,depth),[depth]);// eslint-disable-line react-hooks/exhaustive-deps
  // Bước 4: nhãn nhiễu + độ sâu
  const data=useMemo(()=>{if(!noise)return rows;const r=rng(5);return rows.map((row,i)=>trainIdx.includes(i)&&r()<.12?{...row,y:(row.y+1+Math.floor(r()*2))%3}:row)},[noise]);
  const curve=useMemo(()=>[1,2,3,4,5,6,7,8].map(d=>{const t=grow(trainIdx,data,d);return {tr:accuracy(t,trainIdx,data),te:accuracy(t,testIdx,rows),leaves:leaves(t)}}),[data]);
  const deep=useMemo(()=>grow(trainIdx,data,depth),[data,depth]);
  // Bước 5
  const final=useMemo(()=>grow(all,rows,3),[]);// eslint-disable-line react-hooks/exhaustive-deps
  const res=predict(final,q);

  const plot=(node:Node|null,idx:number[],d:Row[]=rows,extra?:(sx:Scale,sy:Scale)=>ReactNode)=><Plot x={X} y={Y} xLabel={FEAT[0]} yLabel={FEAT[1]}>{(sx,sy)=><>{node&&<Regions node={node} box={[X[0],X[1],Y[0],Y[1]]} sx={sx} sy={sy}/>}<Dots data={d} idx={idx} sx={sx} sy={sy}/>{extra?.(sx,sy)}</>}</Plot>;
  const legend=<p className="chart-caption">{SPECIES.map((s,i)=><span key={s} style={{color:COLORS[i],marginRight:12}}>● {s}</span>)}</p>;
  const split=(f:number,t:number,sx:Scale,sy:Scale,c='#fff')=>f===0?<line x1={sx(t)} x2={sx(t)} y1={sy(Y[0])} y2={sy(Y[1])} stroke={c} strokeWidth={2} strokeDasharray="6 4"/>:<line x1={sx(X[0])} x2={sx(X[1])} y1={sy(t)} y2={sy(t)} stroke={c} strokeWidth={2} strokeDasharray="6 4"/>;

  const steps:StepDef[]=[
    {label:'Câu hỏi đầu tiên',title:'Chia 150 bông bằng một câu hỏi có/không.',
      intro:<p>Chọn một số đo và ngưỡng. Một câu hỏi tốt chia ra hai nhóm <b>thuần</b>: mỗi nhóm gần như chỉ một loài. Gini đo độ lẫn lộn: 0 là thuần hoàn toàn.</p>,
      controls:<><Segmented label="Số đo" options={[[0,'Dài cánh'],[1,'Rộng cánh']]} value={f1} onChange={v=>{setF1(v);setT1(v===0?3.2:1.0)}}/>
        <Slider label="Ngưỡng" value={t1} min={f1===0?1:0.1} max={f1===0?7:2.5} step={.05} onChange={setT1} format={v=>fmt(v)+' cm'}/>
        <GroupBars c={counts(L1,rows)} title="≤ ngưỡng"/><GroupBars c={counts(R1,rows)} title="> ngưỡng"/>
        <div className="stat-row"><Stat label="Gini trung bình" value={fmt(w1,3)} tone={w1<.34?'good':undefined}/></div></>,
      visual:<>{plot(null,all,rows,(sx,sy)=>split(f1,t1,sx,sy))}{legend}</>,
      takeaway:'Có những ngưỡng tách gọn một loài (setosa nhỏ hẳn) chỉ bằng một câu hỏi. Gini trung bình giúp so hai câu hỏi một cách khách quan thay vì cảm tính.'},
    {label:'Câu hỏi thứ hai',title:'Hỏi tiếp nhóm còn lẫn lộn.',
      intro:<p>Câu hỏi thứ hai chỉ áp dụng cho nhóm <b>{mixed==='left'?'≤':'>'} ngưỡng</b> (nhóm lẫn hơn). Vùng tô màu là loài cây của em sẽ đoán.</p>,
      controls:<><Segmented label="Số đo" options={[[0,'Dài cánh'],[1,'Rộng cánh']]} value={f2} onChange={v=>{setF2(v);setT2(v===0?4.9:1.6)}}/>
        <Slider label="Ngưỡng" value={t2} min={f2===0?1:0.1} max={f2===0?7:2.5} step={.05} onChange={setT2} format={v=>fmt(v)+' cm'}/>
        <div className="stat-row"><Stat label="Cây của em đúng" value={pct(accuracy(mine,all,rows))}/><Stat label="Số lá" value={3}/></div>
        <ul className="tree"><TreeView node={mine}/></ul></>,
      visual:<>{plot(mine,all)}{legend}</>,
      takeaway:'Mỗi câu hỏi thêm một đường cắt song song trục, nên vùng quyết định của cây luôn là các hình chữ nhật. Hai câu hỏi đã đủ để đúng khoảng 95%.'},
    {label:'Máy tìm câu hỏi',title:'Máy thử mọi ngưỡng có thể.',
      intro:<p>Ở gốc, máy thử từng ngưỡng nằm giữa hai giá trị liên tiếp và chọn ngưỡng làm Gini giảm nhiều nhất. Rồi lặp lại cho từng nhóm con.</p>,
      controls:<><Slider label="Độ sâu tối đa" value={depth} min={1} max={4} onChange={setDepth}/>
        <div className="stat-row"><Stat label="Máy đúng" value={pct(accuracy(auto,all,rows))}/><Stat label="Số lá" value={leaves(auto)}/></div>
        <ul className="tree"><TreeView node={auto}/></ul></>,
      visual:<>{plot(auto,all)}<LineChart height={160} xLabel="Thứ tự ngưỡng (từ nhỏ đến lớn)" yLabel="Gini giảm" series={[{name:'Dài cánh',values:scan[0].map(s=>s.g),color:PALETTE[0]},{name:'Rộng cánh',values:scan[1].map(s=>s.g),color:PALETTE[2]}]}/></>,
      takeaway:'Thuật toán CART tham lam: ở mỗi nút chỉ chọn câu hỏi tốt nhất ngay lúc đó, không nhìn trước. Nhanh và dễ hiểu, nhưng không bảo đảm ra cây tốt nhất toàn cục.'},
    {label:'Cây quá sâu',title:'Hỏi mãi thì sao?',
      intro:<p>Giờ máy học trên 105 bông và kiểm tra trên 45 bông khác. Bật <b>nhãn nhiễu</b> để giả lập dữ liệu ghi chép sai, rồi tăng độ sâu.</p>,
      controls:<><Slider label="Độ sâu tối đa" value={depth} min={1} max={8} onChange={setDepth}/>
        <label className="check-line"><input type="checkbox" checked={noise} onChange={e=>setNoise(e.target.checked)}/>Ghi sai 12% nhãn trong tập huấn luyện</label>
        <div className="stat-row"><Stat label="Train" value={pct(curve[depth-1].tr)}/><Stat label="Test" value={pct(curve[depth-1].te)} tone={curve[depth-1].te<curve[depth-1].tr-.08?'bad':undefined}/><Stat label="Số lá" value={curve[depth-1].leaves}/></div></>,
      visual:<>{plot(deep,trainIdx,data)}<LineChart height={160} xLabel="Độ sâu − 1" yLabel="Độ chính xác" series={[{name:'Train',values:curve.map(c=>c.tr),color:PALETTE[3]},{name:'Test',values:curve.map(c=>c.te),color:PALETTE[1]}]}/></>,
      takeaway:'Cây đủ sâu học thuộc cả nhãn sai: train lên gần 100% nhưng test không tăng, có khi giảm. Giới hạn độ sâu hay số mẫu mỗi lá là cách chống overfitting của cây.'},
    {label:'Đoán hoa mới',title:'Đi theo các câu hỏi.',
      intro:<p>Em đo một bông hoa mới. Đường đi qua cây được tô sáng, và chấm vàng là vị trí bông hoa trên biểu đồ.</p>,
      controls:<><Slider label="Dài cánh hoa" value={q[0]} min={1} max={7} step={.1} onChange={v=>setQ([v,q[1]])} format={v=>fmt(v,1)+' cm'}/><Slider label="Rộng cánh hoa" value={q[1]} min={.1} max={2.5} step={.05} onChange={v=>setQ([q[0],v])} format={v=>fmt(v)+' cm'}/>
        <div className="stat-row"><Stat label="Máy đoán" value={SPECIES[res.y]}/></div>
        <ul className="tree"><TreeView node={final} active={new Set(res.path)}/></ul></>,
      visual:<>{plot(final,all,rows,(sx,sy)=><circle cx={sx(q[0])} cy={sy(q[1])} r={9} fill="#ffd173" stroke="#0a1220" strokeWidth={2}/>)}{legend}</>,
      takeaway:'Ưu điểm lớn nhất của cây: em giải thích được từng dự đoán bằng chuỗi câu hỏi. Vùng giáp ranh giữa versicolor và virginica là nơi cây dễ sai nhất, đúng như mắt em thấy.'},
  ];
  const reverse:ReverseDef={layers:[
    {from:4,label:'Dùng sản phẩm',title:'Một cây quyết định đã xây xong.',intro:<p>Đo một bông hoa mới bằng hai thanh trượt. Cây trả lời ngay loài hoa, kèm đường đi qua các câu hỏi được tô sáng.</p>,takeaway:'Sản phẩm là một chuỗi câu hỏi có/không. Khác logistic, em đọc được ngay lý do của từng dự đoán.'},
    {from:2,label:'Nó được xây thế nào',title:'Mở lại quá trình dựng cây.',intro:<p>Cây không ai viết tay. Máy thử mọi ngưỡng và chọn câu hỏi làm Gini giảm nhiều nhất. Kéo độ sâu từ 1 lên để xem cây mọc dần.</p>,takeaway:'Mỗi câu hỏi trong sản phẩm là kết quả của một lần quét ngưỡng tham lam.'},
    {from:3,label:'Tháo giới hạn',title:'Bỏ giới hạn độ sâu thì sao?',intro:<p>Sản phẩm bị giới hạn độ sâu 3. Tháo giới hạn đó, bật nhãn ghi sai, rồi xem train và test.</p>,takeaway:'Giới hạn độ sâu là bộ phận chống học thuộc. Tháo nó ra, cây học cả lỗi trong dữ liệu.'},
    {from:0,label:'Tự lắp: câu 1',title:'Em tự đặt câu hỏi gốc.',intro:<p>Lắp lại cây từ trang trắng. Chọn số đo và ngưỡng cho câu hỏi đầu tiên, cố đạt Gini trung bình thấp nhất.</p>,takeaway:'Câu hỏi gốc tốt nhất tách gọn setosa: em vừa tìm lại đúng câu hỏi mà máy đã chọn.'},
    {from:1,label:'Tự lắp: câu 2',title:'Hoàn thiện cây của em.',intro:<p>Thêm câu hỏi thứ hai cho nhóm còn lẫn. So độ chính xác của cây em với sản phẩm.</p>,takeaway:'Hai câu hỏi đã lắp lại gần đủ sản phẩm. Phần thiếu chỉ là vài câu hỏi nhỏ ở vùng giáp ranh.'},
  ]};
  return <ProjectFrame id="iris" steps={steps} reverse={reverse} navigate={navigate}/>;
}
