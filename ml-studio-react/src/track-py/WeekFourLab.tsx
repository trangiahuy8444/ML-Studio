// Mô phỏng tuần 4 (Ngày 22–28): mạng nơ-ron và PyTorch. Mọi mạng ở đây được huấn luyện thật trong trình duyệt
// bằng JavaScript thuần (cùng công thức với bài Python). Dữ liệu thật: Iris (data/iris2d.json), 400 ảnh load_digits
// (data/digits.json). Dữ liệu hai vầng trăng, mặt loss và câu ví dụ ở Ngày 27 là minh họa, được ghi rõ trên trang.
import {useMemo,useState} from 'react';
import {Chart} from '../components/Chart';
import {Range,Table,Transport,usePlayback} from '../components/ObservationControls';
import {SimulationCode,SimulationPanel} from '../components/SimulationStudio';
import type {PlotTrace} from '../types';
import digits from './data/digits.json';
import iris from './data/iris2d.json';

const CY='#42d3e5',PK='#f28db9',GD='#ffd173',GR='#84e0a8',VI='#a8a7ff',OR='#ff9b6a';
const fmt=(v:number,d=3)=>{if(!Number.isFinite(v))return '—';const t=v.toLocaleString('vi-VN',{maximumFractionDigits:d});return /^-0(,0*)?$/.test(t)?'0':t};
const mean=(a:number[])=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;
const sd=(a:number[])=>{const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1))};
function rng(seed:number){let s=(seed*2654435761)>>>0||1;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
function gauss(r:()=>number){return Math.sqrt(-2*Math.log(Math.max(r(),1e-12)))*Math.cos(2*Math.PI*r())}
function shuffle<T>(a:T[],r:()=>number){const b=[...a];for(let i=b.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[b[i],b[j]]=[b[j],b[i]]}return b}
const scatter=(name:string,pts:number[][],color:string,extra:Record<string,unknown>={}):PlotTrace=>({type:'scatter',mode:'markers',name,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),marker:{color,size:7,opacity:.85},...extra});
function heat(f:(x:number,y:number)=>number,xr:[number,number],yr:[number,number],name:string):PlotTrace{
  const xs=Array.from({length:50},(_,i)=>xr[0]+i*(xr[1]-xr[0])/49),ys=Array.from({length:50},(_,i)=>yr[0]+i*(yr[1]-yr[0])/49);
  return {type:'heatmap',x:xs,y:ys,z:ys.map(y=>xs.map(x=>f(x,y))),zmin:0,zmax:1,showscale:false,opacity:.55,hoverinfo:'skip',name,colorscale:[[0,'#123a52'],[.5,'#1b2238'],[1,'#57264a']]};
}

// ───────── MLP một tầng ẩn (ReLU, softmax) dùng chung ─────────
type Net={W1:number[][];b1:number[];W2:number[][];b2:number[]};
function initNet(d:number,h:number,c:number,r:()=>number):Net{
  return {W1:Array.from({length:d},()=>Array.from({length:h},()=>gauss(r)*Math.sqrt(2/d))),b1:new Array(h).fill(0),
    W2:Array.from({length:h},()=>Array.from({length:c},()=>gauss(r)*Math.sqrt(2/h))),b2:new Array(c).fill(0)};
}
function forward(n:Net,x:number[]){
  const h=n.b1.map((b,j)=>Math.max(0,x.reduce((s,v,i)=>s+v*n.W1[i][j],b)));
  const z=n.b2.map((b,k)=>h.reduce((s,v,j)=>s+v*n.W2[j][k],b));const m=Math.max(...z);const e=z.map(v=>Math.exp(v-m));const t=e.reduce((s,v)=>s+v,0);
  return {h,p:e.map(v=>v/t)};
}
function zeroGrad(n:Net):Net{return {W1:n.W1.map(r=>r.map(()=>0)),b1:n.b1.map(()=>0),W2:n.W2.map(r=>r.map(()=>0)),b2:n.b2.map(()=>0)}}
function addGrad(g:Net,n:Net,x:number[],y:number,scale:number){
  const {h,p}=forward(n,x);const d2=p.map((v,k)=>(v-(k===y?1:0))*scale);
  d2.forEach((d,k)=>{g.b2[k]+=d;h.forEach((hv,j)=>g.W2[j][k]+=hv*d)});
  const d1=h.map((hv,j)=>hv>0?d2.reduce((s,d,k)=>s+d*n.W2[j][k],0):0);
  d1.forEach((d,j)=>{g.b1[j]+=d;x.forEach((xv,i)=>g.W1[i][j]+=xv*d)});
  return -Math.log(Math.max(p[y],1e-12));
}
function step(n:Net,g:Net,lr:number){n.W1.forEach((r,i)=>r.forEach((_,j)=>r[j]-=lr*g.W1[i][j]));n.b1.forEach((_,j)=>n.b1[j]-=lr*g.b1[j]);n.W2.forEach((r,j)=>r.forEach((_,k)=>r[k]-=lr*g.W2[j][k]));n.b2.forEach((_,k)=>n.b2[k]-=lr*g.b2[k])}
const accOf=(n:Net,X:number[][],Y:number[])=>mean(X.map((x,i)=>{const p=forward(n,x).p;return Number(p.indexOf(Math.max(...p))===Y[i])}));
function trainMinibatch(X:number[][],Y:number[],Xv:number[][],Yv:number[],o:{hidden:number;lr:number;batch:number;epochs:number;seed:number;accumulate?:boolean}){
  const r=rng(o.seed);const net=initNet(X[0].length,o.hidden,10,r);const steps:number[]=[],epochs:{train:number;val:number}[]=[];let g=zeroGrad(net);
  for(let e=0;e<o.epochs;e++){const order=shuffle(X.map((_,i)=>i),r);
    for(let s=0;s<order.length;s+=o.batch){const ids=order.slice(s,s+o.batch);if(!o.accumulate)g=zeroGrad(net);
      let loss=0;ids.forEach(i=>loss+=addGrad(g,net,X[i],Y[i],1/ids.length));step(net,g,o.lr);steps.push(loss/ids.length)}
    epochs.push({train:accOf(net,X,Y),val:accOf(net,Xv,Yv)})}
  return {net,steps,epochs};
}
const DG=digits as unknown as {source:string;samples:{label:number[];pixels:number[][]}};
const DX=DG.samples.pixels.map(p=>p.map(v=>v/16)),DY=DG.samples.label;
const D_TR={X:DX.slice(0,300),Y:DY.slice(0,300)},D_VA={X:DX.slice(300),Y:DY.slice(300)};
const digitNote=<p className="small-note">Dữ liệu: {DG.source}; 400 ảnh 8×8, 300 ảnh đầu để huấn luyện, 100 ảnh sau để kiểm tra. Mạng 64 → ẩn → 10 huấn luyện ngay trong trình duyệt.</p>;

// ───────── Ngày 22: perceptron và MLP ─────────
const IP=iris.points as [number,number,number][];
const SETS:Record<string,{X:number[][];y:number[];note:string}>={
  iris:{X:IP.map(p=>[p[0],p[1]]),y:IP.map(p=>p[2]===0?1:-1),note:`${iris.source}: setosa (+1) và hai loài còn lại (−1), petal length/width (cm).`},
  and:{X:[[0,0],[0,1],[1,0],[1,1]],y:[-1,-1,-1,1],note:'Hàm AND: tách được tuyến tính.'},
  xor:{X:[[0,0],[0,1],[1,0],[1,1]],y:[-1,1,1,-1],note:'Hàm XOR: không có đường thẳng nào tách được.'},
};
function moons(n:number,seed:number){const r=rng(seed);return Array.from({length:n},(_,i)=>{const c=i%2,t=Math.PI*r();const x=c?1-Math.cos(t):Math.cos(t),y=c?.5-Math.sin(t):Math.sin(t);return {x:[x+gauss(r)*.2,y+gauss(r)*.2],c}})}
const MOON=moons(240,22);
function PerceptronLab(){
  const [set,setSet]=useState<'iris'|'and'|'xor'>('iris'),[hidden,setHidden]=useState(8);
  const S=SETS[set];
  const trace=useMemo(()=>{let w=[0,0],b=0;const out:{w:number[];b:number;i:number;upd:boolean;epoch:number}[]=[];const mist:number[]=[];
    for(let e=0;e<30;e++){let m=0;S.X.forEach((x,i)=>{const s=S.y[i]*(w[0]*x[0]+w[1]*x[1]+b);const upd=s<=0;if(upd){w=[w[0]+S.y[i]*x[0],w[1]+S.y[i]*x[1]];b+=S.y[i];m++}out.push({w,b,i,upd,epoch:e+1})});mist.push(m);if(m===0)break}
    return {out,mist}},[S]);
  const player=usePlayback(trace.out.length,set);const fr=trace.out[player.index];
  const xr:[number,number]=set==='iris'?[0,7.5]:[-.5,1.5],yr:[number,number]=set==='iris'?[-.5,3]:[-.5,1.5];
  const line=Math.abs(fr.w[1])>1e-9?{x:xr,y:xr.map(x=>-(fr.w[0]*x+fr.b)/fr.w[1])}:{x:[-fr.b/(fr.w[0]||1),-fr.b/(fr.w[0]||1)],y:yr};
  const mlp=useMemo(()=>{const X=MOON.map(m=>m.x),Y=MOON.map(m=>m.c);const r=rng(5);const net=initNet(2,hidden,2,r);
    for(let s=0;s<600;s++){const g=zeroGrad(net);X.forEach((x,i)=>addGrad(g,net,x,Y[i],1/X.length));step(net,g,.5)}
    return {net,acc:accOf(net,X,Y)}},[hidden]);
  return <>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Perceptron · vòng ${fr.epoch}, mẫu ${fr.i+1} ${fr.upd?'→ cập nhật':'→ đúng'}`}>
        <label>Dữ liệu<select value={set} onChange={e=>setSet(e.target.value as 'iris'|'and'|'xor')}><option value="iris">Iris setosa / không setosa</option><option value="and">AND</option><option value="xor">XOR</option></select></label>
        <p className="small-note">{S.note}</p>
        <Transport player={player} count={trace.out.length}/>
        <Chart traces={[scatter('+1',S.X.filter((_,i)=>S.y[i]===1),PK),scatter('−1',S.X.filter((_,i)=>S.y[i]===-1),CY),scatter('Mẫu đang xét',[S.X[fr.i]],'rgba(0,0,0,0)',{marker:{size:16,color:'rgba(0,0,0,0)',line:{color:GD,width:2.5}}}),{type:'scatter',mode:'lines',name:'w·x + b = 0',x:line.x,y:line.y,line:{color:GD,width:3}}]} layout={{xaxis:{range:xr},yaxis:{range:yr}}} height={320}/>
        <Table headers={['w₁','w₂','b','Lỗi mỗi vòng']} rows={[[fmt(fr.w[0],2),fmt(fr.w[1],2),fmt(fr.b,2),trace.mist.join(', ')+(trace.mist[trace.mist.length-1]===0?' (hội tụ)':' (chưa hội tụ sau 30 vòng)')]]}/>
      </SimulationPanel>
      <SimulationPanel title={`MLP 2 → ${hidden} → 2 trên hai vầng trăng · accuracy ${fmt(mlp.acc)}`}>
        <p className="small-note">Hai vầng trăng là dữ liệu giả lập (240 điểm, nhiễu 0,2). Gradient descent toàn bộ dữ liệu, 600 bước, lr = 0,5.</p>
        <Range label="Số nơ-ron ẩn" value={hidden} min={1} max={16} change={setHidden}/>
        <Chart traces={[heat((x,y)=>forward(mlp.net,[x,y]).p[1],[-1.5,2.5],[-1,1.5],'p(lớp 1)'),scatter('Lớp 0',MOON.filter(m=>m.c===0).map(m=>m.x),CY,{marker:{color:CY,size:5}}),scatter('Lớp 1',MOON.filter(m=>m.c===1).map(m=>m.x),PK,{marker:{color:PK,size:5}})]} height={320}/>
        <SimulationCode file="mlp.py" lines={['Z1 = X @ W1 + b1','H = np.maximum(0, Z1)        # ReLU','P = softmax(H @ W2 + b2)']} active={1}/>
        <p className="small-note">Mỗi nơ-ron ReLU góp một "nếp gấp" thẳng; 1–2 nơ-ron chỉ vẽ được ranh giới gần thẳng, thêm nơ-ron thì ranh giới uốn theo hai vầng trăng.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 23: lan truyền ngược trên đồ thị tính toán ─────────
function BackpropLab(){
  const [x1,setX1]=useState(1),[x2,setX2]=useState(2),[w,setW]=useState(.5),[label,setLabel]=useState(0);
  const W1=[[w,-1],[.25,.5]],W2=[[1,-1],[.5,1]];
  const z=[0,1].map(j=>x1*W1[0][j]+x2*W1[1][j]),h=z.map(v=>Math.max(0,v)),o=[0,1].map(k=>h[0]*W2[0][k]+h[1]*W2[1][k]);
  const m=Math.max(...o),e=o.map(v=>Math.exp(v-m)),p=e.map(v=>v/(e[0]+e[1])),loss=-Math.log(p[label]);
  const d2=p.map((v,k)=>v-(k===label?1:0)),dh=[0,1].map(j=>d2[0]*W2[j][0]+d2[1]*W2[j][1]),d1=dh.map((v,j)=>z[j]>0?v:0);
  const dW1=[[x1*d1[0],x1*d1[1]],[x2*d1[0],x2*d1[1]]],dW2=[[h[0]*d2[0],h[0]*d2[1]],[h[1]*d2[0],h[1]*d2[1]]];
  const nodes=[{id:'x1',x:40,y:60,v:x1,g:null as number|null,s:0},{id:'x2',x:40,y:170,v:x2,g:null,s:0},{id:'z1',x:170,y:60,v:z[0],g:d1[0],s:1},{id:'z2',x:170,y:170,v:z[1],g:d1[1],s:1},{id:'h1',x:290,y:60,v:h[0],g:dh[0],s:2},{id:'h2',x:290,y:170,v:h[1],g:dh[1],s:2},{id:'o1',x:410,y:60,v:o[0],g:d2[0],s:3},{id:'o2',x:410,y:170,v:o[1],g:d2[1],s:3},{id:'loss',x:530,y:115,v:loss,g:1,s:4}];
  const edges=[['x1','z1'],['x1','z2'],['x2','z1'],['x2','z2'],['z1','h1'],['z2','h2'],['h1','o1'],['h1','o2'],['h2','o1'],['h2','o2'],['o1','loss'],['o2','loss']];
  const player=usePlayback(10,`${x1}-${x2}-${w}-${label}`);const k=player.index;
  const fwdStage=Math.min(k,4),bwdStage=k>=5?9-k:5;
  const pos=Object.fromEntries(nodes.map(n=>[n.id,n]));
  const lines=['z = x @ W1','h = relu(z)','o = h @ W2; p = softmax(o)','loss = −log p[y]','d_o = p − onehot(y)','d_h = d_o @ W2.T','d_z = d_h * (z > 0)','dW1 = x.T @ d_z','(xong)','(xong)'];
  return <>
    <p className="small-note">Mạng 2 → 2 → 2 với trọng số cố định của ví dụ tính tay (W1 = [[w, −1], [0,25, 0,5]], W2 = [[1, −1], [0,5, 1]]). Kéo thanh trượt để thấy giá trị và gradient tính lại ngay.</p>
    <div className="py30-grid"><Range label="x₁" value={x1} min={-2} max={3} step={.25} change={setX1}/><Range label="x₂" value={x2} min={-2} max={3} step={.25} change={setX2}/><Range label="w = W1[0][0]" value={w} min={-2} max={2} step={.25} change={setW}/>
      <label>Nhãn đúng y<select value={label} onChange={e=>setLabel(+e.target.value)}><option value={0}>0</option><option value={1}>1</option></select></label></div>
    <Transport player={player} count={10}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={k<4?`Lan truyền xuôi · bước ${k+1}`:k===4?'Lan truyền xuôi xong':'Lan truyền ngược: gradient đi từ loss về đầu vào'}>
        <svg viewBox="0 0 580 230" width="100%" role="img" aria-label="Đồ thị tính toán của mạng 2-2-2">
          {edges.map(([a,b])=>{const A=pos[a],B=pos[b];const back=k>=5&&B.s>=bwdStage;return <line key={a+b} x1={A.x+22} y1={A.y} x2={B.x-22} y2={B.y} stroke={back?PK:A.s<fwdStage||k>=4?CY:'#2b405a'} strokeWidth={back?2.5:1.5}/>})}
          {nodes.map(n=>{const shown=n.s<=fwdStage||k>=4;const gshown=k>=5&&n.g!==null&&n.s>=bwdStage;return <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={22} fill="#132339" stroke={n.id==='loss'?GD:n.s===2&&n.v===0?'#5a6b80':CY} strokeWidth={2}/>
            <text x={n.x} y={n.y-30} textAnchor="middle" fill="#b5c9df" fontSize={12}>{n.id}</text>
            <text x={n.x} y={n.y+4} textAnchor="middle" fill="#eaf6ff" fontSize={12}>{shown?fmt(n.v,2):'?'}</text>
            {gshown&&<text x={n.x} y={n.y+40} textAnchor="middle" fill={PK} fontSize={12}>∂ {fmt(n.g as number,3)}</text>}</g>})}
        </svg>
        <p className="small-note">Số trắng: giá trị khi lan truyền xuôi. Số hồng: đạo hàm của loss theo nút đó. Nút ẩn có z ≤ 0 bị ReLU chặn nên gradient qua nó bằng 0.</p>
      </SimulationPanel>
      <SimulationPanel title={`p = [${fmt(p[0])}, ${fmt(p[1])}] · loss = ${fmt(loss,4)}`}>
        <Table headers={['Gradient','Giá trị']} rows={[['dW2',`[[${dW2[0].map(v=>fmt(v)).join(', ')}], [${dW2[1].map(v=>fmt(v)).join(', ')}]]`],['d_h',`[${dh.map(v=>fmt(v)).join(', ')}]`],['d_z',`[${d1.map(v=>fmt(v)).join(', ')}]`],['dW1',`[[${dW1[0].map(v=>fmt(v)).join(', ')}], [${dW1[1].map(v=>fmt(v)).join(', ')}]]`]]}/>
        <SimulationCode file="backprop.py" lines={lines.slice(0,8)} active={Math.min(k,7)}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 24: vòng huấn luyện ─────────
function TrainLoopLab(){
  const [batch,setBatch]=useState(32),[lr,setLr]=useState(.3),[forget,setForget]=useState(false),[hidden,setHidden]=useState(32);
  const run=useMemo(()=>trainMinibatch(D_TR.X,D_TR.Y,D_VA.X,D_VA.Y,{hidden,lr,batch,epochs:12,seed:24,accumulate:forget}),[batch,lr,forget,hidden]);
  const perEpoch=Math.ceil(300/batch);
  return <>{digitNote}
    <div className="py30-grid"><Range label="Batch size" value={batch} min={4} max={300} step={4} change={setBatch}/><Range label="Learning rate" value={lr} min={.01} max={1} step={.01} change={setLr}/><Range label="Nơ-ron ẩn" value={hidden} min={4} max={64} step={4} change={setHidden}/>
      <label><input type="checkbox" checked={forget} onChange={e=>setForget(e.target.checked)}/> Quên optimizer.zero_grad()</label></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`${perEpoch} bước mỗi epoch · accuracy kiểm tra sau 12 epoch: ${fmt(run.epochs[run.epochs.length-1].val)}`}>
        <Chart xTitle="Bước cập nhật" yTitle="Loss của batch" height={260} traces={[{type:'scatter',mode:'lines',name:'Loss từng batch',x:run.steps.map((_,i)=>i+1),y:run.steps.map(v=>Math.min(v,10)),line:{color:PK,width:1}}]}/>
        <p className="small-note">Batch nhỏ: nhiều bước mỗi epoch nhưng loss nhảy mạnh. Batch = 300 là gradient descent toàn bộ dữ liệu: mượt nhưng chỉ 1 bước mỗi epoch. Bật "Quên zero_grad": gradient cộng dồn qua các batch nên bước cập nhật lớn dần và loss tăng vọt.</p>
      </SimulationPanel>
      <SimulationPanel title="Accuracy theo epoch">
        <Chart xTitle="Epoch" yTitle="Accuracy" height={260} traces={[{type:'scatter',mode:'lines+markers',name:'Huấn luyện',x:run.epochs.map((_,i)=>i+1),y:run.epochs.map(e=>e.train),line:{color:GD}},{type:'scatter',mode:'lines+markers',name:'Kiểm tra',x:run.epochs.map((_,i)=>i+1),y:run.epochs.map(e=>e.val),line:{color:CY}}]} layout={{yaxis:{range:[0,1.02]}}}/>
        <SimulationCode file="train_loop.py" lines={['for xb, yb in loader:','    optimizer.zero_grad()','    loss = loss_fn(model(xb), yb)','    loss.backward()','    optimizer.step()']} active={forget?1:3}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 25: SGD, momentum, Adam trên mặt loss ─────────
const SURF={
  valley:{name:'Thung lũng hẹp ½(10w₁² + w₂²)',f:(a:number,b:number)=>.5*(10*a*a+b*b),g:(a:number,b:number)=>[10*a,b],start:[-1.8,1.6],range:[[-2,2],[-2,2]]},
  rosen:{name:'Rosenbrock (1 − w₁)² + 10(w₂ − w₁²)²',f:(a:number,b:number)=>(1-a)**2+10*(b-a*a)**2,g:(a:number,b:number)=>[-2*(1-a)-40*a*(b-a*a),20*(b-a*a)],start:[-1.5,2],range:[[-2,2],[-1,3]]},
} as const;
function OptimLab(){
  const [surf,setSurf]=useState<keyof typeof SURF>('valley'),[lr,setLr]=useState(.05),[steps,setSteps]=useState(40);const S=SURF[surf];
  const paths=useMemo(()=>{const run=(kind:'sgd'|'mom'|'adam')=>{let w:number[]=[...S.start];let v=[0,0],m=[0,0],s=[0,0];const out=[w];
      for(let t=1;t<=steps;t++){const g=S.g(w[0],w[1]);
        if(kind==='sgd')w=w.map((x,i)=>x-lr*g[i]);
        else if(kind==='mom'){v=v.map((x,i)=>.9*x+g[i]);w=w.map((x,i)=>x-lr*v[i])}
        else{m=m.map((x,i)=>.9*x+.1*g[i]);s=s.map((x,i)=>.999*x+.001*g[i]*g[i]);w=w.map((x,i)=>x-lr*(m[i]/(1-.9**t))/(Math.sqrt(s[i]/(1-.999**t))+1e-8))}
        if(!w.every(Number.isFinite)||Math.abs(w[0])>50||Math.abs(w[1])>50)break;out.push(w)}
      return out};return {sgd:run('sgd'),mom:run('mom'),adam:run('adam')}},[S,lr,steps]);
  const xs=Array.from({length:60},(_,i)=>S.range[0][0]+i*(S.range[0][1]-S.range[0][0])/59),ys=Array.from({length:60},(_,i)=>S.range[1][0]+i*(S.range[1][1]-S.range[1][0])/59);
  const P=(n:string,pts:number[][],c:string):PlotTrace=>({type:'scatter',mode:'lines+markers',name:n,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),line:{color:c,width:2},marker:{size:4,color:c}});
  const last=(pts:number[][])=>pts[pts.length-1];
  return <>
    <p className="small-note">Mặt loss hai chiều là hàm toán học minh họa. Cả ba thuật toán bắt đầu cùng một điểm và dùng cùng learning rate (momentum 0,9; Adam β = (0,9; 0,999)).</p>
    <div className="py30-grid"><label>Mặt loss<select value={surf} onChange={e=>setSurf(e.target.value as keyof typeof SURF)}>{Object.entries(SURF).map(([k,v])=><option key={k} value={k}>{v.name}</option>)}</select></label>
      <Range label="Learning rate" value={lr} min={.005} max={.25} step={.005} change={setLr}/><Range label="Số bước" value={steps} min={5} max={200} step={5} change={setSteps}/></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Đường đi của ba bộ tối ưu">
        <Chart xTitle="w₁" yTitle="w₂" height={380} traces={[{type:'contour',x:xs,y:ys,z:ys.map(b=>xs.map(a=>Math.log1p(S.f(a,b)))),showscale:false,contours:{coloring:'lines'},colorscale:[[0,'#2b405a'],[1,'#7d93ad']],hoverinfo:'skip',name:'log(1 + loss)'},P('SGD',paths.sgd,CY),P('Momentum',paths.mom,PK),P('Adam',paths.adam,GD)]} layout={{xaxis:{range:S.range[0]},yaxis:{range:S.range[1]}}}/>
      </SimulationPanel>
      <SimulationPanel title="Loss cuối">
        <Table headers={['Bộ tối ưu','Số bước chạy được','w cuối','Loss cuối']} rows={([['SGD',paths.sgd],['Momentum',paths.mom],['Adam',paths.adam]] as [string,number[][]][]).map(([n,p])=>[n,p.length-1,`(${fmt(last(p)[0],2)}; ${fmt(last(p)[1],2)})`,fmt(S.f(last(p)[0],last(p)[1]),4)])}/>
        <SimulationCode file="optim.py" lines={['v = momentum * v + g;  w = w - lr * v','m = b1*m + (1-b1)*g;  s = b2*s + (1-b2)*g**2','w = w - lr * (m/(1-b1**t)) / (sqrt(s/(1-b2**t)) + eps)']} active={0}/>
        <p className="small-note">Thung lũng hẹp với lr ≥ 0,2: SGD dao động và nổ theo w₁ (bảng ghi số bước chạy được trước khi vượt khỏi khung). Adam chuẩn hóa từng tọa độ nên mỗi bước dài gần bằng lr ở mọi hướng.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 26: tích chập trên ảnh chữ số ─────────
const KERNELS:Record<string,{name:string;k:number[][]}>={
  vert:{name:'Cạnh dọc (Sobel x)',k:[[-1,0,1],[-2,0,2],[-1,0,1]]},horiz:{name:'Cạnh ngang (Sobel y)',k:[[-1,-2,-1],[0,0,0],[1,2,1]]},
  blur:{name:'Làm mờ (trung bình 3×3)',k:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]},sharp:{name:'Làm sắc nét',k:[[0,-1,0],[-1,5,-1],[0,-1,0]]},
};
function Grid({vals,cell=26,lo,hi,highlight,label}:{vals:number[][];cell?:number;lo:number;hi:number;highlight?:[number,number,number];label:string}){
  const n=vals.length;return <svg width={n*cell} height={n*cell} viewBox={`0 0 ${n*cell} ${n*cell}`} role="img" aria-label={label} style={{maxWidth:'100%'}}>
    {vals.flatMap((row,i)=>row.map((v,j)=>{const t=(v-lo)/((hi-lo)||1);const g=Math.round(255*Math.max(0,Math.min(1,t)));return <rect key={i+'-'+j} x={j*cell} y={i*cell} width={cell} height={cell} fill={`rgb(${g},${g},${g})`} stroke="#22344d" strokeWidth={.5}/>}))}
    {highlight&&<rect x={highlight[1]*cell} y={highlight[0]*cell} width={highlight[2]*cell} height={highlight[2]*cell} fill="none" stroke={GD} strokeWidth={3}/>}
    <title>{label}</title></svg>;
}
function ConvLab(){
  const [sample,setSample]=useState(3),[kern,setKern]=useState('vert'),[stride,setStride]=useState(1),[pad,setPad]=useState(1);
  const img=Array.from({length:8},(_,i)=>DG.samples.pixels[sample].slice(i*8,i*8+8));
  const P=Array.from({length:8+2*pad},(_,i)=>Array.from({length:8+2*pad},(_,j)=>{const a=i-pad,b=j-pad;return a>=0&&a<8&&b>=0&&b<8?img[a][b]:0}));
  const K=KERNELS[kern].k,o=Math.floor((8+2*pad-3)/stride)+1;
  const out=Array.from({length:o},(_,i)=>Array.from({length:o},(_,j)=>K.reduce((s,r,a)=>s+r.reduce((t,kv,b)=>t+kv*P[i*stride+a][j*stride+b],0),0)));
  const player=usePlayback(o*o,`${sample}-${kern}-${stride}-${pad}`);const oi=Math.floor(player.index/o),oj=player.index%o;
  const flat=out.flat(),lo=Math.min(...flat),hi=Math.max(...flat);
  const terms=K.flatMap((r,a)=>r.map((kv,b)=>`${fmt(kv,2)}·${P[oi*stride+a][oj*stride+b]}`));
  return <>
    {digitNote}
    <div className="py30-grid"><Range label="Ảnh mẫu" value={sample} min={0} max={399} change={setSample}/><label>Bộ lọc<select value={kern} onChange={e=>setKern(e.target.value)}>{Object.entries(KERNELS).map(([k,v])=><option key={k} value={k}>{v.name}</option>)}</select></label>
      <Range label="Stride" value={stride} min={1} max={2} change={setStride}/><Range label="Padding" value={pad} min={0} max={1} change={setPad}/></div>
    <Transport player={player} count={o*o}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Ảnh vào ${8+2*pad}×${8+2*pad} (đã đệm ${pad}) · nhãn ${DG.samples.label[sample]}`}>
        <Grid vals={P} lo={0} hi={16} highlight={[oi*stride,oj*stride,3]} label="Ảnh vào đã đệm"/>
        <p className="small-note">Khung vàng là vùng 3×3 đang nhân với bộ lọc.</p>
      </SimulationPanel>
      <SimulationPanel title={`Bản đồ đặc trưng ${o}×${o} = ⌊(8 + 2·${pad} − 3)/${stride}⌋ + 1`}>
        <Grid vals={out} lo={lo} hi={hi} highlight={[oi,oj,1]} label="Bản đồ đặc trưng đầu ra"/>
        <p>out[{oi}, {oj}] = Σ {terms.slice(0,3).join(' + ')} + … = <strong>{fmt(out[oi][oj],2)}</strong></p>
        <Table headers={['Bộ lọc','','']} rows={K.map(r=>r.map(v=>fmt(v,2)))}/>
        <SimulationCode file="conv.py" lines={['x = F.pad(x, (p, p, p, p))','patch = x[:, :, i*s : i*s+k, j*s : j*s+k]','out[:, :, i, j] = (patch * w).sum((1, 2, 3)) + b']} active={1}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 27: self-attention trên một câu ví dụ ─────────
const TOKENS=['con','mèo','ngồi','trên','tấm','thảm'];
function AttentionLab(){
  const [query,setQuery]=useState(2),[causal,setCausal]=useState(false),[scale,setScale]=useState(true),[usePE,setUsePE]=useState(true),[seed,setSeed]=useState(27);
  const d=8;
  const res=useMemo(()=>{const r=rng(seed);const mat=(a:number,b:number)=>Array.from({length:a},()=>Array.from({length:b},()=>gauss(r)/Math.sqrt(a)));
    const E=mat(TOKENS.length,d).map(row=>row.map(v=>v*Math.sqrt(TOKENS.length)));const Wq=mat(d,d),Wk=mat(d,d);
    const PE=TOKENS.map((_,p)=>Array.from({length:d},(_,i)=>{const a=p/10000**((i-(i%2))/d);return i%2?Math.cos(a):Math.sin(a)}));
    const X=E.map((row,p)=>row.map((v,i)=>v+(usePE?PE[p][i]:0)));const mm=(A:number[][],B:number[][])=>A.map(row=>B[0].map((_,j)=>row.reduce((s,v,k)=>s+v*B[k][j],0)));
    const Q=mm(X,Wq),K=mm(X,Wk);const S=Q.map((q,i)=>K.map((k,j)=>(causal&&j>i)?-Infinity:q.reduce((s,v,t)=>s+v*k[t],0)/(scale?Math.sqrt(d):1)));
    const A=S.map(row=>{const m=Math.max(...row);const e=row.map(v=>Math.exp(v-m));const t=e.reduce((s,v)=>s+v,0);return e.map(v=>v/t)});
    return {A,PE}},[causal,scale,usePE,seed]);
  return <>
    <p className="small-note">Câu ví dụ 6 token với embedding và ma trận chiếu W_Q, W_K ngẫu nhiên (seed {seed}), d = {d}: đây là phép tính attention thật nhưng mô hình chưa được huấn luyện, nên trọng số không mang nghĩa ngôn ngữ.</p>
    <div className="py30-grid"><label>Token truy vấn<select value={query} onChange={e=>setQuery(+e.target.value)}>{TOKENS.map((t,i)=><option key={i} value={i}>{t}</option>)}</select></label>
      <label><input type="checkbox" checked={causal} onChange={e=>setCausal(e.target.checked)}/> Mặt nạ nhân quả</label><label><input type="checkbox" checked={scale} onChange={e=>setScale(e.target.checked)}/> Chia cho √d</label><label><input type="checkbox" checked={usePE} onChange={e=>setUsePE(e.target.checked)}/> Cộng positional encoding</label>
      <button className="btn-outline" onClick={()=>setSeed(s=>s+1)}>Đổi trọng số ngẫu nhiên</button></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Ma trận trọng số attention softmax(QKᵀ/√d)">
        <Chart height={340} traces={[{type:'heatmap',x:TOKENS.map((t,i)=>`${i}:${t}`),y:TOKENS.map((t,i)=>`${i}:${t}`),z:res.A,zmin:0,zmax:1,colorscale:[[0,'#0e1a2c'],[1,'#ffd173']],text:res.A.map(r=>r.map(v=>fmt(v,2))),texttemplate:'%{text}',hoverinfo:'skip',showscale:false}]} layout={{xaxis:{type:'category',title:{text:'key (được nhìn)'}},yaxis:{type:'category',autorange:'reversed',title:{text:'query (đang nhìn)'}}}}/>
        <p className="small-note">Mỗi hàng cộng lại bằng 1. Bật mặt nạ nhân quả: nửa trên đường chéo về 0. Tắt "chia √d": softmax nhọn hơn, gần như dồn hết vào một token.</p>
      </SimulationPanel>
      <SimulationPanel title={`"${TOKENS[query]}" chú ý tới các token`}>
        <Chart xTitle="Token" yTitle="Trọng số" height={220} traces={[{type:'bar',name:'Trọng số',x:TOKENS.map((t,i)=>`${i}:${t}`),y:res.A[query],marker:{color:VI}}]} layout={{xaxis:{type:'category'},yaxis:{range:[0,1]}}}/>
        <Chart height={200} traces={[{type:'heatmap',x:Array.from({length:d},(_,i)=>String(i)),y:TOKENS.map((t,i)=>`${i}:${t}`),z:res.PE,zmin:-1,zmax:1,colorscale:[[0,'#42d3e5'],[.5,'#0e1a2c'],[1,'#f28db9']],showscale:false,hoverinfo:'skip'}]} layout={{xaxis:{type:'category',title:{text:'Positional encoding · chiều'}},yaxis:{type:'category',autorange:'reversed'}}}/>
        <SimulationCode file="attention.py" lines={['scores = q @ k.transpose(-2, -1) / math.sqrt(d)','scores = scores.masked_fill(~mask, float("-inf"))','weights = scores.softmax(dim=-1)','out = weights @ v']} active={causal?1:scale?0:2}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 28: nhiều seed ─────────
function tCrit(df:number){const T=[12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145];return T[df-1]??1.96}
function SeedsLab(){
  const [n,setN]=useState(8);
  const runs=useMemo(()=>{const A:number[]=[],B:number[]=[];for(let s=0;s<12;s++){A.push(trainMinibatch(D_TR.X,D_TR.Y,D_VA.X,D_VA.Y,{hidden:8,lr:.3,batch:32,epochs:10,seed:100+s}).epochs[9].val);B.push(trainMinibatch(D_TR.X,D_TR.Y,D_VA.X,D_VA.Y,{hidden:32,lr:.3,batch:32,epochs:10,seed:100+s}).epochs[9].val)}return {A,B}},[]);
  const A=runs.A.slice(0,n),B=runs.B.slice(0,n),D=B.map((v,i)=>v-A[i]);
  const ci=(x:number[])=>tCrit(x.length-1)*sd(x)/Math.sqrt(x.length);
  const bestA=A.indexOf(Math.max(...A));
  return <>{digitNote}
    <p className="small-note">Cấu hình A: 8 nơ-ron ẩn. Cấu hình B: 32 nơ-ron ẩn. Cả hai: lr = 0,3, batch 32, 10 epoch; seed điều khiển khởi tạo và thứ tự batch. Cùng một seed dùng cho cả hai cấu hình.</p>
    <Range label="Số seed" value={n} min={2} max={12} change={setN}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Accuracy kiểm tra theo seed (đường nối = cùng seed)">
        <Chart xTitle="" yTitle="Accuracy" height={320} traces={[...A.map((a,i)=>({type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip',x:['A · 8 nơ-ron','B · 32 nơ-ron'],y:[a,B[i]],line:{color:'rgba(168,167,255,.35)',width:1}})),scatter('A',A.map(a=>[0,a]),CY,{x:A.map(()=> 'A · 8 nơ-ron'),marker:{color:CY,size:9}}),scatter('B',B.map(b=>[0,b]),PK,{x:B.map(()=> 'B · 32 nơ-ron'),marker:{color:PK,size:9}}),scatter(`Seed tốt nhất của A (${100+bestA})`,[[0,A[bestA]]],'rgba(0,0,0,0)',{x:['A · 8 nơ-ron'],marker:{size:16,color:'rgba(0,0,0,0)',line:{color:GD,width:2.5}}})]} layout={{xaxis:{type:'category'}}}/>
      </SimulationPanel>
      <SimulationPanel title="Tổng hợp">
        <Table headers={['','Trung bình','Độ lệch chuẩn','KTC 95% (t)','Lớn nhất']} rows={[['A',fmt(mean(A)),fmt(sd(A)),`± ${fmt(ci(A))}`,fmt(Math.max(...A))],['B',fmt(mean(B)),fmt(sd(B)),`± ${fmt(ci(B))}`,fmt(Math.max(...B))],['B − A (cặp)',fmt(mean(D)),fmt(sd(D)),`± ${fmt(ci(D))}`,'']]}/>
        <p>{mean(D)-ci(D)>0?'Khoảng tin cậy của hiệu cặp nằm hẳn trên 0: B tốt hơn A trên các seed này.':mean(D)+ci(D)<0?'Khoảng tin cậy của hiệu cặp nằm hẳn dưới 0: A tốt hơn B.':'Khoảng tin cậy của hiệu cặp chứa 0: chưa đủ bằng chứng để nói cấu hình nào tốt hơn.'}</p>
        <p className="small-note">Nếu chỉ báo cáo seed tốt nhất của A ({fmt(A[bestA])}), A trông tốt hơn trung bình thật của nó {fmt(A[bestA]-mean(A))} điểm. Với 2–3 seed, khoảng tin cậy rất rộng vì hệ số t lớn.</p>
        <SimulationCode file="seeds.py" lines={['scores = run_seeds(train_once, seeds)','summary = summarize(scores)          # mean ± t·std/√n','diff = paired_compare(scores_B, scores_A)']} active={2}/>
      </SimulationPanel>
    </div></>;
}

export function WeekFourLab({day}:{day:number}){
  if(day===22)return <PerceptronLab/>;
  if(day===23)return <BackpropLab/>;
  if(day===24)return <TrainLoopLab/>;
  if(day===25)return <OptimLab/>;
  if(day===26)return <ConvLab/>;
  if(day===27)return <AttentionLab/>;
  return <SeedsLab/>;
}
