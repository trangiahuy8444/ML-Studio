// Mô phỏng tuần 2 (Ngày 8–14): phân loại và đánh giá. Mọi số đều tính trực tiếp trong trình duyệt.
// Dữ liệu: Breast Cancer Wisconsin (2 đặc trưng đã chuẩn hóa, y = 1 là ác tính) và điểm số logistic
// trên tập validation, trích sẵn bằng scikit-learn (xem data/breast2d.json). Phần dữ liệu nhiễu được ghi rõ là giả lập.
import {useMemo,useState} from 'react';
import {Chart} from '../components/Chart';
import {Range,Table,Transport,usePlayback} from '../components/ObservationControls';
import {SimulationCode,SimulationPanel} from '../components/SimulationStudio';
import type {PlotTrace} from '../types';
import breast from './data/breast2d.json';

const P=breast.points as [number,number,number][];
const X=P.map(p=>[p[0],p[1]]),Y=P.map(p=>p[2]);
const CY='#42d3e5',PK='#f28db9',GD='#ffd173',GR='#84e0a8',VI='#a8a7ff';
const fmt=(v:number,d=3)=>Number.isFinite(v)?v.toLocaleString('vi-VN',{maximumFractionDigits:d}):'—';
const mean=(a:number[])=>a.reduce((s,v)=>s+v,0)/a.length;
const sigmoid=(z:number)=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
function rng(seed:number){let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
function gauss(r:()=>number){return Math.sqrt(-2*Math.log(Math.max(r(),1e-12)))*Math.cos(2*Math.PI*r())}
const scatter=(name:string,pts:number[][],color:string,extra:Record<string,unknown>={}):PlotTrace=>({type:'scatter',mode:'markers',name,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),marker:{color,size:6,opacity:.85},...extra});
const byClass=(pts:number[][],ys:number[])=>[scatter('Lành tính (0)',pts.filter((_,i)=>ys[i]===0),CY),scatter('Ác tính (1)',pts.filter((_,i)=>ys[i]===1),PK)];
const gridAxes={xaxis:{title:{text:'mean radius (chuẩn hóa)'},range:[-2.2,4]},yaxis:{title:{text:'mean texture (chuẩn hóa)'},range:[-2.5,4.5]}};
function regionHeat(f:(x:number,y:number)=>number,name='p(ác tính)'):PlotTrace{
  const xs=Array.from({length:46},(_,i)=>-2.2+i*6.2/45),ys=Array.from({length:46},(_,i)=>-2.5+i*7/45);
  return {type:'heatmap',x:xs,y:ys,z:ys.map(y=>xs.map(x=>f(x,y))),zmin:0,zmax:1,showscale:false,opacity:.55,name,hoverinfo:'skip',colorscale:[[0,'#123a52'],[.5,'#1b2238'],[1,'#57264a']]};
}
const source=<p className="small-note">Dữ liệu: {breast.source}. Hai đặc trưng: mean radius và mean texture, đã chuẩn hóa.</p>;

// ───────── Ngày 8: logistic bằng gradient descent ─────────
function LogisticLab(){
  const [lr,setLR]=useState(.5);
  const hist=useMemo(()=>{let w=[0,0,0];const out=[{w,loss:Math.log(2),grad:[0,0,0]}];
    for(let s=0;s<60;s++){const g=[0,0,0];let loss=0;
      X.forEach((x,i)=>{const p=sigmoid(w[0]+w[1]*x[0]+w[2]*x[1]);const e=p-Y[i];g[0]+=e/X.length;g[1]+=e*x[0]/X.length;g[2]+=e*x[1]/X.length});
      w=w.map((v,j)=>v-lr*g[j]);
      X.forEach((x,i)=>{const p=Math.min(1-1e-12,Math.max(1e-12,sigmoid(w[0]+w[1]*x[0]+w[2]*x[1])));loss-=(Y[i]*Math.log(p)+(1-Y[i])*Math.log(1-p))/X.length});
      out.push({w,loss,grad:g})}
    return out},[lr]);
  const count=(hist.length-1)*4+1,player=usePlayback(count,hist);
  const done=Math.floor(player.index/4),phase=player.index===0?-1:(player.index-1)%4,fr=hist[done],op=hist[Math.ceil(player.index/4)];
  const [w0,w1,w2]=fr.w;const bx=[-2.2,4];
  const boundary:PlotTrace=Math.abs(w2)>1e-6?{type:'scatter',mode:'lines',name:'σ(z) = 0,5',x:bx,y:bx.map(x=>-(w0+w1*x)/w2),line:{color:GD,width:3}}:{type:'scatter',mode:'lines',x:[],y:[],name:'σ(z) = 0,5'};
  const acc=mean(X.map((x,i)=>Number((sigmoid(w0+w1*x[0]+w2*x[1])>=.5)===(Y[i]===1))));
  return <>{source}<Range label="Learning rate α" value={lr} min={.05} max={3} step={.05} change={setLR}/>
    <Transport player={player} count={count}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Vùng xác suất và ranh giới quyết định">
        <Chart traces={[regionHeat((x,y)=>sigmoid(w0+w1*x+w2*y)),...byClass(X,Y),boundary]} layout={gridAxes}/>
        <p className="small-note">Nền: p(ác tính) theo mô hình hiện tại. Đường vàng: z = w₀ + w₁x₁ + w₂x₂ = 0.</p>
      </SimulationPanel>
      <SimulationPanel title="Log-loss theo bước">
        <Chart xTitle="Bước" yTitle="Log-loss" traces={[{type:'scatter',mode:'lines',name:'Log-loss',x:hist.map((_,i)=>i),y:hist.map(h=>h.loss),line:{color:PK}},{type:'scatter',mode:'markers',name:'Hiện tại',x:[done],y:[fr.loss],marker:{color:GD,size:11}}]} height={260}/>
        <Table headers={['w₀','w₁','w₂','Log-loss','Accuracy']} rows={[[fmt(w0),fmt(w1),fmt(w2),fmt(fr.loss,4),fmt(acc)]]}/>
      </SimulationPanel>
      <SimulationPanel title="Python của một bước">
        <SimulationCode file="logistic.py" lines={['z = Xb @ w','p = sigmoid(z)','grad = Xb.T @ (p - y) / n','w = w - alpha * grad']} active={phase}
          detail={phase<0?<p>w = 0 nên mọi xác suất bằng 0,5, loss = ln 2 ≈ 0,693.</p>:<p>Gradient bước {Math.ceil(player.index/4)}: [{op.grad.map(v=>fmt(v,4)).join('; ')}]</p>}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 9: ngưỡng, ROC/PR, mất cân bằng ─────────
function ThresholdLab(){
  const [which,setWhich]=useState<'score30'|'score2'>('score2'),[th,setTh]=useState(.5),[keep,setKeep]=useState(100);
  const {ys,ss}=useMemo(()=>{const y0=breast.validation.y,s0=breast.validation[which];const r=rng(9);
    const idx=y0.map((_,i)=>i).filter(i=>y0[i]===0||r()*100<keep);return {ys:idx.map(i=>y0[i]),ss:idx.map(i=>s0[i])}},[which,keep]);
  const counts=(t:number)=>{let tp=0,fp=0,fn=0,tn=0;ss.forEach((s,i)=>{const p=s>=t;if(ys[i]){p?tp++:fn++}else{p?fp++:tn++}});return {tp,fp,fn,tn}};
  const c=counts(th),prec=c.tp+c.fp?c.tp/(c.tp+c.fp):0,rec=c.tp/(c.tp+c.fn||1),acc=(c.tp+c.tn)/ys.length,f1=prec+rec?2*prec*rec/(prec+rec):0;
  const curve=useMemo(()=>{const ts=[...new Set(ss)].sort((a,b)=>b-a);const pts=ts.map(t=>{const k=counts(t);return {fpr:k.fp/(k.fp+k.tn||1),tpr:k.tp/(k.tp+k.fn||1),prec:k.tp/(k.tp+k.fp||1)}});
    const roc=[{fpr:0,tpr:0},...pts];let auc=0;for(let i=1;i<roc.length;i++)auc+=(roc[i].fpr-roc[i-1].fpr)*(roc[i].tpr+roc[i-1].tpr)/2;return {pts,auc}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[ss,ys]);
  const posRate=mean(ys);
  return <>
    <p className="small-note">Điểm số p(ác tính) của logistic trên 171 mẫu validation (huấn luyện bằng scikit-learn trên 70% còn lại). "Giữ lớp dương" bỏ ngẫu nhiên bớt mẫu ác tính để mô phỏng lớp hiếm như dữ liệu gian lận.</p>
    <div className="py30-grid">
      <label>Mô hình<select value={which} onChange={e=>setWhich(e.target.value as 'score30'|'score2')}><option value="score2">Logistic 2 đặc trưng (chồng lấn)</option><option value="score30">Logistic 30 đặc trưng</option></select></label>
      <Range label="Ngưỡng t" value={th} min={.01} max={.99} step={.01} change={setTh}/>
      <Range label="Giữ lớp dương (%)" value={keep} min={5} max={100} step={5} change={setKeep}/>
    </div>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Phân phối điểm số theo lớp">
        <Chart xTitle="p(ác tính)" yTitle="Số mẫu" height={260} traces={[{type:'histogram',name:'Lành tính',x:ss.filter((_,i)=>!ys[i]),xbins:{start:0,end:1,size:.05},marker:{color:CY},opacity:.7},{type:'histogram',name:'Ác tính',x:ss.filter((_,i)=>ys[i]),xbins:{start:0,end:1,size:.05},marker:{color:PK},opacity:.75},{type:'scatter',mode:'lines',name:'Ngưỡng',x:[th,th],y:[0,60],line:{color:GD,width:3}}]} layout={{barmode:'overlay'}}/>
        <Table headers={['','Dự đoán 0','Dự đoán 1']} rows={[['Thật 0',c.tn,c.fp],['Thật 1',c.fn,c.tp]]}/>
        <Table headers={['Precision','Recall','F1','Accuracy','Tỉ lệ lớp dương']} rows={[[fmt(prec),fmt(rec),fmt(f1),fmt(acc),fmt(posRate)]]}/>
      </SimulationPanel>
      <SimulationPanel title={`ROC (AUC = ${fmt(curve.auc)}) và Precision–Recall`}>
        <Chart xTitle="FPR" yTitle="TPR" height={250} traces={[{type:'scatter',mode:'lines',name:'ROC',x:[0,...curve.pts.map(p=>p.fpr)],y:[0,...curve.pts.map(p=>p.tpr)],line:{color:CY,shape:'hv'}},{type:'scatter',mode:'lines',name:'Ngẫu nhiên',x:[0,1],y:[0,1],line:{color:'#41536a',dash:'dot'}},{type:'scatter',mode:'markers',name:'Ngưỡng hiện tại',x:[c.fp/(c.fp+c.tn||1)],y:[rec],marker:{color:GD,size:12}}]} layout={{xaxis:{range:[0,1]},yaxis:{range:[0,1.02]}}}/>
        <Chart xTitle="Recall" yTitle="Precision" height={250} traces={[{type:'scatter',mode:'lines',name:'PR',x:curve.pts.map(p=>p.tpr),y:curve.pts.map(p=>p.prec),line:{color:PK}},{type:'scatter',mode:'lines',name:'Mức cơ sở',x:[0,1],y:[posRate,posRate],line:{color:'#41536a',dash:'dot'}},{type:'scatter',mode:'markers',name:'Ngưỡng hiện tại',x:[rec],y:[prec],marker:{color:GD,size:12}}]} layout={{xaxis:{range:[0,1]},yaxis:{range:[0,1.02]}}}/>
      </SimulationPanel>
    </div>
    <p className="small-note">Thử: giảm "Giữ lớp dương" xuống 10%. Đường ROC gần như không đổi, đường PR tụt mạnh, còn accuracy tăng dù mô hình không tốt hơn.</p></>;
}

// ───────── Ngày 10: KNN và lời nguyền số chiều ─────────
function KnnLab(){
  const [k,setK]=useState(5),[scale,setScale]=useState(1),[qx,setQx]=useState(.6),[qy,setQy]=useState(.4);
  const pts=useMemo(()=>X.map(p=>[p[0],p[1]*scale]),[scale]);
  const vote=(x:number,y:number)=>{const d=pts.map((p,i)=>({d:(p[0]-x)**2+(p[1]-y*scale)**2,i})).sort((a,b)=>a.d-b.d).slice(0,k);return {near:d.map(o=>o.i),p:mean(d.map(o=>Y[o.i]))}};
  const q=vote(qx,qy);
  const heat=useMemo(()=>regionHeat((x,y)=>{const d=pts.map((p,i)=>({d:(p[0]-x)**2+(p[1]-y*scale)**2,i})).sort((a,b)=>a.d-b.d).slice(0,k);return mean(d.map(o=>Y[o.i]))},'Tỉ lệ phiếu ác tính'),[k,scale,pts]);
  const curse=useMemo(()=>{const ds=[1,2,5,10,20,50,100,300,1000];const r=rng(10);return ds.map(d=>{const n=200;const pts=Array.from({length:n},()=>Array.from({length:d},()=>r()));
    let acc=0;for(let qi=0;qi<10;qi++){let mn=Infinity,mx=0;for(let j=0;j<n;j++){if(j===qi)continue;let s=0;for(let t=0;t<d;t++)s+=(pts[j][t]-pts[qi][t])**2;const v=Math.sqrt(s);mn=Math.min(mn,v);mx=Math.max(mx,v)}acc+=(mx-mn)/mn}return {d,c:acc/10}})},[]);
  const lines:PlotTrace={type:'scatter',mode:'lines',name:'k láng giềng',x:q.near.flatMap(i=>[qx,X[i][0],null]),y:q.near.flatMap(i=>[qy,X[i][1],null]),line:{color:GD,width:1}};
  return <>{source}
    <div className="py30-grid"><Range label="k" value={k} min={1} max={51} step={2} change={setK}/><Range label="Phóng to thang đo cột 2 ×" value={scale} min={1} max={20} change={setScale}/>
      <Range label="Điểm truy vấn x₁" value={qx} min={-2} max={3.8} step={.05} change={setQx}/><Range label="Điểm truy vấn x₂" value={qy} min={-2.3} max={4.2} step={.05} change={setQy}/></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Vùng quyết định · phiếu ác tính ${Math.round(q.p*k)}/${k}`}>
        <Chart traces={[heat,...byClass(X,Y),lines,{type:'scatter',mode:'markers',name:'Truy vấn',x:[qx],y:[qy],marker:{color:GD,size:14,symbol:'diamond',line:{color:'#fff',width:1}}}]} layout={gridAxes}/>
        <p className="small-note">Khi phóng to thang đo cột 2, khoảng cách gần như chỉ còn phụ thuộc cột đó: vùng quyết định thành các dải ngang. Đó là lý do phải chuẩn hóa.</p>
      </SimulationPanel>
      <SimulationPanel title="Lời nguyền số chiều (điểm đều ngẫu nhiên, giả lập)">
        <Chart xTitle="Số chiều d" yTitle="(d_max − d_min) / d_min" height={260} traces={[{type:'scatter',mode:'lines+markers',name:'Độ tương phản',x:curse.map(c=>c.d),y:curse.map(c=>c.c),line:{color:VI}}]} layout={{xaxis:{type:'log',title:{text:'Số chiều d'}},yaxis:{type:'log',title:{text:'(d_max − d_min) / d_min'}}}}/>
        <SimulationCode file="knn.py" lines={['D = ((X_test[:, None] - X_train[None]) ** 2).sum(-1)','nearest = np.argsort(D, axis=1)[:, :k]','votes = np.bincount(y_train[nearest[0]], minlength=2)','pred = votes.argmax()']} active={3}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 11: Naive Bayes, LDA, QDA ─────────
type Gauss={mu:number[];S:number[][];prior:number};
function fitGauss(mode:'nb'|'lda'|'qda'):Gauss[]{
  const cls=[0,1].map(c=>{const pts=X.filter((_,i)=>Y[i]===c);const mu=[mean(pts.map(p=>p[0])),mean(pts.map(p=>p[1]))];
    const cov=[[0,0],[0,0]];pts.forEach(p=>{const d=[p[0]-mu[0],p[1]-mu[1]];for(let a=0;a<2;a++)for(let b=0;b<2;b++)cov[a][b]+=d[a]*d[b]/pts.length});return {mu,S:cov,prior:pts.length/X.length,n:pts.length}});
  if(mode==='lda'){const S=[[0,0],[0,0]];cls.forEach(c=>{for(let a=0;a<2;a++)for(let b=0;b<2;b++)S[a][b]+=c.S[a][b]*c.n/X.length});return cls.map(c=>({mu:c.mu,S,prior:c.prior}))}
  if(mode==='nb')return cls.map(c=>({mu:c.mu,S:[[c.S[0][0],0],[0,c.S[1][1]]],prior:c.prior}));
  return cls.map(c=>({mu:c.mu,S:c.S,prior:c.prior}));
}
function logDens(g:Gauss,x:number,y:number){const [[a,b],[,d]]=g.S;const det=a*d-b*b;const dx=x-g.mu[0],dy=y-g.mu[1];const q=(d*dx*dx-2*b*dx*dy+a*dy*dy)/det;return Math.log(g.prior)-.5*Math.log(det)-.5*q}
function ellipse(g:Gauss,k:number,color:string,name:string):PlotTrace{const [[a,b],[,d]]=g.S;const tr=a+d,det=a*d-b*b,l1=tr/2+Math.sqrt(tr*tr/4-det),l2=tr/2-Math.sqrt(tr*tr/4-det);const ang=Math.abs(b)<1e-12?(a>=d?0:Math.PI/2):Math.atan2(l1-a,b);
  const ts=Array.from({length:61},(_,i)=>i/60*2*Math.PI);return {type:'scatter',mode:'lines',name,showlegend:k===1,x:ts.map(t=>g.mu[0]+k*(Math.sqrt(l1)*Math.cos(t)*Math.cos(ang)-Math.sqrt(l2)*Math.sin(t)*Math.sin(ang))),y:ts.map(t=>g.mu[1]+k*(Math.sqrt(l1)*Math.cos(t)*Math.sin(ang)+Math.sqrt(l2)*Math.sin(t)*Math.cos(ang))),line:{color,width:k===1?2:1,dash:k===1?'solid':'dot'}}}
function BayesLab(){
  const [mode,setMode]=useState<'nb'|'lda'|'qda'>('nb');
  const G=useMemo(()=>fitGauss(mode),[mode]);
  const post=(x:number,y:number)=>{const a=logDens(G[0],x,y),b=logDens(G[1],x,y),m=Math.max(a,b);return Math.exp(b-m)/(Math.exp(a-m)+Math.exp(b-m))};
  const acc=mean(X.map((p,i)=>Number((post(p[0],p[1])>=.5)===(Y[i]===1))));
  const names={nb:'Gaussian Naive Bayes: hiệp phương sai chéo, riêng từng lớp',lda:'LDA: một ma trận hiệp phương sai chung',qda:'QDA: hiệp phương sai đầy đủ, riêng từng lớp'};
  return <>{source}
    <label>Mô hình<select value={mode} onChange={e=>setMode(e.target.value as 'nb'|'lda'|'qda')}><option value="nb">Naive Bayes</option><option value="lda">LDA</option><option value="qda">QDA</option></select></label>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`${names[mode]} · accuracy trên toàn bộ ${X.length} mẫu = ${fmt(acc)}`}>
        <Chart traces={[regionHeat(post,'p(ác tính | x)'),...byClass(X,Y),...[1,2].flatMap(k=>[ellipse(G[0],k,CY,'Gauss lớp 0'),ellipse(G[1],k,PK,'Gauss lớp 1')])]} layout={gridAxes}/>
        <p className="small-note">Elip là đường 1σ và 2σ của phân phối ước lượng cho mỗi lớp. Naive Bayes có elip song song trục vì giả định các đặc trưng độc lập khi biết lớp; LDA cho ranh giới thẳng, QDA cho ranh giới cong.</p>
      </SimulationPanel>
      <SimulationPanel title="Tham số ước lượng (MLE)">
        <Table headers={['Lớp','Tiên nghiệm','μ₁','μ₂','σ₁²','σ₂²','σ₁₂']} rows={G.map((g,i)=>[i?'Ác tính':'Lành tính',fmt(g.prior),fmt(g.mu[0]),fmt(g.mu[1]),fmt(g.S[0][0]),fmt(g.S[1][1]),fmt(g.S[0][1])])}/>
        <SimulationCode file="naive_bayes.py" lines={['prior[c] = np.mean(y == c)','mean[c] = X[y == c].mean(0); var[c] = X[y == c].var(0)','jll = np.log(prior) - 0.5*np.log(2*np.pi*var).sum(1) - 0.5*((x - mean)**2 / var).sum(1)','posterior = np.exp(jll - logsumexp(jll))']} active={2}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 12: SVM tuyến tính bằng subgradient ─────────
function SvmLab(){
  const [logC,setLogC]=useState(0);const C=10**logC;
  const S=useMemo(()=>X.map((_,i)=>Y[i]?1:-1),[]);
  const hist=useMemo(()=>{let w=[0,0],b=0;const obj=(w:number[],b:number)=>.5*(w[0]**2+w[1]**2)+C*X.reduce((s,x,i)=>s+Math.max(0,1-S[i]*(w[0]*x[0]+w[1]*x[1]+b)),0);
    const out=[{w,b,obj:obj(w,b)}];let best=out[0];const lr0=.5/(C*X.length)**.5;
    for(let t=1;t<=120;t++){const gw=[w[0],w[1]];let gb=0;X.forEach((x,i)=>{if(S[i]*(w[0]*x[0]+w[1]*x[1]+b)<1){gw[0]-=C*S[i]*x[0];gw[1]-=C*S[i]*x[1];gb-=C*S[i]}});
      const lr=lr0/Math.sqrt(t);w=[w[0]-lr*gw[0],w[1]-lr*gw[1]];b-=lr*gb;const o=obj(w,b);if(o<best.obj)best={w,b,obj:o};out.push(best)}
    return out},[C,S]);
  const player=usePlayback(hist.length,hist);const fr=hist[player.index];const [w1,w2]=fr.w;const bx=[-2.2,4];
  const line=(off:number,name:string,color:string,dash='solid'):PlotTrace=>({type:'scatter',mode:'lines',name,x:bx,y:bx.map(x=>Math.abs(w2)>1e-9?(off-fr.b-w1*x)/w2:NaN),line:{color,width:off?1.5:3,dash}});
  const margins=X.map((x,i)=>S[i]*(w1*x[0]+w2*x[1]+fr.b));const sv=X.filter((_,i)=>margins[i]<=1+1e-9);const nrm=Math.hypot(w1,w2);
  return <>{source}<Range label="log₁₀ C" value={logC} min={-3} max={1} step={.25} change={setLogC}/>
    <Transport player={player} count={hist.length}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Lề 2/‖w‖ = ${fmt(nrm?2/nrm:Infinity)} · ${sv.length} điểm nằm trên hoặc trong lề`}>
        <Chart traces={[...byClass(X,Y),scatter('Vector hỗ trợ / vi phạm lề',sv,'rgba(0,0,0,0)',{marker:{size:10,color:'rgba(0,0,0,0)',line:{color:'#ffffff',width:1.2}}}),line(0,'w·x + b = 0',GR),line(1,'w·x + b = +1',GD,'dash'),line(-1,'w·x + b = −1',GD,'dash')]} layout={gridAxes}/>
        <p className="small-note">C nhỏ: lề rộng, chấp nhận nhiều điểm vi phạm. C lớn: lề hẹp, cố phân loại đúng từng điểm. Chỉ các điểm có viền trắng đóng góp vào subgradient.</p>
      </SimulationPanel>
      <SimulationPanel title="Hàm mục tiêu tốt nhất đã gặp">
        <Chart xTitle="Bước" yTitle="½‖w‖² + CΣhinge" height={240} traces={[{type:'scatter',mode:'lines',name:'Mục tiêu',x:hist.map((_,i)=>i),y:hist.map(h=>h.obj),line:{color:PK}},{type:'scatter',mode:'markers',name:'Hiện tại',x:[player.index],y:[fr.obj],marker:{color:GD,size:11}}]}/>
        <SimulationCode file="svm.py" lines={['margin = y * (X @ w + b)','active = margin < 1','gw = w - C * (y[active, None] * X[active]).sum(0)','w, b = w - lr * gw, b + lr * C * y[active].sum()']} active={player.index?2:-1}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 13: cross-validation và rò rỉ ─────────
function noiseExperiment(m:number,reps:number){
  const n=50,p=1000,k=5;const leaky:number[]=[],honest:number[]=[];
  const corrTop=(Xs:number[][],ys:number[],rows:number[])=>{const my=mean(rows.map(i=>ys[i]));const sc:number[]=[];
    for(let j=0;j<p;j++){let mx=0;for(const i of rows)mx+=Xs[i][j];mx/=rows.length;let sxy=0,sxx=0,syy=0;for(const i of rows){const a=Xs[i][j]-mx,b=ys[i]-my;sxy+=a*b;sxx+=a*a;syy+=b*b}sc.push(Math.abs(sxy/Math.sqrt(sxx*syy+1e-12)))}
    return sc.map((v,j)=>[v,j]).sort((a,b)=>b[0]-a[0]).slice(0,m).map(v=>v[1])};
  const evalFold=(Xs:number[][],ys:number[],tr:number[],va:number[],cols:number[])=>{const cen=[0,1].map(c=>cols.map(j=>mean(tr.filter(i=>ys[i]===c).map(i=>Xs[i][j]))));
    return mean(va.map(i=>{const d=cen.map(ce=>cols.reduce((s,j,t)=>s+(Xs[i][j]-ce[t])**2,0));return Number((d[1]<d[0]?1:0)===ys[i])}))};
  for(let r=0;r<reps;r++){const g=rng(100+r);const Xs=Array.from({length:n},()=>Array.from({length:p},()=>gauss(g)));const ys=Array.from({length:n},(_,i)=>i<n/2?0:1);
    const order=Array.from({length:n},(_,i)=>i).sort(()=>g()-.5);const folds=Array.from({length:k},(_,f)=>order.filter((_,i)=>i%k===f));
    const all=Array.from({length:n},(_,i)=>i);const leakCols=corrTop(Xs,ys,all);let a=0,b=0;
    folds.forEach(va=>{const tr=all.filter(i=>!va.includes(i));a+=evalFold(Xs,ys,tr,va,leakCols);b+=evalFold(Xs,ys,tr,va,corrTop(Xs,ys,tr))});
    leaky.push(a/k);honest.push(b/k)}
  return {leaky,honest};
}
function CvLab(){
  const [m,setM]=useState(20),[k,setK]=useState(5),[run,setRun]=useState(0);
  const exp=useMemo(()=>noiseExperiment(m,24),[m,run]);// eslint-disable-line react-hooks/exhaustive-deps
  const folds=useMemo(()=>{const r=rng(13);const order=[...[0,1].flatMap(c=>Y.map((y,i)=>y===c?i:-1).filter(i=>i>=0).sort(()=>r()-.5))];const f=new Array(Y.length);order.forEach((idx,j)=>f[idx]=j%k);return f},[k]);
  const player=usePlayback(k,k);const cur=player.index;
  return <>
    <p className="small-note">Thí nghiệm giả lập: 50 mẫu, 1000 cột nhiễu Gauss, nhãn không liên quan gì tới dữ liệu. Chọn m cột tương quan mạnh nhất rồi phân loại bằng tâm lớp, 5-fold, lặp 24 lần.</p>
    <div className="py30-grid"><Range label="Số cột được chọn m" value={m} min={2} max={100} step={2} change={setM}/><button className="btn-outline" onClick={()=>setRun(v=>v+1)}>Chạy lại</button></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Rò rỉ: ${fmt(mean(exp.leaky))} · Đúng cách: ${fmt(mean(exp.honest))} (mức đoán mò 0,5)`}>
        <Chart xTitle="Accuracy cross-validation" yTitle="Số lần lặp" height={280} traces={[{type:'histogram',name:'Chọn cột trước khi chia fold (sai)',x:exp.leaky,xbins:{start:0,end:1,size:.05},marker:{color:PK},opacity:.75},{type:'histogram',name:'Chọn cột trong từng fold (đúng)',x:exp.honest,xbins:{start:0,end:1,size:.05},marker:{color:CY},opacity:.75}]} layout={{barmode:'overlay',xaxis:{range:[0,1]}}}/>
        <SimulationCode file="cv.py" lines={['for train, val in kfold_indices(n, k, seed):','    cols = select_top(X[train], y[train], m)   # chỉ train','    model = centroid_fit(X[train][:, cols], y[train])','    accs.append(accuracy(model, X[val][:, cols], y[val]))']} active={1}/>
      </SimulationPanel>
      <SimulationPanel title={`Stratified ${k}-fold trên Breast Cancer · fold ${cur+1} làm validation`}>
        <Range label="k" value={k} min={3} max={10} change={setK}/><Transport player={player} count={k}/>
        <Table headers={['Fold','Số mẫu','Ác tính','Tỉ lệ ác tính','Vai trò']} rows={Array.from({length:k},(_,f)=>{const ids=Y.map((_,i)=>i).filter(i=>folds[i]===f);return [f+1,ids.length,ids.filter(i=>Y[i]).length,fmt(mean(ids.map(i=>Y[i]))),f===cur?'validation':'train']})} active={cur}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 14: pipeline Titanic ─────────
const stages=[
  {name:'Dữ liệu gốc',row:{Age:'NaN',Fare:'7,25',Sex:'male',Pclass:'3',Embarked:'X'},note:'Age thiếu; Embarked = "X" không xuất hiện khi fit.'},
  {name:'SimpleImputer',row:{Age:'median(Age) của fold train',Fare:'7,25',Sex:'male',Pclass:'3',Embarked:'X'},note:'Median và mode lấy từ fold train, không từ toàn bộ dữ liệu.'},
  {name:'OneHotEncoder',row:{Age:'28',Fare:'7,25',Sex:'[0, 1]',Pclass:'[0, 0, 1]',Embarked:'[0, 0, 0]'},note:'handle_unknown="ignore": giá trị mới thành vector 0 thay vì báo lỗi.'},
  {name:'StandardScaler',row:{Age:'(Age − mean)/std',Fare:'(Fare − mean)/std',Sex:'[0, 1]',Pclass:'[0, 0, 1]',Embarked:'[0, 0, 0]'},note:'Chuẩn hóa cột số bằng mean/std của fold train.'},
  {name:'LogisticRegression',row:{Age:'',Fare:'',Sex:'',Pclass:'',Embarked:''},note:'Mô hình tính p(sống) = σ(w·x + b) trên vector đã biến đổi; nhãn = 1 khi p ≥ 0,5.'},
];
function PipelineLab(){
  const player=usePlayback(stages.length,'pipe');const s=stages[player.index];
  return <>
    <p className="small-note">Một hàng minh họa có cấu trúc như Titanic (chỉ minh họa phép biến đổi ở từng bước; không có số liệu mô hình).</p>
    <Transport player={player} count={stages.length}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Bước ${player.index+1}/${stages.length}: ${s.name}`}>
        <div className="pipeline-flow">{stages.map((st,i)=><span key={st.name} className={i===player.index?'active':i<player.index?'done':''}>{st.name}</span>)}</div>
        {player.index<stages.length-1?<Table headers={['Age','Fare','Sex','Pclass','Embarked']} rows={[[s.row.Age,s.row.Fare,s.row.Sex,s.row.Pclass,s.row.Embarked]]}/>:null}
        <p>{s.note}</p>
      </SimulationPanel>
      <SimulationPanel title="Python">
        <SimulationCode file="titanic_pipeline.py" lines={['row = test.loc[[i], NUMERIC + CATEGORICAL]','num = SimpleImputer(strategy="median"); cat = SimpleImputer(strategy="most_frequent")','onehot = OneHotEncoder(handle_unknown="ignore")','scale = StandardScaler()','pipe.predict_proba(row)[:, 1]']} active={player.index}/>
      </SimulationPanel>
    </div></>;
}

export function WeekTwoLab({day}:{day:number}){
  if(day===8)return <LogisticLab/>;
  if(day===9)return <ThresholdLab/>;
  if(day===10)return <KnnLab/>;
  if(day===11)return <BayesLab/>;
  if(day===12)return <SvmLab/>;
  if(day===13)return <CvLab/>;
  return <PipelineLab/>;
}
