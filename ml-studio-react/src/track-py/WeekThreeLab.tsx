// Mô phỏng tuần 3 (Ngày 15–21): cây, tổ hợp và học không nhãn. Mọi thuật toán chạy thật trong trình duyệt.
// Dữ liệu thật: Breast Cancer Wisconsin (2 cột chuẩn hóa, data/breast2d.json), Real estate valuation (UCI, data/housing.json),
// Iris (data/iris2d.json), load_digits (data/digits.json, PCA và bảng CV do scikit-learn tính sẵn — xem scripts/gen_week3_data.py).
// Phần dữ liệu tổng hợp (Ngày 20) được ghi rõ là giả lập.
import {useMemo,useState} from 'react';
import {Chart} from '../components/Chart';
import {Range,Table,Transport,usePlayback} from '../components/ObservationControls';
import {SimulationCode,SimulationPanel} from '../components/SimulationStudio';
import {housing} from '../data';
import type {PlotTrace} from '../types';
import breast from './data/breast2d.json';
import digits from './data/digits.json';
import iris from './data/iris2d.json';

const CY='#42d3e5',PK='#f28db9',GD='#ffd173',GR='#84e0a8',VI='#a8a7ff',OR='#ff9b6a';
const PAL=[CY,PK,GD,GR,VI,OR,'#7fb3ff','#e6e6e6','#c49bff','#ffa3a3'];
const fmt=(v:number,d=3)=>Number.isFinite(v)?v.toLocaleString('vi-VN',{maximumFractionDigits:d}):'—';
const mean=(a:number[])=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;
function rng(seed:number){let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
function shuffle<T>(a:T[],r:()=>number){const b=[...a];for(let i=b.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[b[i],b[j]]=[b[j],b[i]]}return b}

const BP=breast.points as [number,number,number][];
const BX=BP.map(p=>[p[0],p[1]]),BY=BP.map(p=>p[2]);
const split70=(n:number,seed:number)=>{const o=shuffle(Array.from({length:n},(_,i)=>i),rng(seed));const c=Math.round(.7*n);return {tr:o.slice(0,c),va:o.slice(c)}};
const BS=split70(BX.length,15);
const scatter=(name:string,pts:number[][],color:string,extra:Record<string,unknown>={}):PlotTrace=>({type:'scatter',mode:'markers',name,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),marker:{color,size:6,opacity:.85},...extra});
const byClass=(idx:number[])=>[scatter('Lành tính (0)',idx.filter(i=>BY[i]===0).map(i=>BX[i]),CY),scatter('Ác tính (1)',idx.filter(i=>BY[i]===1).map(i=>BX[i]),PK)];
const gridAxes={xaxis:{title:{text:'mean radius (chuẩn hóa)'},range:[-2.2,4]},yaxis:{title:{text:'mean texture (chuẩn hóa)'},range:[-2.5,4.5]}};
function regionHeat(f:(x:number,y:number)=>number,name='p(ác tính)'):PlotTrace{
  const xs=Array.from({length:46},(_,i)=>-2.2+i*6.2/45),ys=Array.from({length:46},(_,i)=>-2.5+i*7/45);
  return {type:'heatmap',x:xs,y:ys,z:ys.map(y=>xs.map(x=>f(x,y))),zmin:0,zmax:1,showscale:false,opacity:.55,name,hoverinfo:'skip',colorscale:[[0,'#123a52'],[.5,'#1b2238'],[1,'#57264a']]};
}
const breastNote=<p className="small-note">Dữ liệu: {breast.source}. Hai đặc trưng mean radius và mean texture đã chuẩn hóa; chia ngẫu nhiên 70% train / 30% validation trong trình duyệt (seed 15).</p>;

// ───────── CART dùng chung (phân loại, Gini, ngưỡng là trung điểm) ─────────
type Node={leaf:true;p:number;n:number}|{leaf:false;f:number;t:number;l:Node;r:Node;n:number;gain:number};
const gini=(pos:number,n:number)=>n?1-(pos/n)**2-(1-pos/n)**2:0;
function bestSplit(rows:number[],X:number[][],Y:number[],feats:number[]){
  const n=rows.length,pos=rows.reduce((s,i)=>s+Y[i],0),parent=gini(pos,n);let best={f:-1,t:0,gain:0};
  for(const f of feats){const o=[...rows].sort((a,b)=>X[a][f]-X[b][f]);let lp=0;
    for(let i=1;i<n;i++){lp+=Y[o[i-1]];const a=X[o[i-1]][f],b=X[o[i]][f];if(b<=a)continue;
      const g=parent-(i*gini(lp,i)+(n-i)*gini(pos-lp,n-i))/n;if(g>best.gain+1e-12)best={f,t:(a+b)/2,gain:g}}}
  return best;
}
function buildTree(rows:number[],X:number[][],Y:number[],depth:number,feats:(r:()=>number)=>number[],r:()=>number):Node{
  const n=rows.length,pos=rows.reduce((s,i)=>s+Y[i],0);
  if(depth===0||pos===0||pos===n)return {leaf:true,p:pos/n,n};
  const b=bestSplit(rows,X,Y,feats(r));if(b.f<0)return {leaf:true,p:pos/n,n};
  const L=rows.filter(i=>X[i][b.f]<=b.t),R=rows.filter(i=>X[i][b.f]>b.t);
  return {leaf:false,f:b.f,t:b.t,gain:b.gain,n,l:buildTree(L,X,Y,depth-1,feats,r),r:buildTree(R,X,Y,depth-1,feats,r)};
}
const predictP=(node:Node,x:number[]):number=>node.leaf?node.p:predictP(x[node.f]<=node.t?node.l:node.r,x);
const allFeats=()=>[0,1];
const acc=(f:(x:number[])=>number,idx:number[])=>mean(idx.map(i=>Number((f(BX[i])>=.5?1:0)===BY[i])));
function describe(node:Node,names:string[],depth=0,out:string[][]=[]):string[][]{
  if(depth>2)return out;
  if(node.leaf)out.push(['  '.repeat(depth)+'lá',String(node.n),fmt(node.p,2),'']);
  else{out.push(['  '.repeat(depth)+`${names[node.f]} ≤ ${fmt(node.t,2)}`,String(node.n),'',fmt(node.gain,3)]);describe(node.l,names,depth+1,out);describe(node.r,names,depth+1,out)}
  return out;
}

// ───────── Ngày 15: cây quyết định ─────────
function TreeLab(){
  const [depth,setDepth]=useState(3);
  const trees=useMemo(()=>Array.from({length:10},(_,d)=>buildTree(BS.tr,BX,BY,d+1,allFeats,rng(1))),[]);
  const tree=trees[depth-1];
  const curve=useMemo(()=>trees.map(t=>({tr:acc(x=>predictP(t,x),BS.tr),va:acc(x=>predictP(t,x),BS.va)})),[trees]);
  const scan=useMemo(()=>[0,1].map(f=>{const o=[...BS.tr].sort((a,b)=>BX[a][f]-BX[b][f]);const n=o.length,pos=o.reduce((s,i)=>s+BY[i],0),par=gini(pos,n);let lp=0;const pts:number[][]=[];
    for(let i=1;i<n;i++){lp+=BY[o[i-1]];const a=BX[o[i-1]][f],b=BX[o[i]][f];if(b>a)pts.push([(a+b)/2,par-(i*gini(lp,i)+(n-i)*gini(pos-lp,n-i))/n])}return pts}),[]);
  const lines:PlotTrace[]=[];
  const walk=(node:Node,box:[number,number,number,number])=>{if(node.leaf)return;const [x0,x1,y0,y1]=box;
    if(node.f===0){lines.push({type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip',x:[node.t,node.t],y:[y0,y1],line:{color:GD,width:1.5}});walk(node.l,[x0,node.t,y0,y1]);walk(node.r,[node.t,x1,y0,y1])}
    else{lines.push({type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip',x:[x0,x1],y:[node.t,node.t],line:{color:GD,width:1.5}});walk(node.l,[x0,x1,y0,node.t]);walk(node.r,[x0,x1,node.t,y1])}};
  walk(tree,[-2.2,4,-2.5,4.5]);
  return <>{breastNote}<Range label="max_depth" value={depth} min={1} max={10} change={setDepth}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Vùng quyết định · train ${fmt(curve[depth-1].tr)} · validation ${fmt(curve[depth-1].va)}`}>
        <Chart traces={[regionHeat((x,y)=>predictP(tree,[x,y]),'Tỉ lệ ác tính trong lá'),...byClass(BS.tr),...lines]} layout={gridAxes}/>
        <p className="small-note">Mỗi đường vàng là một phép tách xⱼ ≤ t. Cây chỉ cắt song song trục nên vùng quyết định là các hình chữ nhật.</p>
      </SimulationPanel>
      <SimulationPanel title="Accuracy theo độ sâu">
        <Chart xTitle="max_depth" yTitle="Accuracy" height={250} traces={[{type:'scatter',mode:'lines+markers',name:'Train',x:curve.map((_,i)=>i+1),y:curve.map(c=>c.tr),line:{color:GD}},{type:'scatter',mode:'lines+markers',name:'Validation',x:curve.map((_,i)=>i+1),y:curve.map(c=>c.va),line:{color:CY}},{type:'scatter',mode:'markers',name:'Đang xem',x:[depth],y:[curve[depth-1].va],marker:{color:PK,size:12}}]}/>
        <Table headers={['Nút (3 tầng đầu)','Số mẫu','Tỉ lệ ác tính','Giảm Gini']} rows={describe(tree,['radius','texture'])}/>
      </SimulationPanel>
      <SimulationPanel title="Quét ngưỡng ở nút gốc">
        <Chart xTitle="Ngưỡng t" yTitle="Mức giảm Gini" height={250} traces={scan.map((pts,f)=>({type:'scatter',mode:'lines',name:f?'mean texture':'mean radius',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),line:{color:f?PK:CY}}))}/>
        <SimulationCode file="cart.py" lines={['for j in range(d):','    xs, ys = sort(X[:, j], y)','    gain = gini(y) - (nL*gini(L) + nR*gini(R)) / n','    best = argmax(gain) tại trung điểm hai giá trị kề']} active={2}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 16: bagging và Random Forest ─────────
function ForestLab(){
  const [mode,setMode]=useState<'rf'|'bag'>('rf'),[count,setCount]=useState(30);
  const forest=useMemo(()=>{const r=rng(16);const feats=mode==='rf'?(g:()=>number)=>[g()<.5?0:1]:allFeats;
    return Array.from({length:60},()=>{const boot=BS.tr.map(()=>BS.tr[Math.floor(r()*BS.tr.length)]);const inBag=new Set(boot);
      return {tree:buildTree(boot,BX,BY,40,feats,r),boot,oob:BS.tr.filter(i=>!inBag.has(i))}})},[mode]);
  const curves=useMemo(()=>{const vote=new Map<number,number[]>();const out:{va:number;oob:number;single:number}[]=[];
    const vaVotes=BS.va.map(()=>0);
    forest.forEach((m,t)=>{BS.va.forEach((i,j)=>vaVotes[j]+=predictP(m.tree,BX[i])>=.5?1:0);
      m.oob.forEach(i=>{const v=vote.get(i)??[0,0];v[0]+=predictP(m.tree,BX[i])>=.5?1:0;v[1]++;vote.set(i,v)});
      const oobAcc=mean([...vote.entries()].map(([i,v])=>Number((v[0]/v[1]>=.5?1:0)===BY[i])));
      out.push({va:mean(BS.va.map((i,j)=>Number((vaVotes[j]/(t+1)>=.5?1:0)===BY[i]))),oob:oobAcc,single:acc(x=>predictP(m.tree,x),BS.va)})});
    return out},[forest]);
  const player=usePlayback(count,`${mode}-${count}`);const shown=forest[player.index];
  const used=forest.slice(0,count);
  const votes=(x:number,y:number)=>mean(used.map(m=>predictP(m.tree,[x,y])>=.5?1:0));
  const mult=new Map<number,number>();shown.boot.forEach(i=>mult.set(i,(mult.get(i)??0)+1));
  const inBag=[...mult.keys()];
  return <>{breastNote}
    <div className="py30-grid"><label>Cách chọn cột ở mỗi lần tách<select value={mode} onChange={e=>setMode(e.target.value as 'rf'|'bag')}><option value="rf">Random Forest: 1 cột ngẫu nhiên (max_features = 1)</option><option value="bag">Bagging: xét cả 2 cột</option></select></label>
      <Range label="Số cây" value={count} min={1} max={60} change={setCount}/></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Rừng ${count} cây · validation ${fmt(curves[count-1].va)} · OOB ${fmt(curves[count-1].oob)}`}>
        <Chart traces={[regionHeat(votes,'Tỉ lệ phiếu ác tính'),...byClass(BS.tr)]} layout={gridAxes}/>
        <p className="small-note">Nền là tỉ lệ cây bỏ phiếu "ác tính". Một cây sâu tạo ranh giới lởm chởm; trung bình nhiều cây làm ranh giới mượt hơn.</p>
      </SimulationPanel>
      <SimulationPanel title="Accuracy theo số cây">
        <Chart xTitle="Số cây" yTitle="Accuracy" height={260} traces={[{type:'scatter',mode:'lines',name:'Rừng · validation',x:curves.map((_,i)=>i+1),y:curves.map(c=>c.va),line:{color:CY}},{type:'scatter',mode:'lines',name:'Rừng · OOB (chỉ dùng train)',x:curves.map((_,i)=>i+1),y:curves.map(c=>c.oob),line:{color:GD}},{type:'scatter',mode:'markers',name:'Từng cây · validation',x:curves.map((_,i)=>i+1),y:curves.map(c=>c.single),marker:{color:PK,size:5,opacity:.6}}]}/>
        <p className="small-note">Accuracy trung bình từng cây: {fmt(mean(curves.slice(0,count).map(c=>c.single)))}. Đường OOB chỉ dùng tập train nhưng bám sát đường validation.</p>
      </SimulationPanel>
      <SimulationPanel title={`Mẫu bootstrap của cây ${player.index+1}: ${inBag.length}/${BS.tr.length} điểm khác nhau (${fmt(inBag.length/BS.tr.length*100,1)}%)`}>
        <Transport player={player} count={count}/>
        <Chart height={300} traces={[scatter('Trong túi (cỡ = số lần được chọn)',inBag.map(i=>BX[i]),GR,{marker:{color:GR,opacity:.75,size:inBag.map(i=>3+3*(mult.get(i)??1))}}),scatter('Out-of-bag',shown.oob.map(i=>BX[i]),'rgba(0,0,0,0)',{marker:{size:7,color:'rgba(0,0,0,0)',line:{color:OR,width:1.4}}})]} layout={gridAxes}/>
        <SimulationCode file="forest.py" lines={['idx = rng.integers(0, n, n)          # có hoàn lại','oob = np.setdiff1d(np.arange(n), idx)','tree = DecisionTreeClassifier(max_features=m).fit(X[idx], y[idx])','votes[oob] += tree.predict(X[oob])']} active={1}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 17: gradient boosting 1 chiều ─────────
const HX=housing.rows.map(r=>Math.log10(r[3])),HY=housing.rows.map(r=>r[7]);
const HS=split70(HX.length,17);
function stump(idx:number[],r:number[],depth:number):(x:number)=>number{
  // Cây hồi quy 1 chiều theo bình phương sai số (đệ quy đến độ sâu depth), ngưỡng là trung điểm.
  const o=[...idx].sort((a,b)=>HX[a]-HX[b]);const n=o.length,tot=o.reduce((s,i)=>s+r[i],0),m=tot/n;
  if(depth===0||n<2)return ()=>m;
  let best=-1,bt=0,lsum=0;
  for(let i=1;i<n;i++){lsum+=r[o[i-1]];const a=HX[o[i-1]],b=HX[o[i]];if(b<=a)continue;const g=lsum*lsum/i+(tot-lsum)**2/(n-i);if(g>best+1e-12){best=g;bt=(a+b)/2}}
  if(best<0)return ()=>m;
  const L=stump(o.filter(i=>HX[i]<=bt),r,depth-1),R=stump(o.filter(i=>HX[i]>bt),r,depth-1);
  return x=>x<=bt?L(x):R(x);
}
function BoostLab(){
  const [lr,setLR]=useState(.3),[depth,setDepth]=useState(1);const M=80;
  const run=useMemo(()=>{const f0=mean(HS.tr.map(i=>HY[i]));const F=HY.map(()=>f0);const stages:{h:(x:number)=>number;tr:number;va:number;res:number[]}[]=[];
    const mse=(idx:number[])=>mean(idx.map(i=>(HY[i]-F[i])**2));
    for(let m=0;m<M;m++){const res=HY.map((y,i)=>y-F[i]);const h=stump(HS.tr,res,depth);HY.forEach((_,i)=>F[i]+=lr*h(HX[i]));stages.push({h,tr:mse(HS.tr),va:mse(HS.va),res})}
    return {f0,stages,base:{tr:mean(HS.tr.map(i=>(HY[i]-f0)**2)),va:mean(HS.va.map(i=>(HY[i]-f0)**2))}}},[lr,depth]);
  const player=usePlayback(M,`${lr}-${depth}`);const m=player.index;
  const xs=Array.from({length:160},(_,i)=>.6+i*3.2/159);
  const Fx=(x:number,k:number)=>run.f0+lr*run.stages.slice(0,k).reduce((s,st)=>s+st.h(x),0);
  const best=run.stages.reduce((b,s,i)=>s.va<run.stages[b].va?i:b,0);
  return <>
    <p className="small-note">Dữ liệu: Real estate valuation (UCI), 414 giao dịch ở Tân Bắc, Đài Loan. x = log₁₀(khoảng cách tới ga MRT, m), y = giá mỗi ping (10.000 TWD). Chia 70/30 trong trình duyệt (seed 17).</p>
    <div className="py30-grid"><Range label="Learning rate η" value={lr} min={.05} max={1} step={.05} change={setLR}/><Range label="Độ sâu mỗi cây" value={depth} min={1} max={3} change={setDepth}/></div>
    <Transport player={player} count={M}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Sau ${m+1} cây · MSE train ${fmt(run.stages[m].tr,2)} · validation ${fmt(run.stages[m].va,2)}`}>
        <Chart xTitle="log₁₀ khoảng cách MRT" yTitle="Giá / ping" traces={[scatter('Train',HS.tr.map(i=>[HX[i],HY[i]]),CY,{marker:{color:CY,size:5,opacity:.6}}),scatter('Validation',HS.va.map(i=>[HX[i],HY[i]]),PK,{marker:{color:PK,size:5,opacity:.6}}),{type:'scatter',mode:'lines',name:'F trước cây này',x:xs,y:xs.map(x=>Fx(x,m)),line:{color:'#8aa0b8',dash:'dot'}},{type:'scatter',mode:'lines',name:'F sau cây này',x:xs,y:xs.map(x=>Fx(x,m+1)),line:{color:GD,width:3,shape:'hv'}}]}/>
      </SimulationPanel>
      <SimulationPanel title={`Cây thứ ${m+1} khớp phần dư y − F`}>
        <Chart xTitle="log₁₀ khoảng cách MRT" yTitle="Phần dư" height={260} traces={[scatter('Phần dư (train)',HS.tr.map(i=>[HX[i],run.stages[m].res[i]]),VI,{marker:{color:VI,size:5,opacity:.7}}),{type:'scatter',mode:'lines',name:'Cây h(x)',x:xs,y:xs.map(x=>run.stages[m].h(x)),line:{color:GR,width:3,shape:'hv'}}]}/>
        <SimulationCode file="boosting.py" lines={['r = y - F                        # gradient âm của ½(y − F)²','h = DecisionTreeRegressor(max_depth).fit(X, r)','F = F + lr * h.predict(X)','mse_val.append(np.mean((y_val - F_val) ** 2))']} active={(m%4)}/>
      </SimulationPanel>
      <SimulationPanel title={`MSE theo số cây · validation thấp nhất ở cây ${best+1} (${fmt(run.stages[best].va,2)})`}>
        <Chart xTitle="Số cây" yTitle="MSE" height={260} traces={[{type:'scatter',mode:'lines',name:'Train',x:run.stages.map((_,i)=>i+1),y:run.stages.map(s=>s.tr),line:{color:CY}},{type:'scatter',mode:'lines',name:'Validation',x:run.stages.map((_,i)=>i+1),y:run.stages.map(s=>s.va),line:{color:PK}},{type:'scatter',mode:'markers',name:'Dừng sớm',x:[best+1],y:[run.stages[best].va],marker:{color:GD,size:12}}]}/>
        <p className="small-note">Chỉ dự đoán trung bình: MSE train {fmt(run.base.tr,2)}, validation {fmt(run.base.va,2)}. Train luôn giảm; validation giảm rồi chững hoặc tăng.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 18: K-means trên Iris ─────────
const IP=iris.points as [number,number,number][];const IX=IP.map(p=>[p[0],p[1]]);
function kmeansRun(k:number,init:'first'|'random'|'pp',seed:number){
  const r=rng(seed);let C:number[][];
  if(init==='first')C=IX.slice(0,k).map(p=>[...p]);
  else if(init==='random')C=shuffle(IX.map((_,i)=>i),r).slice(0,k).map(i=>[...IX[i]]);
  else{C=[[...IX[Math.floor(r()*IX.length)]]];while(C.length<k){const d=IX.map(p=>Math.min(...C.map(c=>(p[0]-c[0])**2+(p[1]-c[1])**2)));const tot=d.reduce((s,v)=>s+v,0);let u=r()*tot,j=0;while(u>d[j]&&j<d.length-1){u-=d[j];j++}C.push([...IX[j]])}}
  const frames:{C:number[][];lab:number[];inertia:number;phase:'gán'|'cập nhật'}[]=[];
  for(let it=0;it<30;it++){const lab=IX.map(p=>{let b=0,bd=Infinity;C.forEach((c,j)=>{const d=(p[0]-c[0])**2+(p[1]-c[1])**2;if(d<bd){bd=d;b=j}});return b});
    const inertia=IX.reduce((s,p,i)=>s+(p[0]-C[lab[i]][0])**2+(p[1]-C[lab[i]][1])**2,0);frames.push({C:C.map(c=>[...c]),lab,inertia,phase:'gán'});
    const N=C.map((c,j)=>{const pts=IX.filter((_,i)=>lab[i]===j);return pts.length?[mean(pts.map(p=>p[0])),mean(pts.map(p=>p[1]))]:c});
    const moved=N.some((c,j)=>Math.abs(c[0]-C[j][0])+Math.abs(c[1]-C[j][1])>1e-9);C=N;
    frames.push({C:C.map(c=>[...c]),lab,inertia:IX.reduce((s,p,i)=>s+(p[0]-C[lab[i]][0])**2+(p[1]-C[lab[i]][1])**2,0),phase:'cập nhật'});if(!moved)break}
  return frames;
}
function KMeansLab(){
  const [k,setK]=useState(3),[init,setInit]=useState<'first'|'random'|'pp'>('first'),[seed,setSeed]=useState(1);
  const frames=useMemo(()=>kmeansRun(k,init,seed),[k,init,seed]);
  const elbow=useMemo(()=>Array.from({length:7},(_,i)=>{const kk=i+1;return {k:kk,inertia:Math.min(...[1,2,3,4,5].map(s=>{const f=kmeansRun(kk,'pp',s);return f[f.length-1].inertia}))}}),[]);
  const player=usePlayback(frames.length,frames);const fr=frames[player.index];
  const table=Array.from({length:k},(_,j)=>[`Cụm ${j+1}`,...[0,1,2].map(s=>IP.filter((p,i)=>fr.lab[i]===j&&p[2]===s).length)]);
  return <>
    <p className="small-note">Dữ liệu: {iris.source}; hai cột petal length và petal width (cm). Thuật toán không dùng nhãn loài; bảng bên phải chỉ dùng nhãn để đối chiếu.</p>
    <div className="py30-grid"><Range label="Số cụm k" value={k} min={2} max={6} change={setK}/>
      <label>Khởi tạo<select value={init} onChange={e=>setInit(e.target.value as 'first'|'random'|'pp')}><option value="first">k điểm đầu tiên (đều là setosa)</option><option value="random">k điểm ngẫu nhiên</option><option value="pp">k-means++</option></select></label>
      <button className="btn-outline" disabled={init==='first'} onClick={()=>setSeed(s=>s+1)}>Đổi seed ({seed})</button></div>
    <Transport player={player} count={frames.length}/>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Bước ${player.index+1}: ${fr.phase} · inertia ${fmt(fr.inertia,2)}`}>
        <Chart xTitle="petal length (cm)" yTitle="petal width (cm)" traces={[...Array.from({length:k},(_,j)=>scatter(`Cụm ${j+1}`,IX.filter((_,i)=>fr.lab[i]===j),PAL[j])),{type:'scatter',mode:'markers',name:'Tâm',x:fr.C.map(c=>c[0]),y:fr.C.map(c=>c[1]),marker:{symbol:'x',size:16,color:'#fff',line:{width:2,color:'#000'}}},
          ...Array.from({length:k},(_,j)=>({type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip',x:frames.slice(0,player.index+1).map(f=>f.C[j][0]),y:frames.slice(0,player.index+1).map(f=>f.C[j][1]),line:{color:PAL[j],dash:'dot',width:1.5}}))]}/>
      </SimulationPanel>
      <SimulationPanel title="Inertia qua các bước và đường khuỷu tay">
        <Chart xTitle="Bước" yTitle="Inertia" height={220} traces={[{type:'scatter',mode:'lines+markers',name:'Inertia',x:frames.map((_,i)=>i+1),y:frames.map(f=>f.inertia),line:{color:PK}},{type:'scatter',mode:'markers',name:'Hiện tại',x:[player.index+1],y:[fr.inertia],marker:{color:GD,size:11}}]}/>
        <Chart xTitle="k" yTitle="Inertia tốt nhất (5 lần k-means++)" height={220} traces={[{type:'scatter',mode:'lines+markers',name:'Khuỷu tay',x:elbow.map(e=>e.k),y:elbow.map(e=>e.inertia),line:{color:CY}}]}/>
      </SimulationPanel>
      <SimulationPanel title="Đối chiếu cụm với loài (chỉ để kiểm tra)">
        <Table headers={['',...iris.species]} rows={table}/>
        <SimulationCode file="kmeans.py" lines={['labels = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)','C[k] = X[labels == k].mean(0)   # cụm rỗng giữ tâm cũ','inertia = ((X - C[labels]) ** 2).sum()']} active={fr.phase==='gán'?0:1}/>
        <p className="small-note">Với "k điểm đầu tiên", mọi tâm bắt đầu trong cụm setosa. Hai bông đầu có cùng số đo cánh (1,4; 0,2) nên hai tâm trùng nhau và một cụm không bao giờ nhận điểm (cụm rỗng giữ tâm cũ): kết quả kẹt ở cực tiểu địa phương. Thử k-means++ với vài seed để so inertia cuối.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 19: PCA ─────────
const DG=digits as unknown as {source:string;mean:number[];components:number[][];ratio:number[];samples:{index:number[];label:number[];pixels:number[][];z:number[][]};cv:number[][]};
function DigitImg({pix,size=72,label}:{pix:number[];size?:number;label?:string}){
  const c=size/8;
  return <figure className="py30-digit" style={{margin:0,display:'inline-block',textAlign:'center'}}><svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={label??'Ảnh chữ số 8×8'}>
    {pix.map((v,i)=>{const g=Math.round(255*Math.max(0,Math.min(16,v))/16);return <rect key={i} x={(i%8)*c} y={Math.floor(i/8)*c} width={c} height={c} fill={`rgb(${g},${g},${g})`}/>})}</svg>{label&&<figcaption className="small-note">{label}</figcaption>}</figure>;
}
const reconstruct=(z:number[],k:number)=>DG.mean.map((m,p)=>m+z.slice(0,k).reduce((s,v,j)=>s+v*DG.components[j][p],0));
function PcaLab(){
  const [angle,setAngle]=useState(20),[sample,setSample]=useState(0),[k,setK]=useState(5);
  const pts=BS.tr.map(i=>BX[i]);
  const mx=mean(pts.map(p=>p[0])),my=mean(pts.map(p=>p[1]));
  const S=[[0,0],[0,0]];pts.forEach(p=>{const d=[p[0]-mx,p[1]-my];for(let a=0;a<2;a++)for(let b=0;b<2;b++)S[a][b]+=d[a]*d[b]/(pts.length-1)});
  const varAt=(deg:number)=>{const t=deg*Math.PI/180,u=[Math.cos(t),Math.sin(t)];return u[0]*u[0]*S[0][0]+2*u[0]*u[1]*S[0][1]+u[1]*u[1]*S[1][1]};
  const tr=S[0][0]+S[1][1],det=S[0][0]*S[1][1]-S[0][1]**2,l1=tr/2+Math.sqrt(tr*tr/4-det);
  const pcDeg=((Math.atan2(l1-S[0][0],S[0][1])*180/Math.PI)+180)%180;
  const t=angle*Math.PI/180,u=[Math.cos(t),Math.sin(t)];
  const proj=pts.map(p=>{const s=(p[0]-mx)*u[0]+(p[1]-my)*u[1];return [mx+s*u[0],my+s*u[1]]});
  const cum=DG.ratio.reduce<number[]>((a,v)=>[...a,(a[a.length-1]??0)+v],[]);
  const z=DG.samples.z[sample],orig=DG.samples.pixels[sample],rec=reconstruct(z,k);
  const err=mean(orig.map((v,i)=>(v-rec[i])**2));
  return <>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Chọn hướng chiếu · phương sai ${fmt(varAt(angle))} (lớn nhất ${fmt(l1)} tại ${fmt(pcDeg,1)}°)`}>
        {breastNote}<Range label="Góc của u (độ)" value={angle} min={0} max={179} change={setAngle}/>
        <Chart height={300} traces={[scatter('Điểm train',pts,'#7d93ad',{marker:{color:'#7d93ad',size:5,opacity:.5}}),{type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip',x:pts.flatMap((p,i)=>[p[0],proj[i][0],null]),y:pts.flatMap((p,i)=>[p[1],proj[i][1],null]),line:{color:'rgba(242,141,185,.25)',width:1}},scatter('Hình chiếu lên u',proj,PK,{marker:{color:PK,size:4}}),{type:'scatter',mode:'lines',name:'Hướng u',x:[mx-4*u[0],mx+4*u[0]],y:[my-4*u[1],my+4*u[1]],line:{color:GD,width:3}}]} layout={{...gridAxes,yaxis:{...gridAxes.yaxis,scaleanchor:'x'}}}/>
        <Chart xTitle="Góc (độ)" yTitle="Var(Xu)" height={200} traces={[{type:'scatter',mode:'lines',name:'Phương sai theo góc',x:Array.from({length:180},(_,i)=>i),y:Array.from({length:180},(_,i)=>varAt(i)),line:{color:CY}},{type:'scatter',mode:'markers',name:'Góc đang chọn',x:[angle],y:[varAt(angle)],marker:{color:GD,size:11}}]}/>
      </SimulationPanel>
      <SimulationPanel title={`Dựng lại ảnh chữ số với ${k} thành phần · giữ ${fmt(cum[k-1]*100,1)}% phương sai`}>
        <p className="small-note">Dữ liệu: {DG.source}. PCA tính trên cả 1797 ảnh 8×8 (64 chiều).</p>
        <div className="py30-grid"><Range label="Ảnh mẫu" value={sample} min={0} max={DG.samples.label.length-1} change={setSample}/><Range label="Số thành phần k" value={k} min={1} max={30} change={setK}/></div>
        <div style={{display:'flex',gap:16,flexWrap:'wrap',alignItems:'flex-end'}}><DigitImg pix={orig} size={96} label={`Gốc (nhãn ${DG.samples.label[sample]})`}/><DigitImg pix={rec} size={96} label={`Dựng lại · MSE/pixel ${fmt(err,2)}`}/><DigitImg pix={DG.mean} size={96} label="Ảnh trung bình"/></div>
        <p className="small-note">Bốn thành phần đầu (đã kéo giãn thang xám):</p>
        <div style={{display:'flex',gap:10,flexWrap:'wrap'}}>{DG.components.slice(0,4).map((c,j)=>{const lo=Math.min(...c),hi=Math.max(...c);return <DigitImg key={j} pix={c.map(v=>16*(v-lo)/(hi-lo))} size={64} label={`PC${j+1} · ${fmt(DG.ratio[j]*100,1)}%`}/>})}</div>
        <Chart xTitle="Số thành phần" yTitle="Phương sai tích lũy" height={200} traces={[{type:'scatter',mode:'lines',name:'Tích lũy',x:cum.map((_,i)=>i+1),y:cum,line:{color:VI}},{type:'scatter',mode:'markers',name:'k đang chọn',x:[k],y:[cum[k-1]],marker:{color:GD,size:11}},{type:'scatter',mode:'lines',name:'95%',x:[1,64],y:[.95,.95],line:{color:'#41536a',dash:'dot'}}]}/>
        <SimulationCode file="pca.py" lines={['_, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)','Z = (X - mean) @ Vt[:k].T','X_hat = Z @ Vt[:k] + mean']} active={2}/>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 20: target encoding và rò rỉ (giả lập) ─────────
function auc(y:number[],s:number[]){const o=s.map((v,i)=>[v,y[i]]).sort((a,b)=>a[0]-b[0]);let rank=0,sumPos=0,i=0;const P=y.filter(v=>v===1).length,N=y.length-P;
  while(i<o.length){let j=i;while(j<o.length&&o[j][0]===o[i][0])j++;const avg=(rank+1+rank+(j-i))/2;for(let t=i;t<j;t++)if(o[t][1]===1)sumPos+=avg;rank+=j-i;i=j}
  return (sumPos-P*(P+1)/2)/(P*N)}
function encode(cat:number[],y:number[],tr:number[],apply:number[],m:number){const sum=new Map<number,number>(),cnt=new Map<number,number>();
  tr.forEach(i=>{sum.set(cat[i],(sum.get(cat[i])??0)+y[i]);cnt.set(cat[i],(cnt.get(cat[i])??0)+1)});const prior=mean(tr.map(i=>y[i]));
  return apply.map(i=>cnt.has(cat[i])?(sum.get(cat[i])!+m*prior)/(cnt.get(cat[i])!+m):prior)}
function TargetLab(){
  const [levels,setLevels]=useState(300),[m,setM]=useState(0),[signal,setSignal]=useState(false),[seed,setSeed]=useState(1);
  const res=useMemo(()=>{const r=rng(seed*97+levels);const n=1000;const eff=Array.from({length:levels},()=>r()*3-1.5);
    const cat=Array.from({length:n},()=>Math.floor(r()*levels));const y=cat.map(c=>Number(r()<(signal?1/(1+Math.exp(-eff[c])):.5)));
    const all=Array.from({length:n},(_,i)=>i);const naive=encode(cat,y,all,all,m);
    const order=shuffle(all,rng(seed));const oof=new Array(n).fill(0);
    for(let f=0;f<5;f++){const va=order.filter((_,i)=>i%5===f),vaSet=new Set(va);const tr=all.filter(i=>!vaSet.has(i));encode(cat,y,tr,va,m).forEach((v,j)=>oof[va[j]]=v)}
    const counts=new Map<number,number>();cat.forEach(c=>counts.set(c,(counts.get(c)??0)+1));
    return {y,naive,oof,aNaive:auc(y,naive),aOof:auc(y,oof),single:[...counts.values()].filter(v=>v===1).length}},[levels,m,signal,seed]);
  const hist=(vals:number[],name:string,color:string,cls:number):PlotTrace=>({type:'histogram',name,x:vals.filter((_,i)=>res.y[i]===cls),xbins:{start:0,end:1.0001,size:.05},marker:{color},opacity:.7});
  return <>
    <p className="small-note">Thí nghiệm giả lập: 1000 dòng, một cột phân loại có số giá trị bạn chọn. Không tín hiệu: nhãn là tung đồng xu, không liên quan tới cột. Có tín hiệu: mỗi giá trị có một xác suất riêng.</p>
    <div className="py30-grid"><Range label="Số giá trị khác nhau" value={levels} min={5} max={600} step={5} change={setLevels}/><Range label="Làm mượt m" value={m} min={0} max={50} change={setM}/>
      <label><input type="checkbox" checked={signal} onChange={e=>setSignal(e.target.checked)}/> Nhãn có phụ thuộc vào cột</label><button className="btn-outline" onClick={()=>setSeed(s=>s+1)}>Chạy lại</button></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Trong-mẫu (sai) · AUC ${fmt(res.aNaive)}`}>
        <Chart xTitle="Giá trị mã hóa" yTitle="Số dòng" height={260} traces={[hist(res.naive,'Nhãn 0',CY,0),hist(res.naive,'Nhãn 1',PK,1)]} layout={{barmode:'overlay'}}/>
        <p className="small-note">{res.single} giá trị chỉ xuất hiện đúng một lần: mã của chúng chính là nhãn của dòng đó (khi m = 0), nên hai lớp tách hẳn ra ở 0 và 1.</p>
      </SimulationPanel>
      <SimulationPanel title={`Out-of-fold 5 fold (đúng) · AUC ${fmt(res.aOof)}`}>
        <Chart xTitle="Giá trị mã hóa" yTitle="Số dòng" height={260} traces={[hist(res.oof,'Nhãn 0',CY,0),hist(res.oof,'Nhãn 1',PK,1)]} layout={{barmode:'overlay'}}/>
        <SimulationCode file="target_encoding.py" lines={['for tr, va in kfold_indices(n, 5, seed):','    stats = y[tr].groupby(cat[tr]).agg(["sum", "count"])','    enc = (stats.sum + m * prior) / (stats.count + m)','    out[va] = cat[va].map(enc).fillna(prior)']} active={3}/>
        <p className="small-note">Không tín hiệu: AUC out-of-fold phải quanh 0,5, còn bản trong-mẫu cao giả tạo. Tăng m làm bản trong-mẫu bớt lạc quan nhưng không sửa được gốc vấn đề.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 21: Digit Recognizer (PCA + KNN) ─────────
function DigitsLab(){
  const [nc,setNc]=useState(10),[k,setK]=useState(3),[sample,setSample]=useState(0);
  const Z=DG.samples.z,lab=DG.samples.label;
  const dist=(a:number,b:number)=>{let s=0;for(let j=0;j<nc;j++)s+=(Z[a][j]-Z[b][j])**2;return s};
  const neighbors=(i:number)=>Z.map((_,j)=>j).filter(j=>j!==i).sort((a,b)=>dist(i,a)-dist(i,b)).slice(0,k);
  const vote=(nb:number[])=>{const c=new Array(10).fill(0);nb.forEach(j=>c[lab[j]]++);return c.indexOf(Math.max(...c))};
  const loo=useMemo(()=>mean(Z.map((_,i)=>Number(vote(neighbors(i))===lab[i]))),[nc,k]);// eslint-disable-line react-hooks/exhaustive-deps
  const nb=neighbors(sample),pred=vote(nb);
  const ncs=[...new Set(DG.cv.map(r=>r[0]))],ks=[...new Set(DG.cv.map(r=>r[1]))];
  return <>
    <p className="small-note">Dữ liệu: {DG.source}. Biểu đồ và láng giềng dùng 400 ảnh mẫu chiếu lên các thành phần chính của cả 1797 ảnh. Bảng accuracy là stratified 5-fold trên 1797 ảnh, do scikit-learn tính khi soạn bài.</p>
    <div className="py30-grid"><Range label="Số thành phần dùng cho khoảng cách" value={nc} min={2} max={30} change={setNc}/><Range label="k láng giềng" value={k} min={1} max={9} step={2} change={setK}/><Range label="Ảnh truy vấn" value={sample} min={0} max={Z.length-1} change={setSample}/></div>
    <div className="simulation-studio-grid">
      <SimulationPanel title="400 ảnh trên hai thành phần chính đầu tiên">
        <Chart xTitle="PC1" yTitle="PC2" traces={[...Array.from({length:10},(_,d)=>scatter(`Chữ số ${d}`,Z.map((z,i)=>lab[i]===d?[z[0],z[1]]:null).filter((p):p is number[]=>p!==null),PAL[d],{marker:{color:PAL[d],size:6,opacity:.8}})),{type:'scatter',mode:'markers',name:'Truy vấn',x:[Z[sample][0]],y:[Z[sample][1]],marker:{color:'#fff',size:15,symbol:'diamond',line:{color:'#000',width:1}}}]}/>
        <p className="small-note">Chỉ với 2 chiều, vài nhóm (như chữ số 0) đã tách khá rõ nhưng nhiều nhóm còn chồng lên nhau; bảng bên dưới cho thấy 2 thành phần chỉ đạt khoảng 0,6.</p>
      </SimulationPanel>
      <SimulationPanel title={`Dự đoán ${pred} cho ảnh nhãn ${lab[sample]} · leave-one-out trên 400 ảnh: ${fmt(loo)}`}>
        <div style={{display:'flex',gap:12,flexWrap:'wrap',alignItems:'flex-end'}}><DigitImg pix={DG.samples.pixels[sample]} size={88} label="Truy vấn"/>{nb.map(j=><DigitImg key={j} pix={DG.samples.pixels[j]} size={56} label={`nhãn ${lab[j]}`}/>)}</div>
        <SimulationCode file="digits.py" lines={['pipe = Pipeline([("pca", PCA(nc)), ("knn", KNeighborsClassifier(k))])','cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))','pipe.fit(X_train, y_train).predict(X_test)']} active={0}/>
      </SimulationPanel>
      <SimulationPanel title="Accuracy stratified 5-fold (scikit-learn, 1797 ảnh)">
        <Chart xTitle="k" yTitle="n_components" height={300} traces={[{type:'heatmap',x:ks.map(String),y:ncs.map(String),z:ncs.map(a=>ks.map(b=>DG.cv.find(r=>r[0]===a&&r[1]===b)![2])),colorscale:[[0,'#123a52'],[1,'#ffd173']],text:ncs.map(a=>ks.map(b=>fmt(DG.cv.find(r=>r[0]===a&&r[1]===b)![2],3))),texttemplate:'%{text}',hoverinfo:'skip',showscale:false}]} layout={{xaxis:{type:'category',title:{text:'k'}},yaxis:{type:'category',title:{text:'n_components'}}}}/>
        <p className="small-note">Cấu hình tốt nhất trong bảng: {(()=>{const b=DG.cv.reduce((a,r)=>r[2]>a[2]?r:a);return `${b[0]} thành phần, k = ${b[1]}: ${fmt(b[2],4)}`})()}. Đây là điểm CV; điểm Kaggle trên ảnh 28×28 phải tự chạy lại.</p>
      </SimulationPanel>
    </div></>;
}

export function WeekThreeLab({day}:{day:number}){
  if(day===15)return <TreeLab/>;
  if(day===16)return <ForestLab/>;
  if(day===17)return <BoostLab/>;
  if(day===18)return <KMeansLab/>;
  if(day===19)return <PcaLab/>;
  if(day===20)return <TargetLab/>;
  return <DigitsLab/>;
}
