// Mô phỏng tuần 5 (Ngày 29–30): dự án tổng kết. Dữ liệu thật: Real estate valuation (UCI, data/housing.json).
// Mô hình tuyến tính trong trang được khớp bằng bình phương tối thiểu ngay trong trình duyệt; các số của gradient boosting
// là kết quả scikit-learn khi soạn bài (xem scripts/gen_week5.py). Thí nghiệm "dùng lại holdout" là giả lập, được ghi rõ.
import {useMemo,useState} from 'react';
import {Chart} from '../components/Chart';
import {Range,Table} from '../components/ObservationControls';
import {SimulationCode,SimulationPanel} from '../components/SimulationStudio';
import {housing} from '../data';
import {leastSquares} from '../lib/domain';
import type {PlotTrace} from '../types';

const CY='#42d3e5',PK='#f28db9',GD='#ffd173',GR='#84e0a8',VI='#a8a7ff';
const fmt=(v:number,d=3)=>Number.isFinite(v)?v.toLocaleString('vi-VN',{maximumFractionDigits:d}):'—';
const mean=(a:number[])=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;
function rng(seed:number){let s=(seed*2654435761)>>>0||1;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
function shuffle<T>(a:T[],r:()=>number){const b=[...a];for(let i=b.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[b[i],b[j]]=[b[j],b[i]]}return b}

// housing.rows: [id, ngày giao dịch, tuổi nhà, khoảng cách MRT, số cửa hàng, vĩ độ, kinh độ, giá]
const ROWS=housing.rows,Y=ROWS.map(r=>r[7]);
const FEATS=[{key:'date',name:'thời gian giao dịch',f:(r:number[])=>r[1]},{key:'age',name:'tuổi nhà',f:(r:number[])=>r[2]},{key:'mrt',name:'log(1 + khoảng cách MRT)',f:(r:number[])=>Math.log1p(r[3])},{key:'stores',name:'số cửa hàng',f:(r:number[])=>r[4]},{key:'lat',name:'vĩ độ',f:(r:number[])=>r[5]},{key:'lon',name:'kinh độ',f:(r:number[])=>r[6]}];
const GROUPS:Record<string,string[]>={'vị trí':['lat','lon'],'tiện ích':['mrt','stores'],'tuổi nhà':['age'],'thời gian':['date']};
const source=<p className="small-note">Dữ liệu: Real estate valuation (Yeh, 2018; UCI, CC BY 4.0), 414 giao dịch ở Tân Bắc, Đài Loan; giá theo 10.000 TWD/ping.</p>;
// Phép chia dev/holdout giống bài Python (seed 0) không tái tạo được bằng bộ sinh số JS, nên trang tự chia 80/20 với seed riêng.
const SPLIT=(()=>{const o=shuffle(ROWS.map((_,i)=>i),rng(29));return {hold:o.slice(0,83).sort((a,b)=>a-b),dev:o.slice(83).sort((a,b)=>a-b)}})();
function cvLinear(idx:number[],keys:string[],seed=0){
  const r=rng(seed+1),order=shuffle(idx,r),folds=Array.from({length:5},(_,k)=>order.filter((_,i)=>i%5===k));const fs=FEATS.filter(f=>keys.includes(f.key));
  const pred=new Map<number,number>();const rmses:number[]=[];
  folds.forEach(va=>{const vs=new Set(va),tr=idx.filter(i=>!vs.has(i));
    const mu=fs.map(f=>mean(tr.map(i=>f.f(ROWS[i])))),sd=fs.map((f,k)=>Math.sqrt(mean(tr.map(i=>(f.f(ROWS[i])-mu[k])**2)))||1);
    const z=(row:number[])=>[1,...fs.map((f,k)=>(f.f(row)-mu[k])/sd[k])];const th=leastSquares(tr.map(i=>z(ROWS[i])),tr.map(i=>Y[i]));
    const p=va.map(i=>z(ROWS[i]).reduce((s,v,j)=>s+v*th[j],0));va.forEach((i,j)=>pred.set(i,p[j]));rmses.push(Math.sqrt(mean(va.map((i,j)=>(p[j]-Y[i])**2))))});
  return {rmse:mean(rmses),pred};
}

// ───────── Ngày 29: holdout và baseline ─────────
function HoldoutLab(){
  const [K,setK]=useState(50),[n,setN]=useState(200),[run,setRun]=useState(0);
  const exp=useMemo(()=>{const r=rng(1000+run);const best:number[]=[],fresh:number[]=[];
    const draw=()=>{let c=0;for(let i=0;i<n;i++)if(r()<.8)c++;return c/n};
    for(let rep=0;rep<300;rep++){let b=-1;for(let k=0;k<K;k++){const a=draw();if(a>b)b=a}best.push(b);fresh.push(draw())}
    return {best,fresh}},[K,n,run]);
  const ladder=[['Trung bình',13.82],['Tuyến tính 1 biến',9.22],['Tuyến tính 6 biến',9.04],['Gradient boosting',7.81]] as const;
  return <>
    <div className="simulation-studio-grid">
      <SimulationPanel title="Khóa holdout trước khi nhìn dữ liệu">
        {source}
        <svg viewBox="0 0 460 190" width="100%" role="img" aria-label="414 dòng dữ liệu, 83 dòng holdout tô màu hồng">
          {ROWS.map((_,i)=>{const h=SPLIT.hold.includes(i);return <rect key={i} x={(i%23)*20} y={Math.floor(i/23)*10.5} width={18} height={9} rx={2} fill={h?PK:'#2b405a'}/>})}
        </svg>
        <p className="small-note">Ô hồng: 83 dòng holdout (20%), khóa lại tới cuối Ngày 30. Ô xanh: 331 dòng dev để khám phá, chọn mô hình và cross-validation.</p>
        <SimulationCode file="holdout.py" lines={['dev_idx, holdout_idx = lock_holdout(len(df), 0.2, seed=0)','print(fingerprint(holdout_idx))   # ghi vào README','# ... mọi thử nghiệm chỉ dùng df.iloc[dev_idx] ...','score = rmsle(y[holdout_idx], model.predict(X[holdout_idx]))  # đúng MỘT lần']} active={1}/>
      </SimulationPanel>
      <SimulationPanel title={`Dùng lại holdout ${K} lần: điểm "tốt nhất" trung bình ${fmt(mean(exp.best))} (thật: 0,8)`}>
        <p className="small-note">Thí nghiệm giả lập: {K} mô hình có độ chính xác thật đều bằng 0,8. Mỗi lần thử, chấm trên cùng holdout {n} mẫu rồi giữ mô hình điểm cao nhất; lặp 300 lần.</p>
        <div className="py30-grid"><Range label="Số lần thử trên holdout" value={K} min={1} max={200} change={setK}/><Range label="Cỡ holdout" value={n} min={50} max={2000} step={50} change={setN}/><button className="btn-outline" onClick={()=>setRun(v=>v+1)}>Chạy lại</button></div>
        <Chart xTitle="Accuracy" yTitle="Số lần lặp" height={260} traces={[{type:'histogram',name:'Điểm holdout của mô hình được chọn',x:exp.best,marker:{color:PK},opacity:.75,xbins:{size:.005}},{type:'histogram',name:'Điểm của nó trên dữ liệu mới',x:exp.fresh,marker:{color:CY},opacity:.75,xbins:{size:.005}}]} layout={{barmode:'overlay'}}/>
        <p className="small-note">Càng thử nhiều lần trên cùng holdout, điểm được báo cáo càng cao hơn thực tế; trên dữ liệu mới, mô hình được chọn vẫn chỉ đạt khoảng 0,8. Holdout lớn hơn làm độ lệch nhỏ đi nhưng không triệt tiêu.</p>
      </SimulationPanel>
      <SimulationPanel title="Thang baseline trên 331 dòng dev (RMSE 5-fold)">
        <Chart yTitle="RMSE (càng thấp càng tốt)" height={280} traces={[{type:'bar',name:'RMSE 5-fold',x:ladder.map(l=>l[0]),y:ladder.map(l=>l[1]),marker:{color:[VI,CY,CY,GR]},text:ladder.map(l=>fmt(l[1],2)),textposition:'outside'}]} layout={{xaxis:{type:'category'},yaxis:{range:[0,16]}}}/>
        <p className="small-note">Tuyến tính 1 biến dùng log(1 + khoảng cách MRT); tuyến tính 6 biến dùng mọi cột đã chuẩn hóa; gradient boosting là GradientBoostingRegressor(random_state=0).</p>
        <p className="small-note">Số do scikit-learn tính khi soạn bài Ngày 29 (KFold 5, shuffle, seed 0). Mô hình cuối phải được so với cả bốn mốc này, không chỉ với mốc yếu nhất.</p>
      </SimulationPanel>
    </div></>;
}

// ───────── Ngày 30: ablation và phân tích lỗi ─────────
const GB_ABL=[['tuổi nhà',8.23,.42],['thời gian',8.22,.41],['tiện ích',8.21,.40],['vị trí',7.99,.18]] as const;
function AblationLab(){
  const [keys,setKeys]=useState<string[]>(FEATS.map(f=>f.key));
  const full=useMemo(()=>cvLinear(SPLIT.dev,keys),[keys]);
  const abl=useMemo(()=>Object.entries(GROUPS).map(([g,cols])=>{const rest=keys.filter(k=>!cols.includes(k));return {g,used:cols.some(c=>keys.includes(c)),rmse:rest.length?cvLinear(SPLIT.dev,rest).rmse:NaN}}).map(a=>({...a,delta:a.rmse-full.rmse})).sort((a,b)=>(b.delta||0)-(a.delta||0)),[keys,full]);
  const dev=SPLIT.dev,pred=dev.map(i=>full.pred.get(i) as number),err=dev.map((i,j)=>pred[j]-Y[i]);
  const qs=[...dev.map(i=>ROWS[i][3])].sort((a,b)=>a-b),cut=[.25,.5,.75].map(q=>qs[Math.floor(q*(qs.length-1))]);
  const bin=(v:number)=>v<=cut[0]?0:v<=cut[1]?1:v<=cut[2]?2:3;
  const slices=[0,1,2,3].map(b=>{const ids=dev.map((i,j)=>j).filter(j=>bin(ROWS[dev[j]][3])===b);return {b,n:ids.length,mae:mean(ids.map(j=>Math.abs(err[j]))),bias:mean(ids.map(j=>err[j]))}});
  const worst=dev.map((_,j)=>j).sort((a,b)=>Math.abs(err[b])-Math.abs(err[a])).slice(0,5);
  const labels=['≤ '+fmt(cut[0],0)+' m',fmt(cut[0],0)+'–'+fmt(cut[1],0)+' m',fmt(cut[1],0)+'–'+fmt(cut[2],0)+' m','> '+fmt(cut[2],0)+' m'];
  const toggle=(k:string)=>setKeys(old=>old.includes(k)?(old.length>1?old.filter(x=>x!==k):old):[...old,k]);
  return <>
    {source}
    <div className="py30-grid">{FEATS.map(f=><label key={f.key}><input type="checkbox" checked={keys.includes(f.key)} onChange={()=>toggle(f.key)}/> {f.name}</label>)}</div>
    <div className="simulation-studio-grid">
      <SimulationPanel title={`Ablation · hồi quy tuyến tính trên ${keys.length} cột: RMSE 5-fold ${fmt(full.rmse,2)}`}>
        <Chart xTitle="Δ RMSE khi bỏ nhóm (dương = nhóm có ích)" height={240} traces={[{type:'bar',orientation:'h',name:'Tuyến tính (tính trong trang)',y:abl.map(a=>a.g),x:abl.map(a=>a.used?a.delta:0),marker:{color:abl.map(a=>a.delta>0?GD:'#5a6b80')},text:abl.map(a=>a.used?fmt(a.delta,2):'đã bỏ'),textposition:'auto'}]} layout={{yaxis:{type:'category',autorange:'reversed'}}}/>
        <Table headers={['Nhóm','Gradient boosting: RMSE khi bỏ','Δ']} rows={GB_ABL.map(r=>[r[0],fmt(r[1],2),'+'+fmt(r[2],2)])}/>
        <p className="small-note">Bảng dưới là kết quả scikit-learn (GradientBoostingRegressor, dev Ngày 30, RMSE đầy đủ 7,81). Bỏ "vị trí" chỉ tăng 0,18 vì thông tin vị trí phần lớn đã có trong khoảng cách tới ga: ablation đo đóng góp riêng khi các nhóm khác vẫn còn.</p>
      </SimulationPanel>
      <SimulationPanel title="Phân tích lỗi theo tứ phân vị khoảng cách MRT">
        <Chart xTitle="Giá thật" yTitle="Dự đoán (out-of-fold)" height={280} traces={[{type:'scatter',mode:'markers',name:'Dev',x:dev.map(i=>Y[i]),y:pred,marker:{color:CY,size:5,opacity:.6}},{type:'scatter',mode:'markers',name:'5 ca sai nhiều nhất',x:worst.map(j=>Y[dev[j]]),y:worst.map(j=>pred[j]),marker:{color:PK,size:11,symbol:'x'}},{type:'scatter',mode:'lines',name:'Dự đoán = thật',x:[0,120],y:[0,120],line:{color:'#41536a',dash:'dot'}}] as PlotTrace[]}/>
        <Table headers={['Khoảng cách MRT','n','MAE','Bias (dự đoán − thật)']} rows={slices.sort((a,b)=>b.mae-a.mae).map(s=>[labels[s.b],s.n,fmt(s.mae,2),fmt(s.bias,2)])}/>
        <SimulationCode file="errors.py" lines={['table = error_slices(X_dev, y_dev, pred, by="mrt_distance_m", q=4)','worst = worst_cases(y_dev, pred, k=10)','problems = check_submission(sub, test.Id)']} active={0}/>
      </SimulationPanel>
    </div></>;
}

export function WeekFiveLab({day}:{day:number}){
  return day===29?<HoldoutLab/>:<AblationLab/>;
}
