import {SimulationPanel,SimulationCode} from '../components/SimulationStudio';
import {useMemo,useState} from 'react';
import {Chart} from '../components/Chart';
import {Range,Table,Transport,usePlayback} from '../components/ObservationControls';
import {housing} from '../data';
import {solve} from '../lib/domain';
import type {PlotTrace} from '../types';

const rows=housing.rows;
const fmt=(n:number)=>Number.isFinite(n)?n.toLocaleString('vi-VN',{maximumFractionDigits:3}):'Vượt miền số';
const mean=(xs:number[])=>xs.reduce((a,b)=>a+b,0)/xs.length;

function shape(text:string) {
  const parts=text.trim().replace(/^\(/,'').replace(/\)$/,'').split(',').map(x=>x.trim());
  return parts.length<=4&&parts.every(x=>/^[1-9]\d?$/.test(x))?parts.map(Number):null;
}
function Broadcasting() {
  const [a,setA]=useState('3, 1'),[b,setB]=useState('1, 2');
  const A=shape(a),B=shape(b),n=Math.max(A?.length??0,B?.length??0);
  const dims=A&&B?Array.from({length:n},(_,i)=>{
    const x=A.at(-n+i)??1,y=B.at(-n+i)??1;
    return [n-i,x,y,x===y||x===1||y===1?Math.max(x,y):-1];
  }):[];
  const result=dims.length&&dims.every(row=>row[3]>0)?dims.map(row=>row[3]):null;
  const ar=A?.at(-2)??1,ac=A?.at(-1)??1,br=B?.at(-2)??1,bc=B?.at(-1)??1;
  return <>
    <div className="py30-actions"><label>Shape A <input value={a} onChange={e=>setA(e.target.value)}/></label><label>Shape B <input value={b} onChange={e=>setB(e.target.value)}/></label></div>
    {dims.length>0&&<Table headers={['Chiều từ phải','A','B','Kết quả']} rows={dims.map(r=>[r[0],r[1],r[2],r[3]===-1?'Không khớp':r[3]])}/>}
    <p role="status">{result?`Shape kết quả: (${result.join(', ')}).`:dims.length?'Không thể broadcast các chiều trên.':'Nhập từ 1 đến 4 kích thước dương, phân cách bằng dấu phẩy.'}</p>
    {result&&result.length<=2&&Math.max(ar,ac,br,bc)<=8&&<Table headers={['Hàng',...Array.from({length:Math.max(ac,bc)},(_,j)=>`Cột ${j+1}`)]} rows={Array.from({length:Math.max(ar,br)},(_,i)=>[i+1,...Array.from({length:Math.max(ac,bc)},(_,j)=>`A[${ar===1?0:i},${ac===1?0:j}] + B[${br===1?0:i},${bc===1?0:j}]`)])}/>}
  </>;
}
function rng(seed:number) {
  let s=seed>>>0;
  return ()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};
}
function Leakage() {
  const [k,setK]=useState(20),[count,setCount]=useState(24);
  const [selected,fresh]=useMemo(()=>{
    const random=rng(42),a:number[]=[],b:number[]=[];
    const labels=()=>Array.from({length:count},()=>random()<.5);
    const accuracy=(truth:boolean[],predict:()=>number)=>mean(truth.map(label=>Number(label===(predict()<.5))));
    for(let i=0;i<150;i++){
      const selection=labels(), independent=labels();
      const models=Array.from({length:k},()=>rng(Math.floor(random()*4294967296)));
      const scores=models.map(model=>accuracy(selection,model));
      const best=scores.indexOf(Math.max(...scores));
      a.push(scores[best]);
      b.push(accuracy(independent,models[best]));
    }
    return [a,b];
  },[k,count]);
  return <><Range label="Số mô hình đoán ngẫu nhiên" value={k} min={1} max={100} change={setK}/><Range label="Số nhãn mỗi tập" value={count} min={12} max={80} change={setCount}/>
    <p>150 lần thử với nhãn giả lập. Chọn mô hình trên tập test cho accuracy trung bình {fmt(mean(selected))}. Đo lại mô hình đã chọn trên dữ liệu độc lập cho {fmt(mean(fresh))}.</p>
    <Chart xTitle="Accuracy" yTitle="Số lần" traces={[{type:'histogram',x:selected,name:'Tập dùng để chọn',opacity:.65},{type:'histogram',x:fresh,name:'Tập mới',opacity:.65}]} layout={{barmode:'overlay',xaxis:{title:{text:'Accuracy'},range:[0,1]}}}/>
  </>;
}

const y=rows.map(r=>r[7]);
const raw=rows.map(r=>Math.log1p(r[3]));
const xMean=mean(raw);
const sd=Math.sqrt(mean(raw.map(x=>(x-xMean)**2)));
const standardized=raw.map(x=>(x-xMean)/sd);
const points=(x:number[],y:number[],name:string):PlotTrace=>({type:'scatter',mode:'markers',x,y,name});
const line=(x:number[],y:number[],name:string):PlotTrace=>({type:'scatter',mode:'lines',x,y,name});
const mse=(x:number[],w:number,b:number)=>mean(y.map((v,i)=>(w*x[i]+b-v)**2));

function Regression({day}:{day:number}) {
  const [w,setW]=useState(0),[b,setB]=useState(15),[lr,setLR]=useState(day===4?.01:.08),[batch,setBatch]=useState(16),[scale,setScale]=useState(true);
  const x=day===5&&scale?standardized:raw;
  const mu=mean(x),ym=mean(y);
  const optimal=mean(x.map((v,i)=>(v-mu)*(y[i]-ym)))/mean(x.map(v=>(v-mu)**2));
  const offset=ym-optimal*mu;
  const history=useMemo(()=>{
    const entries=[{w,b,loss:mse(x,w,b),samples:0,sample:-1,batchSize:0,residual:0,dw:0,db:0}];
    const random=rng(2026);
    let order:number[]=[];
    const size=day===4?x.length:batch;
    for(let step=0;step<55;step++){
      if(!order.length){
        order=x.map((_,i)=>i);
        for(let i=order.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
      }
      const subset=order.splice(0,size),previous=entries.at(-1)!;
      const residual=subset.map(i=>previous.w*x[i]+previous.b-y[i]);
      const dw=2*mean(subset.map((id,i)=>x[id]*residual[i])),db=2*mean(residual);
      const nextW=previous.w-lr*dw,nextB=previous.b-lr*db;
      const loss=mse(x,nextW,nextB);
      if(!Number.isFinite(loss)||loss>1e15)break;
      entries.push({w:nextW,b:nextB,loss,samples:previous.samples+subset.length,sample:subset[0],batchSize:subset.length,residual:residual[0],dw,db});
    }
    return entries;
  },[x,w,b,lr,batch,day]);
  const count=(history.length-1)*4+1;
  const player=usePlayback(count,history);
  const completed=Math.floor(player.index/4);
  const phase=player.index===0?-1:(player.index-1)%4;
  const frame=history[completed];
  const operation=history[Math.ceil(player.index/4)];
  const grid=useMemo(()=>{
    const minW=Math.min(w-3,optimal-22),maxW=Math.max(w+3,optimal+22);
    const minB=Math.min(b-6,offset-65),maxB=Math.max(b+6,offset+65);
    const slopes=Array.from({length:33},(_,i)=>minW+i*(maxW-minW)/32);
    const offsets=Array.from({length:33},(_,i)=>minB+i*(maxB-minB)/32);
    return {x:slopes,y:offsets,z:offsets.map(j=>slopes.map(i=>Math.log10(Math.max(mse(x,i,j),1e-9))))};
  },[x,w,b,optimal,offset]);
  const chosenX=[Math.min(...x),Math.max(...x)];
  const visiblePath=history.slice(0,completed+1);
  const traces:PlotTrace[]=[{type:day===3?'heatmap':'contour',...grid,name:'log₁₀(MSE)',showscale:false,colorscale:[[0,'#20718a'],[.35,'#173c5b'],[.72,'#292746'],[1,'#432645']],hovertemplate:'w = %{x:.2f}<br>b = %{y:.2f}<br>log₁₀(MSE) = %{z:.3f}<extra></extra>'},
    {...points([optimal],[offset],'Nghiệm đóng'),marker:{color:'#42d3e5',size:10,symbol:'x'}},
    {type:'scatter',mode:'lines',x:visiblePath.map(v=>v.w),y:visiblePath.map(v=>v.b),name:'Quỹ đạo',line:{color:'#f28db9',width:2}},
    {...points([frame.w],[frame.b],'Vị trí hiện tại'),marker:{color:'#ffd173',size:12,line:{color:'#fff1cd',width:1}}}];
  return <>
    <p>UCI Real Estate, 414 giao dịch. Mục tiêu là giá trên đơn vị diện tích, tính theo 10.000 TWD/ping.</p>
    <div className="py30-grid">
      <Range label={day===3?'Hệ số w':'Hệ số w ban đầu'} value={w} min={-20} max={15} step={.1} change={setW}/><Range label={day===3?'Hệ số b':'Hệ số b ban đầu'} value={b} min={-25} max={125} step={.1} change={setB}/>
      {day!==3&&<Range label="Learning rate" value={lr} min={.001} max={.25} step={.001} change={setLR}/>}
      {day===5&&<><label>Thuật toán <select value={batch} onChange={e=>setBatch(+e.target.value)}><option value={x.length}>Batch GD</option><option value={1}>SGD</option><option value={16}>Mini-batch (16)</option><option value={64}>Mini-batch (64)</option></select></label><label><input type="checkbox" checked={scale} onChange={e=>setScale(e.target.checked)}/> Chuẩn hóa log(1 + MRT)</label></>}
    </div>
    {day!==3&&<Transport player={player} count={count}/>}
    <div className="simulation-studio-grid">
      <SimulationPanel title="Đường dự đoán và phần dư">
        <Chart xTitle={day===5&&scale?'Khoảng cách MRT chuẩn hóa':'log(1 + khoảng cách MRT)'} yTitle="Giá trên diện tích" traces={[
          points(x,y,'Quan sát'),line(chosenX,chosenX.map(u=>frame.w*u+frame.b),'Dự đoán'),
          {type:'scatter',mode:'lines',x:x.slice(0,25).flatMap(u=>[u,u,null]),y:x.slice(0,25).flatMap((u,i)=>[y[i],frame.w*u+frame.b,null]),name:'Phần dư của 25 mẫu đầu',line:{color:'#e85d35',width:1}},
        ]}/>
      </SimulationPanel>
      <SimulationPanel title="Hành trình giảm sai số">
        <Chart xTitle="Hệ số w" yTitle="Hệ số b" traces={traces} layout={{xaxis:{title:{text:'Hệ số w'},range:[grid.x[0],grid.x.at(-1)]},yaxis:{title:{text:'Hệ số b'},range:[grid.y[0],grid.y.at(-1)],...(day===5?{scaleanchor:'x',scaleratio:1}:{})}}}/>
        <p className="small-note">Nền biểu diễn log₁₀(MSE). Điểm vàng là cặp hệ số đang dùng để dự đoán.</p>
      </SimulationPanel>
      {day!==3&&<SimulationPanel title="Python tương ứng với lần cập nhật">
        <SimulationCode lines={['xb, yb = x[batch], y[batch]','error = w * xb + b - yb','dw, db = 2*np.mean(xb*error), 2*np.mean(error)','w, b = w - alpha*dw, b - alpha*db']} active={phase}
          detail={phase<0?<p>Hệ số khởi tạo. Bấm từng bước để theo dõi lần cập nhật đầu tiên.</p>:<Table headers={['Mẫu đầu','Số mẫu','Sai số trước cập nhật','∂MSE/∂w','∂MSE/∂b']} rows={[[operation.sample+1,operation.batchSize,phase>=1?fmt(operation.residual):'—',phase>=2?fmt(operation.dw):'—',phase>=2?fmt(operation.db):'—']]}/>}/>
      </SimulationPanel>}
      <SimulationPanel title="Trạng thái mô hình">
        <Table headers={['w','b','MSE','Lượt cập nhật','Số mẫu đã xem']} rows={[[fmt(frame.w),fmt(frame.b),fmt(frame.loss),completed,frame.samples]]}/>
      </SimulationPanel>
    </div>
    {day!==3&&history.length<56&&<p role="status">Dấu hiệu phân kỳ. Hãy giảm learning rate hoặc chuẩn hóa đầu vào rồi thử lại.</p>}
  </>;
}

function ridge(x:number[][],target:number[],lambda:number) {
  const A=x.map(row=>[1,...row]);
  const d=A[0].length;
  const gram=Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>
    A.reduce((s,row)=>s+row[i]*row[j],0)+(i===j&&i>0?lambda+1e-8:0)));
  const rhs=Array.from({length:d},(_,i)=>A.reduce((s,row,n)=>s+row[i]*target[n],0));
  const w=solve(gram,rhs);
  return (row:number[])=>w[0]+row.reduce((s,v,i)=>s+w[i+1]*v,0);
}
const poly=(x:number,d:number)=>Array.from({length:d},(_,i)=>x**(i+1));
const noise=rng(91);
const simulated=Array.from({length:58},(_,i)=>{
  const x=i<20?-.96+.1*i:-.98+.052*(i-20);
  return [x,Math.sin(3*x)+.32*(noise()-.5)];
});
function Polynomial() {
  const [degree,setDegree]=useState(5),[lambda,setLambda]=useState(0);
  const train=simulated.slice(0,20),validation=simulated.slice(20);
  const fit=(d:number)=>ridge(train.map(p=>poly(p[0],d)),train.map(p=>p[1]),lambda);
  const predict=fit(degree);
  const grid=Array.from({length:101},(_,i)=>-1+2*i/100);
  const degrees=Array.from({length:15},(_,i)=>i+1);
  const errors=degrees.map(d=>{
    const f=fit(d);
    return [train,validation].map(set=>mean(set.map(([x,y])=>(f(poly(x,d))-y)**2)));
  });
  return <><p>Dữ liệu giả lập để quan sát ảnh hưởng của bậc và λ. Các cột đa thức tính trên miền [−1, 1].</p>
    <Range label="Bậc đa thức" value={degree} min={1} max={15} change={setDegree}/><Range label="λ Ridge" value={lambda} min={0} max={1} step={.01} change={setLambda}/>
    <Chart xTitle="x" yTitle="y" traces={[points(train.map(p=>p[0]),train.map(p=>p[1]),'Train'),points(validation.map(p=>p[0]),validation.map(p=>p[1]),'Validation'),line(grid,grid.map(x=>predict(poly(x,degree))),'Dự đoán')]}/>
    <Chart height={260} xTitle="Bậc" yTitle="MSE" traces={[line(degrees,errors.map(e=>e[0]),'Train'),line(degrees,errors.map(e=>e[1]),'Validation')]}/>
  </>;
}

const features=[{column:2,label:'Tuổi nhà'},{column:3,label:'log(1 + MRT)'},{column:4,label:'Số cửa hàng'}];
const splitRandom=rng(42);
const shuffled=rows.map((_,i)=>({i,key:splitRandom()})).sort((a,b)=>a.key-b.key).map(r=>r.i);
const trainIds=shuffled.slice(0,330),valIds=shuffled.slice(330);
function Pipeline() {
  const [columns,setColumns]=useState([2,3]),[lambda,setLambda]=useState(1);
  const transform=(id:number,j:number)=>j===3?Math.log1p(rows[id][j]):rows[id][j];
  const mu=columns.map(j=>mean(trainIds.map(i=>transform(i,j))));
  const sd=columns.map((j,k)=>Math.sqrt(mean(trainIds.map(i=>(transform(i,j)-mu[k])**2)))||1);
  const x=(id:number)=>columns.map((j,k)=>(transform(id,j)-mu[k])/sd[k]);
  const ys=trainIds.map(i=>y[i]),predict=ridge(trainIds.map(x),ys,lambda);
  const preds=valIds.map(i=>predict(x(i))),res=valIds.map((i,k)=>y[i]-preds[k]);
  const sorted=[...ys].sort((a,b)=>a-b),baseline=(sorted[164]+sorted[165])/2;
  return <><p>Thử pipeline trên UCI Real Estate. Trong bài thực hành bạn áp dụng các bước này cho House Prices và xuất submission.</p>
    <div className="py30-actions">{features.map(({column,label})=><label key={column}><input type="checkbox" checked={columns.includes(column)} onChange={e=>setColumns(e.target.checked?[...columns,column]:columns.filter(x=>x!==column))}/> {label}</label>)}</div>
    <Range label="λ" value={lambda} min={0} max={100} change={setLambda}/>
    <p role="status">MAE validation: {fmt(mean(res.map(Math.abs)))} · MAE baseline median train: {fmt(mean(valIds.map(i=>Math.abs(y[i]-baseline))))}</p>
    <Chart xTitle="Giá dự đoán" yTitle="Phần dư (giá thật − dự đoán)" traces={[points(preds,res,'Validation')]}/>
  </>;
}
export function WeekOneLab({day}:{day:number}) {
  if(day===1)return <Broadcasting/>;
  if(day===2)return <Leakage/>;
  if(day<=5)return <Regression day={day}/>;
  return day===6?<Polynomial/>:<Pipeline/>;
}
