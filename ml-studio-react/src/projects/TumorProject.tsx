import {useMemo,useState} from 'react';
import type {Route} from '../types';
import breast from '../track-py/data/breast2d.json';
import {Confusion,LineChart,Plot,PlayerBar,ProjectFrame,Slider,Stat,fmt,pct,shuffled,usePlayer} from './ui';
import type {Scale,StepDef,ReverseDef} from './ui';

type Row={x:number;y:number;label:0|1};
const rows:Row[]=(breast.points as number[][]).map(p=>({x:p[0],y:p[1],label:(p[2]===1?1:0)}));
const order=shuffled(rows.map((_,i)=>i),7);
const train=order.slice(0,400).map(i=>rows[i]),valid=order.slice(400).map(i=>rows[i]);
const X:[number,number]=[-2.3,4.1],Y:[number,number]=[-2.5,4.4];
const RED='#f28db9',BLUE='#42d3e5';
const sig=(z:number)=>1/(1+Math.exp(-z));

interface Fit {w:[number,number];b:number;loss:number;acc:number}
function trainLogistic(epochs=150,lr=.4):Fit[]{
  let w:[number,number]=[0,0],b=0;const out:Fit[]=[];
  for(let e=0;e<=epochs;e++){
    let gw0=0,gw1=0,gb=0,loss=0,ok=0;
    for(const r of train){const p=sig(w[0]*r.x+w[1]*r.y+b),d=p-r.label;gw0+=d*r.x;gw1+=d*r.y;gb+=d;loss-=r.label?Math.log(p+1e-9):Math.log(1-p+1e-9);ok+=+((p>=.5)===(r.label===1))}
    out.push({w:[...w],b,loss:loss/train.length,acc:ok/train.length});
    const n=train.length;w=[w[0]-lr*gw0/n,w[1]-lr*gw1/n];b-=lr*gb/n;
  }
  return out;
}
function knn(q:{x:number;y:number},k:number,data=train){
  const near=data.map(r=>({r,d:Math.hypot(r.x-q.x,r.y-q.y)})).sort((a,b)=>a.d-b.d).slice(0,k);
  const votes=near.filter(n=>n.r.label===1).length;return {near,votes,pred:(votes*2>k?1:0) as 0|1};
}
function count(data:Row[],predict:(r:Row)=>0|1){
  let tp=0,fp=0,tn=0,fn=0;for(const r of data){const p=predict(r);if(p&&r.label)tp++;else if(p)fp++;else if(r.label)fn++;else tn++}
  return {tp,fp,tn,fn,acc:(tp+tn)/data.length,recall:tp/Math.max(1,tp+fn),precision:tp/Math.max(1,tp+fp)};
}
function Points({data,sx,sy,predict}:{data:Row[];sx:Scale;sy:Scale;predict?:(r:Row)=>0|1}){
  return <>{data.map((r,i)=>{const wrong=predict&&predict(r)!==r.label;return <circle key={i} cx={sx(r.x)} cy={sy(r.y)} r={wrong?4.6:3.3} fill={r.label?RED:BLUE} fillOpacity={wrong?1:.75} stroke={wrong?'#fff':'none'} strokeWidth={1.4}/>})}</>;
}
/** Đường thẳng a·x + b·y = c cắt ngang vùng vẽ. */
function Boundary({a,b,c,sx,sy,color='#ffd173'}:{a:number;b:number;c:number;sx:Scale;sy:Scale;color?:string}){
  if(Math.abs(b)<1e-6){const x=c/a;return <line x1={sx(x)} x2={sx(x)} y1={sy(Y[0])} y2={sy(Y[1])} stroke={color} strokeWidth={2.5}/>}
  const yAt=(x:number)=>(c-a*x)/b;return <line x1={sx(X[0])} y1={sy(yAt(X[0]))} x2={sx(X[1])} y2={sy(yAt(X[1]))} stroke={color} strokeWidth={2.5}/>;
}
function Heat({prob,sx,sy}:{prob:(x:number,y:number)=>number;sx:Scale;sy:Scale}){
  const nx=32,ny=24,cells=[];const dx=(X[1]-X[0])/nx,dy=(Y[1]-Y[0])/ny;
  for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){const x=X[0]+(i+.5)*dx,y=Y[0]+(j+.5)*dy,p=prob(x,y);cells.push(<rect key={i+'-'+j} x={sx(x-dx/2)} y={sy(y+dy/2)} width={sx(x+dx/2)-sx(x-dx/2)+.5} height={sy(y-dy/2)-sy(y+dy/2)+.5} fill={p>=.5?RED:BLUE} opacity={.06+.22*Math.abs(p-.5)*2}/>)}
  return <>{cells}</>;
}
const legend=<p className="chart-caption"><span style={{color:BLUE}}>●</span> lành tính · <span style={{color:RED}}>●</span> ác tính · viền trắng = mô hình đoán sai</p>;

export function TumorProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [pr,setPr]=useState({x:1.2,y:.8});
  const [t1,setT1]=useState(1.5);
  const [angle,setAngle]=useState(30),[offset,setOffset]=useState(.6);
  const fits=useMemo(()=>trainLogistic(),[]);
  const player=usePlayer(fits.length,60);
  const [query,setQuery]=useState({x:.6,y:.4}),[k,setK]=useState(5);
  const [thr,setThr]=useState(.5);
  const final=fits[fits.length-1];
  const knnCurve=useMemo(()=>[1,3,5,7,9,15,21,31,51].map(kk=>({k:kk,acc:count(valid,r=>knn(r,kk).pred).acc})),[]);

  // Bước 1: một ngưỡng trên bán kính
  const s1=count(train,r=>r.x>t1?1:0);
  // Bước 2: đường thẳng em tự đặt
  const a=Math.cos(angle*Math.PI/180),b=Math.sin(angle*Math.PI/180);
  const s2=count(train,r=>a*r.x+b*r.y>offset?1:0);
  // Bước 3: logistic theo từng epoch
  const fit=fits[player.frame];
  const s3=count(train,r=>sig(fit.w[0]*r.x+fit.w[1]*r.y+fit.b)>=.5?1:0);
  // Bước 4: KNN
  const nn=knn(query,k);
  // Bước 5: ngưỡng xác suất
  const pv=(r:Row)=>sig(final.w[0]*r.x+final.w[1]*r.y+final.b);
  const s5=count(valid,r=>pv(r)>=thr?1:0),base=count(valid,()=>0);

  const steps:StepDef[]=[
    {label:'Một ngưỡng',title:'Chỉ được nhìn một con số.',
      intro:<p>Mỗi chấm là một khối u, trục ngang là bán kính (đã chuẩn hóa). Em kéo ngưỡng: lớn hơn ngưỡng thì gọi là <b>ác tính</b>. Thử tìm ngưỡng ít sai nhất.</p>,
      controls:<><Slider label="Ngưỡng bán kính" value={t1} min={-1.5} max={3} step={.05} onChange={setT1} format={v=>fmt(v)}/>
        <div className="stat-row"><Stat label="Đúng" value={pct(s1.acc)}/><Stat label="Bỏ sót ác tính" value={s1.fn} tone={s1.fn>20?'bad':undefined}/><Stat label="Báo nhầm" value={s1.fp}/></div></>,
      visual:<><Plot x={X} y={Y} xLabel="Bán kính trung bình" yLabel="Độ sần bề mặt">{(sx,sy)=><><Points data={train} sx={sx} sy={sy} predict={r=>r.x>t1?1:0}/><Boundary a={1} b={0} c={t1} sx={sx} sy={sy}/></>}</Plot>{legend}</>,
      takeaway:'Một ngưỡng đã là một mô hình. Nhưng kéo về bên nào cũng phải đánh đổi: giảm báo nhầm thì tăng bỏ sót. Muốn tốt hơn phải nhìn thêm thông tin.'},
    {label:'Tự vẽ ranh giới',title:'Dùng cả hai con số.',
      intro:<p>Giờ em xoay và dịch một đường thẳng. Phía bên phải-trên đường là ác tính. Số chấm viền trắng là số ca em đoán sai.</p>,
      controls:<><Slider label="Góc xoay" value={angle} min={0} max={180} onChange={setAngle} format={v=>v+'°'}/><Slider label="Dịch chuyển" value={offset} min={-2} max={3} step={.05} onChange={setOffset} format={v=>fmt(v)}/>
        <div className="stat-row"><Stat label="Đúng" value={pct(s2.acc)}/><Stat label="Bỏ sót" value={s2.fn}/><Stat label="Báo nhầm" value={s2.fp}/></div></>,
      visual:<><Plot x={X} y={Y} xLabel="Bán kính trung bình" yLabel="Độ sần bề mặt">{(sx,sy)=><><Points data={train} sx={sx} sy={sy} predict={r=>a*r.x+b*r.y>offset?1:0}/><Boundary a={a} b={b} c={offset} sx={sx} sy={sy}/></>}</Plot>{legend}</>,
      takeaway:`Hai đặc trưng cho phép một đường chéo tách tốt hơn một ngưỡng (${pct(s2.acc)} so với ${pct(s1.acc)} lúc ở bước 1 nếu em đã chỉnh khéo). Việc tìm góc và vị trí tốt nhất chính là việc máy sẽ làm.`},
    {label:'Cho máy học',title:'Hồi quy logistic tự xoay đường thẳng.',
      intro:<p>Máy bắt đầu từ trọng số 0 và mỗi epoch dịch đường một chút để giảm <b>log-loss</b>. Nền tô màu là xác suất ác tính: đậm là chắc chắn, nhạt là phân vân.</p>,
      controls:<><PlayerBar player={player} count={fits.length} label="Epoch"/>
        <div className="stat-row"><Stat label="Loss" value={fmt(fit.loss,3)}/><Stat label="Đúng (train)" value={pct(s3.acc)}/></div>
        <p className="small-note">w = ({fmt(fit.w[0])}; {fmt(fit.w[1])}), b = {fmt(fit.b)}</p></>,
      visual:<><Plot x={X} y={Y} xLabel="Bán kính trung bình" yLabel="Độ sần bề mặt">{(sx,sy)=><><Heat prob={(x,y)=>sig(fit.w[0]*x+fit.w[1]*y+fit.b)} sx={sx} sy={sy}/><Points data={train} sx={sx} sy={sy} predict={r=>sig(fit.w[0]*r.x+fit.w[1]*r.y+fit.b)>=.5?1:0}/>{(fit.w[0]||fit.w[1])?<Boundary a={fit.w[0]} b={fit.w[1]} c={-fit.b} sx={sx} sy={sy}/>:null}</>}</Plot>
        <LineChart height={150} series={[{name:'Loss',values:fits.slice(0,player.frame+1).map(f=>f.loss),color:'#ffd173'}]} xLabel="Epoch"/></>,
      takeaway:'Logistic không chỉ vẽ đường ranh giới mà còn cho xác suất. Gần đường thì xác suất quanh 50%: đó là những ca cần bác sĩ xem kỹ.'},
    {label:'Hàng xóm gần nhất',title:'Không cần đường thẳng: hỏi các ca giống nhất.',
      intro:<p>Bấm vào biểu đồ để đặt một khối u mới. KNN tìm <b>k</b> ca gần nhất trong dữ liệu và cho chúng bỏ phiếu.</p>,
      controls:<><Slider label="Số hàng xóm k" value={k} min={1} max={31} step={2} onChange={setK}/>
        <div className="stat-row"><Stat label="Phiếu ác tính" value={`${nn.votes}/${k}`}/><Stat label="Kết luận" value={nn.pred?'Ác tính':'Lành tính'} tone={nn.pred?'bad':'good'}/></div>
        <p className="small-note">Độ chính xác trên tập kiểm tra theo k:</p>
        <div className="mini-cols">{knnCurve.map(c=><div key={c.k} className={c.k===k?'on':''}><i style={{height:`${(c.acc-.75)*400}%`}}/><span>{c.k}</span></div>)}</div></>,
      visual:<><Plot x={X} y={Y} xLabel="Bán kính trung bình" yLabel="Độ sần bề mặt" onPick={(x,y)=>setQuery({x,y})}>{(sx,sy)=><><Points data={train} sx={sx} sy={sy}/>{nn.near.map((n,i)=><line key={i} x1={sx(query.x)} y1={sy(query.y)} x2={sx(n.r.x)} y2={sy(n.r.y)} stroke={n.r.label?RED:BLUE} strokeWidth={1.4} opacity={.9}/>)}<circle cx={sx(query.x)} cy={sy(query.y)} r={8} fill="#ffd173" stroke="#0a1220" strokeWidth={2}/></>}</Plot><p className="chart-caption">Bấm bất kỳ đâu trên biểu đồ để đặt ca mới.</p></>,
      takeaway:'k nhỏ thì nghe theo vài ca rất gần, dễ bị nhiễu. k lớn thì mượt nhưng xóa mất chi tiết. Chọn k bằng tập kiểm tra, không phải bằng tập huấn luyện.'},
    {label:'Cái giá của sai lầm',title:'Accuracy cao vẫn có thể nguy hiểm.',
      intro:<p>Trên 169 ca kiểm tra máy chưa từng thấy, em chọn ngưỡng xác suất. Bỏ sót ca ác tính nguy hiểm hơn nhiều so với báo nhầm.</p>,
      controls:<><Slider label="Ngưỡng xác suất ác tính" value={thr} min={.05} max={.95} step={.05} onChange={setThr} format={v=>fmt(v)}/>
        <div className="stat-row"><Stat label="Accuracy" value={pct(s5.acc)}/><Stat label="Recall ác tính" value={pct(s5.recall)} tone={s5.recall<.85?'bad':'good'}/><Stat label="Precision" value={pct(s5.precision)}/></div>
        <p className="observation">Một "mô hình" luôn đoán <b>lành tính</b> vẫn đúng <strong>{pct(base.acc)}</strong> nhưng bỏ sót <b>tất cả {base.fn}</b> ca ác tính.</p></>,
      visual:<><Confusion labels={['Lành','Ác']} matrix={[[s5.tn,s5.fp],[s5.fn,s5.tp]]} highlight={[1,0]}/>
        <p className="chart-caption">Ô tô đỏ: ca ác tính bị bỏ sót. Hạ ngưỡng để làm nó nhỏ lại và xem precision thay đổi ra sao.</p></>,
      takeaway:'Chọn ngưỡng là quyết định về con người, không phải về toán: em cần biết bỏ sót tốn kém hơn báo nhầm bao nhiêu. Khi dữ liệu lệch lớp, hãy nhìn recall và precision thay vì chỉ accuracy.'},
  ];
  const prodP=sig(final.w[0]*pr.x+final.w[1]*pr.y+final.b);
  const reverse:ReverseDef={product:{label:'Dùng sản phẩm',title:'Một bộ chẩn đoán đã huấn luyện xong.',
    intro:<p>Đây là mô hình hoàn chỉnh. Kéo hai số đo của một khối u mới và xem nó trả lời. Chưa cần biết bên trong có gì: chỉ quan sát khi nào nó <b>tự tin</b>, khi nào <b>phân vân</b>.</p>,
    controls:<><Slider label="Bán kính" value={pr.x} min={-2} max={4} step={.05} onChange={v=>setPr({...pr,x:v})} format={v=>fmt(v)}/><Slider label="Độ sần bề mặt" value={pr.y} min={-2.4} max={4.2} step={.05} onChange={v=>setPr({...pr,y:v})} format={v=>fmt(v)}/>
      <div className="stat-row"><Stat label="Xác suất ác tính" value={pct(prodP)} tone={prodP>=.5?'bad':'good'}/><Stat label="Kết luận" value={prodP>=.5?'Ác tính':'Lành tính'}/></div>
      <p className="small-note">Sản phẩm đúng {pct(final.acc)} trên 400 ca huấn luyện và {pct(count(valid,r=>pv(r)>=.5?1:0).acc)} trên 169 ca kiểm tra.</p></>,
    visual:<><Plot x={X} y={Y} xLabel="Bán kính trung bình" yLabel="Độ sần bề mặt" onPick={(x,y)=>setPr({x,y})}>{(sx,sy)=><><Heat prob={(x,y)=>sig(final.w[0]*x+final.w[1]*y+final.b)} sx={sx} sy={sy}/><circle cx={sx(pr.x)} cy={sy(pr.y)} r={9} fill="#ffd173" stroke="#0a1220" strokeWidth={2}/></>}</Plot><p className="chart-caption">Nền đậm là vùng sản phẩm tự tin, nhạt là vùng phân vân. Bấm lên biểu đồ để đặt ca mới.</p></>,
    takeaway:'Sản phẩm nhận 2 con số và trả về một xác suất. Vùng phân vân là một dải hẹp chạy chéo: manh mối đầu tiên cho thấy bên trong có một đường thẳng.'},
  layers:[
    {from:2,label:'Mở hộp: w và b',title:'Bên trong chỉ là ba con số.',intro:<p>Mở mô hình ra: nó gồm w₁, w₂ và b, tạo thành một đường thẳng. Kéo về <b>epoch 0</b> để thấy lúc chưa học nó trống rỗng, rồi bấm Chạy để xem ba con số được tìm ra.</p>,takeaway:'Toàn bộ “trí khôn” của sản phẩm nằm trong ba con số tìm được bằng gradient descent. Dải phân vân ở lớp trước chính là vùng quanh đường thẳng này.'},
    {from:4,label:'Quy tắc quyết định',title:'Từ xác suất đến câu trả lời có hay không.',intro:<p>Sản phẩm nói “ác tính” khi xác suất vượt 0,5. Con số 0,5 không do máy học mà do người chọn. Đổi nó trên 169 ca kiểm tra.</p>,takeaway:'Lớp ngoài cùng là một quyết định của con người: chấp nhận báo nhầm bao nhiêu để bớt bỏ sót. Accuracy một mình không nói được điều đó.'},
    {from:3,label:'Thay động cơ',title:'Không có đường thẳng vẫn chạy được?',intro:<p>Tháo động cơ logistic, lắp KNN vào: không học tham số nào, chỉ hỏi các ca giống nhất. Bấm lên biểu đồ để đặt ca mới.</p>,takeaway:'Hai động cơ rất khác nhau cho kết quả gần nhau trên dữ liệu này. Hiểu sản phẩm là biết bộ phận nào thay được, bộ phận nào cốt lõi.'},
    {from:0,label:'Tháo bớt đầu vào',title:'Bỏ đi một đặc trưng thì sao?',intro:<p>Tháo đặc trưng “độ sần”, chỉ còn bán kính. Mô hình lúc này chỉ còn là một ngưỡng. Kéo thử và so với sản phẩm đầy đủ.</p>,takeaway:'Bỏ một đặc trưng là mất vài phần trăm độ chính xác và bỏ sót nhiều hơn. Tháo từng bộ phận là cách nhanh nhất biết bộ phận nào đáng giá.'},
    {from:1,label:'Tự lắp lại',title:'Giờ em lắp lại bằng tay.',intro:<p>Em đã biết bên trong là một đường thẳng. Tự xoay và dịch nó để đạt độ chính xác gần bằng sản phẩm (khoảng 90% trên tập huấn luyện).</p>,takeaway:'Lắp lại được gần bằng máy nghĩa là em đã hiểu sản phẩm: một đường thẳng tốt nhất, một ngưỡng do người chọn, trên dữ liệu hai đặc trưng.'},
  ]};
  return <ProjectFrame id="tumor" steps={steps} reverse={reverse} navigate={navigate}/>;
}
