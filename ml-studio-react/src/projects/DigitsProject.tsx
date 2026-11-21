import {useMemo,useRef,useState} from 'react';
import type {PointerEvent} from 'react';
import type {Route} from '../types';
import digits from '../track-py/data/digits.json';
import {Bars,Confusion,LineChart,ProjectFrame,Slider,Stat,pct,shuffled} from './ui';
import type {StepDef} from './ui';

const pixels=digits.samples.pixels as number[][],labels=digits.samples.label as number[];
const order=shuffled(pixels.map((_,i)=>i),17),trainIdx=order.slice(0,300),testIdx=order.slice(300);
const norm=(p:number[])=>p.map(v=>v/16);
const D=['0','1','2','3','4','5','6','7','8','9'];

function PixelGrid({values,size=120,diverging=false,max=16,onPaint,showNumbers=false}:{values:number[];size?:number;diverging?:boolean;max?:number;onPaint?:(i:number,erase:boolean)=>void;showNumbers?:boolean}){
  const down=useRef(false),erase=useRef(false);const n=Math.round(Math.sqrt(values.length)),c=size/n;
  const at=(e:PointerEvent<SVGSVGElement>)=>{const r=e.currentTarget.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*n),y=Math.floor((e.clientY-r.top)/r.height*n);if(x>=0&&x<n&&y>=0&&y<n)onPaint?.(y*n+x,erase.current)};
  return <svg className={'pixel-grid'+(onPaint?' paint':'')} viewBox={`0 0 ${size} ${size}`} width={size} height={size} style={{touchAction:onPaint?'none':undefined}}
    onPointerDown={e=>{if(!onPaint)return;down.current=true;erase.current=e.button===2||e.shiftKey;e.currentTarget.setPointerCapture(e.pointerId);at(e)}} onPointerMove={e=>{if(down.current)at(e)}} onPointerUp={()=>{down.current=false}} onContextMenu={e=>onPaint&&e.preventDefault()}>
    {values.map((v,i)=>{const t=Math.max(-1,Math.min(1,v/max));const fill=diverging?(t>=0?`rgba(66,211,229,${t})`:`rgba(242,141,185,${-t})`):`rgba(230,238,248,${Math.max(0,t)})`;
      return <g key={i}><rect x={(i%n)*c} y={Math.floor(i/n)*c} width={c-.6} height={c-.6} rx={1.5} fill={fill} stroke="var(--line)" strokeWidth={.4}/>{showNumbers&&<text x={(i%n)*c+c/2} y={Math.floor(i/n)*c+c/2+3.5} textAnchor="middle" className="pix-num" fill={t>.55?'#0a1220':'var(--text-3)'}>{Math.round(v)}</text>}</g>})}
  </svg>;
}

interface Snap {epoch:number;W:number[][];b:number[];acc:number;testAcc:number;loss:number}
function softmax(z:number[]){const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
const logits=(W:number[][],b:number[],x:number[])=>W.map((w,k)=>w.reduce((s,v,j)=>s+v*x[j],b[k]));
function trainSoftmax(){
  const X=trainIdx.map(i=>norm(pixels[i])),Y=trainIdx.map(i=>labels[i]),XT=testIdx.map(i=>norm(pixels[i])),YT=testIdx.map(i=>labels[i]);
  let W=Array.from({length:10},()=>Array(64).fill(0)),b=Array(10).fill(0);const snaps:Snap[]=[],keep=new Set([0,1,2,3,5,8,12,20,30,45,70,100]);
  const acc=(xs:number[][],ys:number[])=>xs.filter((x,i)=>{const z=logits(W,b,x);return z.indexOf(Math.max(...z))===ys[i]}).length/xs.length;
  for(let e=0;e<=100;e++){
    const gW=W.map(r=>r.map(()=>0)),gb=Array(10).fill(0);let loss=0;
    X.forEach((x,i)=>{const p=softmax(logits(W,b,x));loss-=Math.log(p[Y[i]]+1e-9);p.forEach((pk,k)=>{const d=pk-(k===Y[i]?1:0);gb[k]+=d;for(let j=0;j<64;j++)gW[k][j]+=d*x[j]})});
    if(keep.has(e))snaps.push({epoch:e,W:W.map(r=>[...r]),b:[...b],acc:acc(X,Y),testAcc:acc(XT,YT),loss:loss/X.length});
    const lr=.8/X.length;W=W.map((r,k)=>r.map((v,j)=>v-lr*gW[k][j]-1e-4*v));b=b.map((v,k)=>v-lr*gb[k]);
  }
  return snaps;
}
const KERNELS:{name:string;k:number[]}[]=[
  {name:'Cạnh dọc',k:[-1,0,1,-2,0,2,-1,0,1]},{name:'Cạnh ngang',k:[-1,-2,-1,0,0,0,1,2,1]},{name:'Chéo /',k:[0,1,2,-1,0,1,-2,-1,0]},{name:'Làm mờ',k:[1,1,1,1,1,1,1,1,1].map(v=>v/9)}];
function conv(img:number[],k:number[]){const out:number[]=[];for(let y=0;y<6;y++)for(let x=0;x<6;x++){let s=0;for(let a=0;a<3;a++)for(let c=0;c<3;c++)s+=img[(y+a)*8+x+c]*k[a*3+c];out.push(Math.max(0,s))}return out}

export function DigitsProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [step,setStep]=useState(0);
  const [img,setImg]=useState<number[]>(()=>[...pixels[testIdx[0]]]);
  const [k,setK]=useState(5);
  const snaps=useMemo(()=>trainSoftmax(),[]);
  const [snapI,setSnapI]=useState(snaps.length-1);
  const [kernel,setKernel]=useState(0);
  const paint=(i:number,erase:boolean)=>setImg(p=>{const n=[...p];const add=(j:number,v:number)=>{if(j>=0&&j<64)n[j]=erase?0:Math.min(16,n[j]+v)};add(i,9);if(!erase){if(i%8)add(i-1,3);if(i%8<7)add(i+1,3);add(i-8,3);add(i+8,3)}return n});
  const randomSample=()=>setImg([...pixels[testIdx[Math.floor(Math.random()*testIdx.length)]]]);
  const drawPad=<div className="draw-pad"><PixelGrid values={img} size={200} onPaint={paint}/><div className="draw-actions"><button className="btn-outline" onClick={()=>setImg(Array(64).fill(0))}>Xóa</button><button className="btn-outline" onClick={randomSample}>Lấy một ảnh mẫu</button></div><p className="small-note">Kéo chuột để vẽ, giữ Shift (hoặc chuột phải) để tẩy.</p></div>;

  const near=trainIdx.map(i=>({i,d:Math.hypot(...pixels[i].map((v,j)=>v-img[j]))})).sort((a,b)=>a.d-b.d).slice(0,k);
  const votes=Array(10).fill(0);near.forEach(n=>votes[labels[n.i]]++);const knnPred=votes.indexOf(Math.max(...votes));
  const snap=snaps[snapI],final=snaps[snaps.length-1];
  const probs=softmax(logits(snap.W,snap.b,norm(img)));const pred=probs.indexOf(Math.max(...probs));
  const cm=Array.from({length:10},()=>Array(10).fill(0));const mistakes:{i:number;p:number}[]=[];
  testIdx.forEach(i=>{const z=logits(final.W,final.b,norm(pixels[i])),p=z.indexOf(Math.max(...z));cm[labels[i]][p]++;if(p!==labels[i])mistakes.push({i,p})});
  const maps=KERNELS.map(kk=>conv(img,kk.k));

  const steps:StepDef[]=[
    {label:'Máy nhìn thấy gì',title:'Một ảnh chỉ là 64 con số.',
      intro:<p>Vẽ một chữ số lên lưới 8×8. Với máy, mỗi ô là một số từ 0 (trắng) đến 16 (đậm). Ghép 64 số thành một hàng là có <b>vector</b> đầu vào.</p>,
      controls:drawPad,
      visual:<div className="digit-numbers"><PixelGrid values={img} size={300} showNumbers/><p className="small-note">Vector: [{img.slice(0,12).map(v=>Math.round(v)).join(', ')}, …] (64 số)</p></div>,
      takeaway:'Máy không thấy "nét" hay "vòng tròn", chỉ thấy bảng số. Mọi thứ phía sau, từ KNN đến CNN, đều là các cách so sánh và biến đổi bảng số này.'},
    {label:'Hàng xóm giống nhất',title:'Tìm những ảnh có bảng số gần nhất.',
      intro:<p>KNN đo khoảng cách giữa ảnh của em và từng ảnh trong 300 ảnh huấn luyện, rồi cho <b>k</b> ảnh gần nhất bỏ phiếu.</p>,
      controls:<>{drawPad}<Slider label="Số hàng xóm k" value={k} min={1} max={15} step={2} onChange={setK}/><div className="stat-row"><Stat label="KNN đoán" value={knnPred}/><Stat label="Số phiếu" value={`${votes[knnPred]}/${k}`}/></div></>,
      visual:<div className="neighbor-grid">{near.map(n=><figure key={n.i} className={labels[n.i]===knnPred?'agree':''}><PixelGrid values={pixels[n.i]} size={88}/><figcaption><b>{labels[n.i]}</b> · cách {Math.round(n.d)}</figcaption></figure>)}</div>,
      takeaway:'KNN không học gì cả, chỉ nhớ và so sánh. Dịch chữ số sang trái một ô là khoảng cách tăng vọt dù với mắt người vẫn y hệt: so từng pixel rất mong manh.'},
    {label:'Mạng học khuôn mẫu',title:'Mỗi chữ số có một tấm "khuôn" trọng số.',
      intro:<p>Mạng một lớp (softmax) có 10 tấm trọng số 8×8. Điểm của chữ số <i>d</i> = tổng (pixel × trọng số). <span style={{color:'var(--accent)'}}>Xanh</span> là chỗ nên có mực, <span style={{color:'var(--pink)'}}>hồng</span> là chỗ không nên có mực.</p>,
      controls:<><Slider label="Epoch huấn luyện" value={snapI} min={0} max={snaps.length-1} onChange={setSnapI} format={i=>String(snaps[i].epoch)}/>
        <div className="stat-row"><Stat label="Đúng (train)" value={pct(snap.acc)}/><Stat label="Đúng (test)" value={pct(snap.testAcc)}/></div>{drawPad}</>,
      visual:<><div className="template-grid">{snap.W.map((w,d)=><figure key={d} className={d===pred?'agree':''}><PixelGrid values={w} size={76} diverging max={Math.max(.05,...snap.W.flat().map(Math.abs))}/><figcaption>{d} · {pct(probs[d])}</figcaption></figure>)}</div>
        <LineChart height={130} xLabel="Mốc huấn luyện" series={[{name:'Train',values:snaps.slice(0,snapI+1).map(s=>s.acc),color:'#84e0a8'},{name:'Test',values:snaps.slice(0,snapI+1).map(s=>s.testAcc),color:'#f28db9'}]}/></>,
      takeaway:'Sau khi học, các tấm khuôn trông mờ mờ giống chữ số tương ứng: trọng số chính là thứ mạng đã "nhìn ra" từ dữ liệu. Mạng nhiều lớp làm điều này nhiều tầng, mỗi tầng ghép khuôn của tầng trước.'},
    {label:'Bộ lọc tích chập',title:'Trượt một khung 3×3 để tìm nét.',
      intro:<p>Thay vì nhìn từng pixel riêng lẻ, CNN trượt một <b>bộ lọc</b> nhỏ khắp ảnh. Bộ lọc cạnh dọc sáng lên ở chỗ có nét thẳng đứng, ở đâu cũng vậy.</p>,
      controls:<>{drawPad}<div className="segmented">{KERNELS.map((kk,i)=><button key={kk.name} className={i===kernel?'on':''} onClick={()=>setKernel(i)}>{kk.name}</button>)}</div>
        <div className="kernel-view"><PixelGrid values={KERNELS[kernel].k} size={96} diverging max={2} showNumbers/></div></>,
      visual:<div className="conv-row">{maps.map((m,i)=><figure key={i} className={i===kernel?'agree':''}><PixelGrid values={m} size={150} max={Math.max(1,...m)}/><figcaption>{KERNELS[i].name}</figcaption></figure>)}</div>,
      takeaway:'Cùng một bộ lọc dùng cho mọi vị trí, nên CNN nhận ra nét dù chữ số bị dịch đi. Đây là lý do CNN mạnh hơn hẳn việc so pixel trực tiếp như KNN.'},
    {label:'Máy nhầm ở đâu',title:'100 ảnh máy chưa từng thấy.',
      intro:<p>Bảng nhầm lẫn cho biết chữ số thật (hàng) bị đoán thành gì (cột). Bên dưới là những ảnh mạng một lớp đoán sai.</p>,
      controls:<><div className="stat-row"><Stat label="Đúng (test)" value={pct(final.testAcc)}/><Stat label="Số ảnh sai" value={mistakes.length}/></div>
        <Bars rows={D.map((d,i)=>({label:'Chữ số '+d,value:cm[i][i]/Math.max(1,cm[i].reduce((a,b)=>a+b,0))}))} max={1} format={v=>pct(v)}/></>,
      visual:<><Confusion labels={D} matrix={cm}/><div className="neighbor-grid">{mistakes.slice(0,10).map(m=><figure key={m.i}><PixelGrid values={pixels[m.i]} size={76}/><figcaption>thật <b>{labels[m.i]}</b> → đoán {m.p}</figcaption></figure>)}</div></>,
      takeaway:'Lỗi tập trung ở các cặp có nét giống nhau. Nhìn ảnh sai giúp em biết nên thêm dữ liệu gì hoặc dùng mô hình nào, đó là phân tích lỗi, kỹ năng quan trọng hơn cả việc tăng 1% accuracy.'},
  ];
  return <ProjectFrame id="digits" steps={steps} step={step} setStep={setStep} navigate={navigate}/>;
}
