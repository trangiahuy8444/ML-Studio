import {useMemo,useState} from 'react';
import type {Route} from '../types';
import {Bars,LineChart,PALETTE,Plot,PlayerBar,ProjectFrame,Segmented,Stat,fmt,gauss,rng,usePlayer} from './ui';
import type {StepDef} from './ui';

const FEATURES=['Lượt mua / tháng','Chi tiêu TB (nghìn đ)','Tuổi','Phút xem web / lượt'];
// Bốn kiểu khách giả lập; thuật toán không được biết nhãn này.
const SEGMENTS=[[8,90,21,45],[4,650,38,10],[10,950,46,25],[1.5,160,31,6]];
const SPREAD=[1.4,70,4,8];
const customers:number[][]=(()=>{const r=rng(21),out:number[][]=[];for(let i=0;i<90;i++){const s=SEGMENTS[i%4];out.push(s.map((m,j)=>Math.max(j===0?.2:j===2?16:1,m+gauss(r)*SPREAD[j])))}return out})();
const mean=FEATURES.map((_,j)=>customers.reduce((s,c)=>s+c[j],0)/customers.length);
const sd=FEATURES.map((_,j)=>Math.sqrt(customers.reduce((s,c)=>s+(c[j]-mean[j])**2,0)/customers.length));
const Z=customers.map(c=>c.map((v,j)=>(v-mean[j])/sd[j]));
const d2=(a:number[],b:number[])=>a.reduce((s,v,j)=>s+(v-b[j])**2,0);

interface Frame {centers:number[][];labels:number[];phase:'assign'|'move';inertia:number}
function kmeans(points:number[][],init:number[][],iters=10):Frame[]{
  let centers=init.map(c=>[...c]);const frames:Frame[]=[];
  for(let it=0;it<iters;it++){
    const labels=points.map(p=>{let b=0;centers.forEach((c,k)=>{if(d2(p,c)<d2(p,centers[b]))b=k});return b});
    const inertia=points.reduce((s,p,i)=>s+d2(p,centers[labels[i]]),0);
    frames.push({centers:centers.map(c=>[...c]),labels,phase:'assign',inertia});
    const next=centers.map((c,k)=>{const m=points.filter((_,i)=>labels[i]===k);return m.length?c.map((_,j)=>m.reduce((s,p)=>s+p[j],0)/m.length):c});
    const moved=next.some((c,k)=>d2(c,centers[k])>1e-10);centers=next;
    frames.push({centers:centers.map(c=>[...c]),labels,phase:'move',inertia:points.reduce((s,p,i)=>s+d2(p,centers[labels[i]]),0)});
    if(!moved)break;
  }
  return frames;
}
function bestKmeans(points:number[][],k:number){
  let best:Frame|null=null;
  for(let s=0;s<6;s++){const r=rng(100+s+k*7),init=Array.from({length:k},()=>points[Math.floor(r()*points.length)]);const f=kmeans(points,init,30).at(-1)!;if(!best||f.inertia<best.inertia)best=f}
  return best!;
}
/** Hai thành phần chính bằng phép lặp lũy thừa trên ma trận hiệp phương sai. */
function pca(points:number[][]){
  const n=points.length,dim=points[0].length,C=Array.from({length:dim},(_,a)=>Array.from({length:dim},(_,b)=>points.reduce((s,p)=>s+p[a]*p[b],0)/n));
  const comps:number[][]=[],vals:number[]=[];let M=C.map(r=>[...r]);
  for(let c=0;c<2;c++){let v=Array.from({length:dim},(_,i)=>1+i*.1);for(let it=0;it<200;it++){const w=M.map(r=>r.reduce((s,x,j)=>s+x*v[j],0));const norm=Math.hypot(...w);v=w.map(x=>x/norm)}
    const lambda=M.map(r=>r.reduce((s,x,j)=>s+x*v[j],0)).reduce((s,x,i)=>s+x*v[i],0);comps.push(v);vals.push(lambda);M=M.map((r,a)=>r.map((x,b)=>x-lambda*v[a]*v[b]))}
  const total=C.reduce((s,r,i)=>s+r[i],0);
  return {comps,ratio:vals.map(v=>v/total),proj:points.map(p=>comps.map(v=>v.reduce((s,x,j)=>s+x*p[j],0)))};
}
const toZ=(x:number,y:number)=>[(x-mean[0])/sd[0],(y-mean[1])/sd[1]];
const fromZ=(c:number[])=>[c[0]*sd[0]+mean[0],c[1]*sd[1]+mean[1]];
const XR:[number,number]=[-1,14],YR:[number,number]=[-50,1200];

export function CustomersProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [step,setStep]=useState(0);
  const [k,setK]=useState(4);
  const [mine,setMine]=useState<number[][]>([]);
  const Z2=useMemo(()=>Z.map(z=>[z[0],z[1]]),[]);
  const init=useMemo(()=>{const r=rng(9);return Array.from({length:k},(_,i)=>mine[i]??Z2[Math.floor(r()*Z2.length)])},[k,mine,Z2]);
  const frames=useMemo(()=>kmeans(Z2,init,12),[Z2,init]);
  const player=usePlayer(frames.length,500);
  const elbow=useMemo(()=>[1,2,3,4,5,6,7,8].map(kk=>bestKmeans(Z,kk).inertia),[]);
  const full=useMemo(()=>bestKmeans(Z,k),[k]);
  const P=useMemo(()=>pca(Z),[]);
  const [names,setNames]=useState<string[]>([]);

  const mineLabels=Z2.map(p=>{if(!mine.length)return -1;let b=0;mine.forEach((c,i)=>{if(d2(p,c)<d2(p,mine[b]))b=i});return b});
  const mineInertia=mine.length?Z2.reduce((s,p,i)=>s+d2(p,mine[mineLabels[i]]),0):NaN;
  const fr=frames[player.frame];
  const profile=Array.from({length:k},(_,c)=>{const m=customers.filter((_,i)=>full.labels[i]===c);return {n:m.length,avg:FEATURES.map((_,j)=>m.length?m.reduce((s,p)=>s+p[j],0)/m.length:0)}});

  const scatter=(labels:number[],centers?:number[][],onPick?:(x:number,y:number)=>void)=><Plot x={XR} y={YR} xLabel={FEATURES[0]} yLabel={FEATURES[1]} onPick={onPick}>{(sx,sy)=><>
    {customers.map((c,i)=><circle key={i} cx={sx(c[0])} cy={sy(c[1])} r={4.2} fill={labels[i]>=0?PALETTE[labels[i]]:'var(--text-3)'} fillOpacity={.8}/>)}
    {centers?.map((c,i)=>{const [x,y]=fromZ(c);return <g key={i}><circle cx={sx(x)} cy={sy(y)} r={11} fill="none" stroke={PALETTE[i]} strokeWidth={3}/><path d={`M${sx(x)-6} ${sy(y)}h12M${sx(x)} ${sy(y)-6}v12`} stroke="#fff" strokeWidth={2}/></g>})}
  </>}</Plot>;
  const kPick=<Segmented label="Số nhóm" options={[[2,'2'],[3,'3'],[4,'4'],[5,'5'],[6,'6']]} value={k} onChange={v=>{setK(v);setMine(m=>m.slice(0,v));player.reset()}}/>;

  const steps:StepDef[]=[
    {label:'Em tự chia nhóm',title:'Không có nhãn. Em thấy mấy nhóm khách?',
      intro:<p>90 khách hàng, mỗi chấm là một người. Chọn số nhóm rồi <b>bấm lên biểu đồ</b> để đặt tâm từng nhóm. Mỗi khách thuộc về tâm gần nhất.</p>,
      controls:<>{kPick}<div className="stat-row"><Stat label="Tâm đã đặt" value={`${mine.length}/${k}`}/><Stat label="Tổng bình phương khoảng cách" value={Number.isNaN(mineInertia)?'—':fmt(mineInertia,1)}/></div>
        <button className="btn-outline" onClick={()=>setMine([])}>Xóa các tâm</button><p className="small-note">Khoảng cách được tính sau khi chuẩn hóa hai trục, nếu không thì trục tiền (hàng trăm) sẽ lấn át trục số lượt (vài lượt).</p></>,
      visual:<>{scatter(mineLabels,mine,(x,y)=>setMine(m=>m.length>=k?[...m.slice(1),toZ(x,y)]:[...m,toZ(x,y)]))}<p className="chart-caption">Bấm để đặt tâm. Đặt quá số nhóm thì tâm cũ nhất bị thay.</p></>,
      takeaway:'Tổng bình phương khoảng cách (inertia) càng nhỏ thì khách càng sát tâm nhóm mình. Đó chính là con số K-means sẽ cố làm nhỏ.'},
    {label:'K-means từng bước',title:'Gán vào tâm gần nhất, rồi dời tâm về giữa.',
      intro:<p>K-means lặp hai việc: <b>gán</b> mỗi khách cho tâm gần nhất, rồi <b>dời</b> mỗi tâm về trung bình các khách của nó. Nếu em đã đặt tâm ở bước 1, máy bắt đầu từ đó.</p>,
      controls:<>{kPick}<PlayerBar player={player} count={frames.length} label="Bước"/>
        <div className="stat-row"><Stat label="Đang làm" value={fr.phase==='assign'?'Gán khách':'Dời tâm'}/><Stat label="Inertia" value={fmt(fr.inertia,1)}/></div></>,
      visual:<>{scatter(fr.labels,fr.centers)}<LineChart height={140} xLabel="Bước" yLabel="Inertia" series={[{name:'Inertia',values:frames.slice(0,player.frame+1).map(f=>f.inertia),color:'#ffd173'}]}/></>,
      takeaway:'Inertia không bao giờ tăng qua các bước, nên K-means luôn dừng. Nhưng điểm dừng phụ thuộc chỗ đặt tâm ban đầu; thử đặt tâm tệ ở bước 1 rồi quay lại xem.'},
    {label:'Chọn số nhóm',title:'Bao nhiêu nhóm là đủ?',
      intro:<p>Thêm nhóm luôn làm inertia nhỏ đi (k = 90 thì inertia bằng 0!). Ta tìm chỗ <b>"khuỷu tay"</b>: sau đó thêm nhóm không còn giảm được bao nhiêu.</p>,
      controls:<>{kPick}<Bars rows={elbow.map((v,i)=>({label:`k = ${i+1}`,value:v,color:i+1===k?'var(--gold)':undefined}))} format={v=>fmt(v,0)}/></>,
      visual:<LineChart height={300} xLabel="k − 1" yLabel="Inertia (4 đặc trưng)" series={[{name:'Inertia theo k',values:elbow,color:PALETTE[0]}]}/>,
      takeaway:'Đường cong gãy rõ ở k = 4, khớp với bốn kiểu khách mình giả lập. Với dữ liệu thật, khuỷu tay thường mờ hơn và em cần thêm hiểu biết về bài toán để chọn.'},
    {label:'Nhìn 4 chiều',title:'PCA ép 4 đặc trưng xuống mặt phẳng.',
      intro:<p>Mỗi khách có 4 con số, mắt người không vẽ được 4 chiều. PCA tìm hai hướng giữ được nhiều độ biến thiên nhất. Màu là nhóm K-means tính trên đủ 4 đặc trưng.</p>,
      controls:<>{kPick}<div className="stat-row"><Stat label="PC1 giữ" value={fmt(P.ratio[0]*100,0)+'%'}/><Stat label="PC2 giữ" value={fmt(P.ratio[1]*100,0)+'%'}/></div>
        <p className="small-note">Mỗi đặc trưng góp vào PC1 bao nhiêu:</p><Bars rows={FEATURES.map((f,j)=>({label:f,value:P.comps[0][j],color:P.comps[0][j]<0?'var(--pink)':undefined}))} max={1}/></>,
      visual:<Plot x={[-3.4,3.4]} y={[-3,3]} xLabel="PC1" yLabel="PC2">{(sx,sy)=><>{P.proj.map((p,i)=><circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={4.4} fill={PALETTE[full.labels[i]]} fillOpacity={.85}/>)}</>}</Plot>,
      takeaway:'Hai thành phần chính giữ phần lớn thông tin, nên các nhóm vẫn tách rõ trên mặt phẳng. Trọng số trên PC1 cho biết trục mới này "nói về" điều gì, ví dụ chi tiêu và tuổi.'},
    {label:'Đặt tên nhóm',title:'Máy chia nhóm, con người đặt tên.',
      intro:<p>K-means không biết nhóm nghĩa là gì. Nhìn chân dung trung bình từng nhóm rồi đặt tên và nghĩ một hành động cho cửa hàng.</p>,
      controls:<>{kPick}{profile.map((p,c)=><label key={c} className="name-field"><span style={{background:PALETTE[c]}}/><input value={names[c]??''} placeholder={`Tên nhóm ${c+1}…`} onChange={e=>setNames(n=>{const x=[...n];x[c]=e.target.value;return x})}/></label>)}</>,
      visual:<div className="table-wrap"><table><thead><tr><th>Nhóm</th><th>Số khách</th>{FEATURES.map(f=><th key={f}>{f}</th>)}</tr></thead>
        <tbody>{profile.map((p,c)=><tr key={c}><td><span className="swatch" style={{background:PALETTE[c]}}/>{names[c]||`Nhóm ${c+1}`}</td><td>{p.n}</td>{p.avg.map((v,j)=><td key={j}>{fmt(v,j===0?1:0)}</td>)}</tr>)}</tbody></table>
        <p className="chart-caption">Gợi ý: nhóm chi tiêu cao nhưng ít xem web có thể là "gia đình mua cuối tuần"; nhóm trẻ xem web lâu, chi ít có thể là "sinh viên săn khuyến mãi".</p></div>,
      takeaway:'Phân cụm là công cụ khám phá: giá trị thật nằm ở bước con người diễn giải và kiểm chứng nhóm, ví dụ thử gửi ưu đãi khác nhau cho từng nhóm và đo kết quả.'},
  ];
  return <ProjectFrame id="customers" steps={steps} step={step} setStep={setStep} navigate={navigate}/>;
}
