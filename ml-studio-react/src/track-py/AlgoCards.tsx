// Thẻ thuật toán động: mỗi thẻ có tên, công thức, một cảnh SVG chạy vòng lặp và chỉ số tính trực tiếp.
// Mọi cảnh đều tính bằng chính thuật toán trên một bộ dữ liệu minh họa nhỏ (ghi nhãn "dữ liệu minh họa"),
// không phải hình vẽ tay. Hoạt ảnh dừng khi thẻ ra khỏi màn hình hoặc khi hệ điều hành bật giảm chuyển động.
import {useEffect,useMemo,useRef,useState} from 'react';
import type {ReactNode} from 'react';

const C={cyan:'#42d3e5',pink:'#f28db9',gold:'#ffd173',green:'#84e0a8',violet:'#a8a7ff',orange:'#ff9b6a',grid:'#24354f',text:'#b5c9df'};
const W=320,H=150;

function rng(seed:number){let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}}
function gauss(r:()=>number){const u=Math.max(r(),1e-9),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const clamp=(v:number,a:number,b:number)=>Math.max(a,Math.min(b,v));
const fmt=(v:number,d=2)=>v.toFixed(d);

/** Thời gian (giây) chạy khi thẻ đang hiện; đứng yên ở `still` khi giảm chuyển động. */
function useClock(still=3){
  const ref=useRef<HTMLDivElement>(null);
  const [t,setT]=useState(still);
  useEffect(()=>{
    const el=ref.current;if(!el)return;
    if(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)return;
    let visible=false,raf=0,last=0,acc=still;
    const loop=(now:number)=>{if(!visible)return;if(last){acc+=Math.min(.1,(now-last)/1000)}last=now;setT(acc);raf=requestAnimationFrame(loop)};
    const io=new IntersectionObserver(([e])=>{visible=e.isIntersecting;last=0;cancelAnimationFrame(raf);if(visible)raf=requestAnimationFrame(loop)},{threshold:.15});
    io.observe(el);
    return()=>{io.disconnect();cancelAnimationFrame(raf)};
  },[still]);
  return {ref,t};
}

type Scene=(t:number)=>{svg:ReactNode;badge:string;badge2?:string};
export interface AlgoCardSpec {id:string;title:string;formula:string;tag:string;color:string;caption:string;days:number[];scene:Scene}

// ───────── Dữ liệu minh họa dùng chung ─────────
const lin=(()=>{const r=rng(7);return Array.from({length:12},(_,i)=>{const x=.5+i*.45;return {x,y:1.2+.9*x+gauss(r)*.55}})})();
const sx=(x:number)=>24+x*52, sy=(y:number)=>H-14-y*16;
function twoBlobs(seed:number,n=18,sep=1.6){const r=rng(seed);return Array.from({length:2*n},(_,i)=>{const c=i<n?0:1;return {x:(c?sep:-sep)+gauss(r)*.8,y:(c?.6:-.6)+gauss(r)*.8,c}})}
const blobs=twoBlobs(11);
const bx=(x:number)=>W/2+x*42, by=(y:number)=>H/2-y*30;

// ───────── Các cảnh ─────────
const linearScene:Scene=t=>{
  // Gradient descent thật trên MSE, khởi động lại mỗi 6 giây.
  const steps=Math.floor((t%6)/6*60);let w=0,b=4;const lr=.03,n=lin.length;
  for(let k=0;k<steps;k++){let gw=0,gb=0;for(const p of lin){const e=w*p.x+b-p.y;gw+=2*e*p.x/n;gb+=2*e/n}w-=lr*gw;b-=lr*gb}
  const mse=lin.reduce((s,p)=>s+(w*p.x+b-p.y)**2,0)/n;
  return {badge:`MSE ${fmt(mse)}`,badge2:`bước ${steps}`,svg:<>
    {lin.map((p,i)=><line key={'r'+i} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y)} y2={sy(w*p.x+b)} stroke={C.pink} strokeOpacity={.55}/>)}
    <line x1={sx(0)} x2={sx(5.6)} y1={sy(b)} y2={sy(w*5.6+b)} stroke={C.cyan} strokeWidth={2.5}/>
    {lin.map((p,i)=><circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="#eaf6ff"/>)}
  </>};
};
const gdScene:Scene=t=>{
  // L(w)=(w-3)^2 với ba learning rate: chậm, vừa, dao động.
  const k=Math.floor((t%5)/5*14);const rates=[[.08,C.green],[.35,C.cyan],[.92,C.pink]] as const;
  const px=(w:number)=>20+w*45, py=(L:number)=>H-18-L*11;
  const curve=Array.from({length:61},(_,i)=>{const w=-.4+i*.11;return `${i?'L':'M'}${px(w)},${py((w-3)**2)}`}).join('');
  return {badge:'L(w) = (w−3)²',badge2:`bước ${k}`,svg:<>
    <path d={curve} fill="none" stroke={C.grid} strokeWidth={2}/>
    {rates.map(([lr,col],j)=>{let w=0;const pts=[w];for(let s=0;s<k;s++){w-=lr*2*(w-3);pts.push(w)}
      return <g key={j}><polyline points={pts.map(v=>`${px(v)},${py((v-3)**2)}`).join(' ')} fill="none" stroke={col} strokeOpacity={.7}/>
        <circle cx={px(w)} cy={py((w-3)**2)} r={5} fill={col}/></g>})}
    <text x={W-8} y={14} textAnchor="end" fill={C.text} fontSize={10}>α = 0,08 · 0,35 · 0,92</text>
  </>};
};
const sgdScene:Scene=t=>{
  // Đường đồng mức elip; batch GD đi thẳng, SGD có nhiễu.
  const k=Math.floor((t%6)/6*40);const cx=W/2,cy=H/2;
  const f=(a:number,b:number)=>[a*.86,b*.6];
  let g=[-2.6,1.9],s=[-2.6,1.9];const r=rng(3);const gp=[g],sp=[s];
  for(let i=0;i<k;i++){g=f(g[0],g[1]);gp.push(g);s=[s[0]*.86+gauss(r)*.18,s[1]*.6+gauss(r)*.18];sp.push(s)}
  const P=(p:number[])=>`${cx+p[0]*48},${cy-p[1]*28}`;
  return {badge:`epoch ${k}`,badge2:'batch · SGD',svg:<>
    {[1,2,3,4].map(i=><ellipse key={i} cx={cx} cy={cy} rx={i*34} ry={i*14} fill="none" stroke={C.grid}/>)}
    <polyline points={gp.map(P).join(' ')} fill="none" stroke={C.cyan} strokeWidth={2}/>
    <polyline points={sp.map(P).join(' ')} fill="none" stroke={C.pink} strokeWidth={1.4}/>
    <circle cx={cx} cy={cy} r={3} fill={C.gold}/>
  </>};
};
function solve(A:number[][],b:number[]){const n=b.length,M=A.map((r,i)=>[...r,b[i]]);for(let i=0;i<n;i++){let p=i;for(let j=i+1;j<n;j++)if(Math.abs(M[j][i])>Math.abs(M[p][i]))p=j;[M[i],M[p]]=[M[p],M[i]];for(let j=i+1;j<n;j++){const f=M[j][i]/M[i][i];for(let k=i;k<=n;k++)M[j][k]-=f*M[i][k]}}const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=M[i][n];for(let k=i+1;k<n;k++)s-=M[i][k]*x[k];x[i]=s/M[i][i]}return x}
const polyData=(()=>{const r=rng(21);return Array.from({length:22},(_,i)=>{const x=-1+i*2/21;return {x,y:Math.sin(2.6*x)+gauss(r)*.22,val:i%2===1}})})();
const polyScene:Scene=t=>{
  const d=1+Math.floor((t%9)/9*11);const tr=polyData.filter(p=>!p.val),va=polyData.filter(p=>p.val);
  const feats=(x:number)=>Array.from({length:d+1},(_,j)=>x**j);
  const A=Array.from({length:d+1},(_,i)=>Array.from({length:d+1},(_,j)=>tr.reduce((s,p)=>s+feats(p.x)[i]*feats(p.x)[j],0)+(i===j?1e-6:0)));
  const w=solve(A,Array.from({length:d+1},(_,i)=>tr.reduce((s,p)=>s+feats(p.x)[i]*p.y,0)));
  const f=(x:number)=>feats(x).reduce((s,v,j)=>s+v*w[j],0);
  const err=(S:typeof tr)=>S.reduce((s,p)=>s+(f(p.x)-p.y)**2,0)/S.length;
  const X=(x:number)=>W/2+x*140,Y=(y:number)=>H/2-y*45;
  return {badge:`bậc ${d}`,badge2:`train ${fmt(err(tr))} · val ${fmt(err(va))}`,svg:<>
    <path d={Array.from({length:81},(_,i)=>{const x=-1+i/40;return `${i?'L':'M'}${X(x)},${Y(clamp(f(x),-1.6,1.6))}`}).join('')} fill="none" stroke={C.cyan} strokeWidth={2}/>
    {polyData.map((p,i)=><circle key={i} cx={X(p.x)} cy={Y(p.y)} r={3.5} fill={p.val?C.pink:'#eaf6ff'}/>)}
  </>};
};
const logisticScene:Scene=t=>{
  const pts=Array.from({length:16},(_,i)=>{const x=-3+i*.4;return {x,y:x+((i*7)%5-2)*.35>0?1:0}});
  const steps=Math.floor((t%6)/6*80);let w=0,b=0;
  for(let k=0;k<steps;k++){let gw=0,gb=0;for(const p of pts){const q=1/(1+Math.exp(-(w*p.x+b)));gw+=(q-p.y)*p.x;gb+=q-p.y}w-=.3*gw/pts.length;b-=.3*gb/pts.length}
  const probe=-3+((t*.8)%6);const pr=1/(1+Math.exp(-(w*probe+b)));
  const X=(x:number)=>W/2+x*46,Y=(p:number)=>H-18-p*(H-36);
  return {badge:`p = ${fmt(pr)}`,badge2:`w = ${fmt(w)}`,svg:<>
    <line x1={10} x2={W-10} y1={Y(.5)} y2={Y(.5)} stroke={C.grid} strokeDasharray="4 4"/>
    <path d={Array.from({length:61},(_,i)=>{const x=-3.2+i*.107;return `${i?'L':'M'}${X(x)},${Y(1/(1+Math.exp(-(w*x+b))))}`}).join('')} fill="none" stroke={C.pink} strokeWidth={2.5}/>
    {pts.map((p,i)=><circle key={i} cx={X(p.x)} cy={Y(p.y)} r={4} fill={p.y?C.pink:C.cyan}/>)}
    <line x1={X(probe)} x2={X(probe)} y1={Y(0)} y2={Y(pr)} stroke={C.gold}/><circle cx={X(probe)} cy={Y(pr)} r={5} fill={C.gold}/>
  </>};
};
const rocScene:Scene=t=>{
  const r=rng(5);const neg=Array.from({length:120},()=>gauss(r)),pos=Array.from({length:40},()=>1.6+gauss(r));
  const th=-2+((t*.5)%5);const tp=pos.filter(v=>v>=th).length,fp=neg.filter(v=>v>=th).length,fn=pos.length-tp;
  const prec=tp+fp?tp/(tp+fp):1,rec=tp/(tp+fn);
  const hist=(arr:number[])=>{const h=Array(24).fill(0);for(const v of arr){const i=clamp(Math.floor((v+3)/7*24),0,23);h[i]++}return h};
  const X=(i:number)=>14+i*12;
  return {badge:`P ${fmt(prec)} · R ${fmt(rec)}`,badge2:`ngưỡng ${fmt(th,1)}`,svg:<>
    {hist(neg).map((c,i)=><rect key={'n'+i} x={X(i)} y={H-16-c*3.2} width={10} height={c*3.2} fill={C.cyan} fillOpacity={.55}/>)}
    {hist(pos).map((c,i)=><rect key={'p'+i} x={X(i)} y={H-16-c*3.2} width={10} height={c*3.2} fill={C.pink} fillOpacity={.75}/>)}
    <line x1={14+(th+3)/7*288} x2={14+(th+3)/7*288} y1={10} y2={H-12} stroke={C.gold} strokeWidth={2}/>
    <text x={W-8} y={14} textAnchor="end" fill={C.text} fontSize={10}>âm · dương</text>
  </>};
};
const knnScene:Scene=t=>{
  const k=5,q={x:Math.cos(t*.5)*2.2,y:Math.sin(t*.7)*1.2};
  const near=[...blobs].map(p=>({...p,d:Math.hypot(p.x-q.x,p.y-q.y)})).sort((a,b)=>a.d-b.d).slice(0,k);
  const votes=[near.filter(p=>p.c===0).length,near.filter(p=>p.c===1).length];const rad=near[k-1].d;
  return {badge:`k = ${k}`,badge2:`phiếu ${votes[0]}:${votes[1]}`,svg:<>
    <ellipse cx={bx(q.x)} cy={by(q.y)} rx={rad*42} ry={rad*30} fill={C.gold} fillOpacity={.08} stroke={C.gold} strokeOpacity={.7}/>
    {near.map((p,i)=><line key={'l'+i} x1={bx(q.x)} y1={by(q.y)} x2={bx(p.x)} y2={by(p.y)} stroke={C.gold} strokeOpacity={.6}/>)}
    {blobs.map((p,i)=><circle key={i} cx={bx(p.x)} cy={by(p.y)} r={3.6} fill={p.c?C.pink:C.cyan}/>)}
    <rect x={bx(q.x)-5} y={by(q.y)-5} width={10} height={10} transform={`rotate(45 ${bx(q.x)} ${by(q.y)})`} fill={votes[1]>votes[0]?C.pink:C.cyan} stroke="#fff"/>
  </>};
};
const nbScene:Scene=t=>{
  // Ước lượng Gauss cho từng lớp, tô vùng theo hậu nghiệm (giả định độc lập: ma trận chéo).
  const stat=(c:number)=>{const s=blobs.filter(p=>p.c===c);const mx=s.reduce((a,p)=>a+p.x,0)/s.length,my=s.reduce((a,p)=>a+p.y,0)/s.length;
    return {mx,my,vx:s.reduce((a,p)=>a+(p.x-mx)**2,0)/s.length,vy:s.reduce((a,p)=>a+(p.y-my)**2,0)/s.length}};
  const S=[stat(0),stat(1)];const ll=(s:typeof S[0],x:number,y:number)=>-((x-s.mx)**2/s.vx+(y-s.my)**2/s.vy)/2-Math.log(Math.sqrt(s.vx*s.vy));
  const reveal=clamp((t%6)/3,0,1);const cells=[];
  for(let i=0;i<20;i++)for(let j=0;j<10;j++){const x=-3.6+i*.38,y=-2.3+j*.5;if((i+j)/29>reveal)continue;const c=ll(S[1],x,y)>ll(S[0],x,y);
    cells.push(<rect key={i+'-'+j} x={bx(x)-8} y={by(y)-7.5} width={16} height={15} fill={c?C.pink:C.cyan} fillOpacity={.13}/>)}
  return {badge:'p(x|y)·p(y)',badge2:'độc lập theo trục',svg:<>
    {cells}{S.map((s,i)=>[1,2].map(k=><ellipse key={i+'e'+k} cx={bx(s.mx)} cy={by(s.my)} rx={k*Math.sqrt(s.vx)*42} ry={k*Math.sqrt(s.vy)*30} fill="none" stroke={i?C.pink:C.cyan} strokeOpacity={.8/k}/>))}
    {blobs.map((p,i)=><circle key={i} cx={bx(p.x)} cy={by(p.y)} r={3.2} fill={p.c?C.pink:C.cyan}/>)}
  </>};
};
const svmScene:Scene=t=>{
  // Subgradient hinge loss + ||w||²/2 trên hai cụm; vẽ lề w·x+b = ±1.
  const pts=twoBlobs(4,14,1.9).map(p=>({...p,s:p.c?1:-1}));const steps=Math.floor((t%7)/7*160);
  let w=[0,0],b=0;const C0=1;
  for(let k=1;k<=steps;k++){const lr=.5/Math.sqrt(k);const g=[w[0],w[1]];let gb=0;
    for(const p of pts)if(p.s*(w[0]*p.x+w[1]*p.y+b)<1){g[0]-=C0*p.s*p.x/pts.length*8;g[1]-=C0*p.s*p.y/pts.length*8;gb-=C0*p.s/pts.length*8}
    w=[w[0]-lr*g[0],w[1]-lr*g[1]];b-=lr*gb}
  const nrm=Math.hypot(w[0],w[1])||1e-9;
  const lineAt=(off:number)=>{const xs=[-4,4];return xs.map(x=>[x,(off-b-w[0]*x)/(Math.abs(w[1])>1e-6?w[1]:1e-6)])};
  const L=(off:number,col:string,dash?:string)=>{const [[x1,y1],[x2,y2]]=lineAt(off);return <line x1={bx(x1)} y1={by(y1)} x2={bx(x2)} y2={by(y2)} stroke={col} strokeWidth={off?1.2:2.4} strokeDasharray={dash}/>};
  return {badge:`2/‖w‖ = ${fmt(2/nrm)}`,badge2:`bước ${steps}`,svg:<>
    {L(1,C.gold,'5 4')}{L(0,C.green)}{L(-1,C.gold,'5 4')}
    {pts.map((p,i)=>{const m=p.s*(w[0]*p.x+w[1]*p.y+b);return <circle key={i} cx={bx(p.x)} cy={by(p.y)} r={m<=1.02?5:3.4} fill={p.c?C.pink:C.cyan} stroke={m<=1.02?'#fff':'none'}/>})}
  </>};
};
const cvScene:Scene=t=>{
  const k=5,fold=Math.floor(t/1.2)%k;
  return {badge:`fold ${fold+1}/${k}`,badge2:`val = phần ${fold+1}`,svg:<>
    {Array.from({length:k},(_,r)=><g key={r}>{Array.from({length:k},(_,c)=><rect key={c} x={40+c*50} y={16+r*24} width={46} height={18} rx={3}
      fill={c===r?C.pink:C.cyan} fillOpacity={r===fold?(c===r?.95:.55):.15}/>)}<text x={30} y={29+r*24} textAnchor="end" fill={C.text} fontSize={10}>{r+1}</text></g>)}
    <text x={W-8} y={H-6} textAnchor="end" fill={C.text} fontSize={10}>xanh: train · hồng: validation</text>
  </>};
};
const treeScene:Scene=t=>{
  const pts=(()=>{const r=rng(9);return Array.from({length:40},()=>{const x=r()*6-3,y=r()*4-2;return {x,y,c:(x>-.4&&y>-.3)||(x>1.6)?1:0}})})();
  const depth=Math.floor((t%6)/2);const splits=[{v:true,at:-.4},{v:false,at:-.3},{v:true,at:1.6}].slice(0,depth+ (depth>0?0:0));
  return {badge:`độ sâu ${depth}`,badge2:depth?`tách x${splits.at(-1)!.v?1:2} ≤ ${splits.at(-1)!.at}`:'gốc',svg:<>
    {splits.map((s,i)=>s.v?<line key={i} x1={bx(s.at)} x2={bx(s.at)} y1={8} y2={H-8} stroke={C.violet} strokeWidth={2}/>:<line key={i} x1={bx(-.4)} x2={W-8} y1={by(s.at)} y2={by(s.at)} stroke={C.violet} strokeWidth={2}/>)}
    {pts.map((p,i)=><circle key={i} cx={bx(p.x)} cy={by(p.y)} r={3.4} fill={p.c?C.pink:C.cyan}/>)}
  </>};
};
const forestScene:Scene=t=>{
  const n=5,active=Math.floor(t/0.8)%(n+2);const votes=[1,1,0,1,1].slice(0,Math.min(active,n));
  return {badge:`cây ${Math.min(active,n)}/${n}`,badge2:`phiếu ${votes.filter(v=>v).length}:${votes.filter(v=>!v).length}`,svg:<>
    {Array.from({length:n},(_,i)=>{const x=34+i*62,on=i<active;return <g key={i} opacity={on?1:.3}>
      <rect x={x-26} y={14} width={52} height={44} rx={6} fill="none" stroke={C.cyan}/>
      <line x1={x} y1={24} x2={x-12} y2={46} stroke={C.cyan}/><line x1={x} y1={24} x2={x+12} y2={46} stroke={C.cyan}/>
      <circle cx={x} cy={24} r={3} fill={C.gold}/><circle cx={x-12} cy={46} r={3} fill={[1,1,0,1,1][i]?C.pink:C.cyan}/><circle cx={x+12} cy={46} r={3} fill={C.cyan}/>
      {on&&<line x1={x} y1={60} x2={W/2} y2={104} stroke={[1,1,0,1,1][i]?C.pink:C.cyan} strokeOpacity={.7}/>}</g>})}
    <rect x={W/2-46} y={104} width={92} height={22} rx={5} fill="#0b1528" stroke={C.gold}/>
    <text x={W/2} y={119} textAnchor="middle" fill={C.gold} fontSize={11}>bỏ phiếu đa số</text>
  </>};
};
const boostScene:Scene=t=>{
  const xs=Array.from({length:30},(_,i)=>i/29),ys=xs.map(x=>Math.sin(6*x)*.8+x);
  const m=Math.floor((t%8)/8*8);let F=xs.map(()=>ys.reduce((a,b)=>a+b,0)/ys.length);const eta=.6;
  for(let s=0;s<m;s++){const r=ys.map((y,i)=>y-F[i]);let best={e:Infinity,cut:0,l:0,rr:0};
    for(let c=1;c<29;c++){const L=r.slice(0,c),R=r.slice(c);const ml=L.reduce((a,b)=>a+b,0)/L.length,mr=R.reduce((a,b)=>a+b,0)/R.length;
      const e=L.reduce((a,v)=>a+(v-ml)**2,0)+R.reduce((a,v)=>a+(v-mr)**2,0);if(e<best.e)best={e,cut:c,l:ml,rr:mr}}
    F=F.map((f,i)=>f+eta*(i<best.cut?best.l:best.rr))}
  const mse=ys.reduce((a,y,i)=>a+(y-F[i])**2,0)/ys.length;const X=(x:number)=>20+x*280,Y=(y:number)=>H/2+20-y*38;
  return {badge:`m = ${m}`,badge2:`MSE ${fmt(mse,3)}`,svg:<>
    <polyline points={xs.map((x,i)=>`${X(x)},${Y(F[i])}`).join(' ')} fill="none" stroke={C.cyan} strokeWidth={2}/>
    {xs.map((x,i)=><circle key={i} cx={X(x)} cy={Y(ys[i])} r={3} fill={C.gold}/>)}
  </>};
};
const kmeansScene:Scene=t=>{
  const pts=(()=>{const r=rng(2);const cs=[[-2,1],[2,1],[0,-1.2]];return Array.from({length:45},(_,i)=>{const c=cs[i%3];return {x:c[0]+gauss(r)*.55,y:c[1]+gauss(r)*.45}})})();
  const step=Math.floor((t%7)/7*7);let mu=[[-3,-1.5],[-2.6,-1.2],[-3.2,-.8]];let lab=pts.map(()=>0);
  for(let s=0;s<step;s++){lab=pts.map(p=>{let b=0,bd=Infinity;mu.forEach((m,j)=>{const d=(p.x-m[0])**2+(p.y-m[1])**2;if(d<bd){bd=d;b=j}});return b});
    mu=mu.map((m,j)=>{const g=pts.filter((_,i)=>lab[i]===j);return g.length?[g.reduce((a,p)=>a+p.x,0)/g.length,g.reduce((a,p)=>a+p.y,0)/g.length]:m})}
  const cols=[C.cyan,C.gold,C.pink];const inertia=pts.reduce((a,p,i)=>a+(p.x-mu[lab[i]][0])**2+(p.y-mu[lab[i]][1])**2,0);
  return {badge:`K = 3 · bước ${step}`,badge2:`inertia ${fmt(inertia,1)}`,svg:<>
    {pts.map((p,i)=><circle key={i} cx={bx(p.x)} cy={by(p.y)} r={3.2} fill={step?cols[lab[i]]:'#8aa0b8'}/>)}
    {mu.map((m,j)=><rect key={j} x={bx(m[0])-6} y={by(m[1])-6} width={12} height={12} transform={`rotate(45 ${bx(m[0])} ${by(m[1])})`} fill="none" stroke={cols[j]} strokeWidth={2.4}/>)}
  </>};
};
const pcaScene:Scene=t=>{
  const r=rng(13);const pts=Array.from({length:40},()=>{const a=gauss(r)*1.6,b=gauss(r)*.4;return {x:a*.85-b*.5,y:a*.5+b*.85}});
  const ang=(t*.6)%Math.PI;const u=[Math.cos(ang),Math.sin(ang)];
  const proj=pts.map(p=>p.x*u[0]+p.y*u[1]);const total=pts.reduce((a,p)=>a+p.x**2+p.y**2,0);const varFrac=proj.reduce((a,v)=>a+v*v,0)/total;
  return {badge:`phương sai giữ lại ${Math.round(varFrac*100)}%`,badge2:`góc ${Math.round(ang*180/Math.PI)}°`,svg:<>
    <line x1={bx(-4*u[0])} y1={by(-4*u[1])} x2={bx(4*u[0])} y2={by(4*u[1])} stroke={C.gold} strokeWidth={2}/>
    {pts.map((p,i)=><line key={'p'+i} x1={bx(p.x)} y1={by(p.y)} x2={bx(proj[i]*u[0])} y2={by(proj[i]*u[1])} stroke={C.pink} strokeOpacity={.35}/>)}
    {pts.map((p,i)=><circle key={i} cx={bx(p.x)} cy={by(p.y)} r={3} fill={C.cyan}/>)}
  </>};
};
const mlpScene:Scene=t=>{
  const layers=[3,4,4,2];const X=(l:number)=>40+l*80,Y=(l:number,i:number)=>H/2+(i-(layers[l]-1)/2)*28;
  const phase=(t%4)/4*(layers.length-1);const r=rng(17);
  const edges:ReactNode[]=[];
  for(let l=0;l<layers.length-1;l++)for(let i=0;i<layers[l];i++)for(let j=0;j<layers[l+1];j++){const wv=gauss(r);
    const lit=phase>l&&phase<l+1.4;edges.push(<line key={`${l}${i}${j}`} x1={X(l)} y1={Y(l,i)} x2={X(l+1)} y2={Y(l+1,j)} stroke={wv>0?C.cyan:C.pink} strokeOpacity={lit?.85:.18} strokeWidth={Math.abs(wv)+.3}/>)}
  const z=[1.3,-.2],p=Math.exp(z[0])/(Math.exp(z[0])+Math.exp(z[1]));
  return {badge:`p(c₀) = ${fmt(p)}`,badge2:'forward',svg:<>
    {edges}{layers.map((n,l)=>Array.from({length:n},(_,i)=><circle key={l+'-'+i} cx={X(l)} cy={Y(l,i)} r={7} fill="#0b1528" stroke={phase>=l-.2?C.cyan:C.grid} strokeWidth={2}/>))}
  </>};
};
const cnnScene:Scene=t=>{
  const img=[[0,0,1,1,0,0],[0,1,1,1,1,0],[1,1,0,0,1,1],[1,1,0,0,1,1],[0,1,1,1,1,0],[0,0,1,1,0,0]];
  const K=[[1,0,-1],[1,0,-1],[1,0,-1]];const pos=Math.floor(t*2)%16,pr=Math.floor(pos/4),pc=pos%4;
  const conv=(r:number,c:number)=>K.reduce((s,row,i)=>s+row.reduce((a,k,j)=>a+k*img[r+i][c+j],0),0);
  const cell=16,ox=18,oy=24,mx=210;
  return {badge:`S[${pr},${pc}] = ${conv(pr,pc)}`,badge2:'kernel 3×3 · stride 1',svg:<>
    {img.map((row,r)=>row.map((v,c)=><rect key={r+'-'+c} x={ox+c*cell} y={oy+r*cell} width={cell-2} height={cell-2} rx={2} fill={v?'#d9e9ff':'#1d2c44'}/>))}
    <rect x={ox+pc*cell-2} y={oy+pr*cell-2} width={3*cell+2} height={3*cell+2} fill="none" stroke={C.pink} strokeWidth={2.4} rx={3}/>
    <text x={150} y={H/2+4} textAnchor="middle" fill={C.pink} fontSize={12}>∗K</text>
    {Array.from({length:16},(_,i)=>{const r=Math.floor(i/4),c=i%4,v=conv(r,c),done=i<=pos;
      return <rect key={i} x={mx+c*(cell+4)} y={oy+r*(cell+4)} width={cell+1} height={cell+1} rx={3} fill={v>0?C.pink:v<0?C.cyan:'#2a3954'} fillOpacity={done?.25+Math.min(Math.abs(v),3)/4:.12} stroke={i===pos?'#fff':'none'}/>})}
  </>};
};
const attnScene:Scene=t=>{
  const toks=['Mèo','ngồi','trên','thảm'];const Q=[[1,0],[0,1],[.5,.5],[1,.2]],Kt=[[1,.1],[.2,1],[.4,.6],[.9,.3]];
  const qi=Math.floor(t/1.4)%4;const sc=Kt.map(k=>(Q[qi][0]*k[0]+Q[qi][1]*k[1])/Math.SQRT2*3);const mx=Math.max(...sc);const e=sc.map(s=>Math.exp(s-mx));const sum=e.reduce((a,b)=>a+b,0);const a=e.map(v=>v/sum);
  const X=(i:number)=>30+i*58;
  return {badge:`Q = ${toks[qi]}`,badge2:`Σ trọng số = ${fmt(a.reduce((x,y)=>x+y,0))}`,svg:<>
    {toks.map((_,j)=>j!==qi&&<path key={j} d={`M${X(qi)},96 Q${(X(qi)+X(j))/2},${30-a[j]*20} ${X(j)},96`} fill="none" stroke={C.violet} strokeOpacity={.25+a[j]} strokeWidth={1+a[j]*6}/>)}
    {toks.map((w,i)=><g key={i}><circle cx={X(i)} cy={104} r={11} fill={i===qi?C.violet:'#1d2c44'} stroke={C.violet}/><text x={X(i)} y={130} textAnchor="middle" fill={C.text} fontSize={10}>{w}</text></g>)}
    {a.map((v,j)=><rect key={j} x={250+(j%2)*28} y={18+Math.floor(j/2)*28} width={24} height={24} rx={3} fill={C.violet} fillOpacity={.15+v*.85}/>)}
    <text x={278} y={84} textAnchor="middle" fill={C.text} fontSize={9}>softmax</text>
  </>};
};

export const ALGO_CARDS:AlgoCardSpec[]=[
  {id:'linear',title:'Hồi quy tuyến tính',formula:'ŷ = wx + b',tag:'HỒI QUY',color:C.cyan,caption:'Kéo đường thẳng để tổng bình phương phần dư nhỏ nhất.',days:[3,7],scene:linearScene},
  {id:'gd',title:'Gradient descent',formula:'w ← w − α∇L(w)',tag:'TỐI ƯU',color:C.green,caption:'α nhỏ thì chậm, vừa thì hội tụ, lớn thì dao động.',days:[4],scene:gdScene},
  {id:'sgd',title:'SGD và mini-batch',formula:'∇L ≈ (1/|B|) Σ∇ℓᵢ',tag:'NGẪU NHIÊN',color:C.pink,caption:'Mỗi bước dùng một phần dữ liệu: rẻ hơn nhưng nhiễu hơn.',days:[5],scene:sgdScene},
  {id:'poly',title:'Bias–variance',formula:'E = bias² + var + σ²',tag:'ĐỘ PHỨC TẠP',color:C.gold,caption:'Bậc tăng thì train giảm mãi, validation thì không.',days:[6],scene:polyScene},
  {id:'logistic',title:'Hồi quy logistic',formula:'p = 1 / (1 + e⁻ᶻ)',tag:'PHÂN LOẠI',color:C.pink,caption:'Sigmoid biến điểm z thành xác suất trong (0, 1).',days:[8],scene:logisticScene},
  {id:'roc',title:'Ngưỡng và ROC/PR',formula:'P = TP/(TP+FP) · R = TP/(TP+FN)',tag:'ĐÁNH GIÁ',color:C.gold,caption:'Dời ngưỡng là đổi giữa precision và recall, mô hình không đổi.',days:[9,14],scene:rocScene},
  {id:'knn',title:'K láng giềng gần nhất',formula:'ŷ = mode(yᵢ), i ∈ Nₖ(x)',tag:'BỎ PHIẾU',color:C.orange,caption:'Đa số trong k điểm gần nhất quyết định; phải chuẩn hóa thang đo.',days:[10],scene:knnScene},
  {id:'nb',title:'Naive Bayes / GDA',formula:'ŷ = argmax p(x|y)p(y)',tag:'MÔ HÌNH SINH',color:C.violet,caption:'Mô hình hóa mỗi lớp bằng một phân phối rồi so hậu nghiệm.',days:[11],scene:nbScene},
  {id:'svm',title:'Máy vector hỗ trợ',formula:'w·x + b = ±1',tag:'LỀ CỰC ĐẠI',color:C.green,caption:'Chọn ranh giới có lề rộng nhất; chỉ vector hỗ trợ quyết định.',days:[12],scene:svmScene},
  {id:'cv',title:'Cross-validation',formula:'CV = (1/k) Σ điểm(foldᵢ)',tag:'CHỌN MÔ HÌNH',color:C.cyan,caption:'Mỗi fold lần lượt làm validation; mọi biến đổi fit trong fold.',days:[13,14],scene:cvScene},
  {id:'tree',title:'Cây quyết định',formula:'tách: xⱼ ≤ t (ΔGini)',tag:'TÁCH NHÁNH',color:C.violet,caption:'Tách dần cho lá thuần hơn; cây quá sâu sẽ quá khớp.',days:[15],scene:treeScene},
  {id:'forest',title:'Random forest',formula:'ŷ = mode(T₁(x)…T_B(x))',tag:'BAGGING',color:C.orange,caption:'Nhiều cây trên mẫu bootstrap bỏ phiếu, giảm phương sai.',days:[16],scene:forestScene},
  {id:'boost',title:'Gradient boosting',formula:'Fₘ = Fₘ₋₁ + ηhₘ(x)',tag:'BOOSTING',color:C.cyan,caption:'Mỗi cây gốc nhỏ sửa phần dư của tổng các cây trước.',days:[17],scene:boostScene},
  {id:'kmeans',title:'K-means',formula:'min Σ ‖xᵢ − μₖ‖²',tag:'PHÂN CỤM',color:C.pink,caption:'Gán điểm về tâm gần nhất, dời tâm về trung bình, lặp lại.',days:[18],scene:kmeansScene},
  {id:'pca',title:'PCA',formula:'u₁ = argmax Var(Xu)',tag:'GIẢM CHIỀU',color:C.gold,caption:'Trục chính là hướng giữ được nhiều phương sai nhất.',days:[19,21],scene:pcaScene},
  {id:'mlp',title:'MLP và backprop',formula:'y = σ(W₂ σ(W₁x + b₁) + b₂)',tag:'MẠNG NƠ-RON',color:C.cyan,caption:'Tín hiệu đi xuôi qua các tầng; gradient đi ngược để cập nhật.',days:[22,23,24,25],scene:mlpScene},
  {id:'cnn',title:'Tích chập (CNN)',formula:'S = (I ∗ K) + b',tag:'LƯỚI KHÔNG GIAN',color:C.pink,caption:'Một bộ lọc trượt khắp ảnh để phát hiện mẫu cục bộ.',days:[26,28],scene:cnnScene},
  {id:'attn',title:'Self-attention',formula:'softmax(QKᵀ/√d)V',tag:'TRANSFORMER',color:C.violet,caption:'Mỗi token chấm điểm mọi token khác rồi lấy tổng có trọng số.',days:[27],scene:attnScene},
];

export function AlgoCard({spec,onOpen,status}:{spec:AlgoCardSpec;onOpen?:()=>void;status?:string}){
  const {ref,t}=useClock(2.6);
  const {svg,badge,badge2}=useMemo(()=>spec.scene(t),[spec,t]);
  const body=<>
    <header><h3 style={{color:spec.color}}>{spec.title}</h3><span className="algo-card-tag">{spec.tag}</span></header>
    <code className="algo-card-formula">{spec.formula}</code>
    <div className="algo-card-stage">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${spec.title}: ${badge}`}>{svg}</svg>
      <span className="algo-card-badge" style={{borderColor:spec.color,color:spec.color}}>{badge}</span>
      {badge2&&<span className="algo-card-badge right">{badge2}</span>}
    </div>
    <p>{spec.caption}</p>
    <small>{status??`Ngày ${spec.days.join(', ')}`} · dữ liệu minh họa</small>
  </>;
  return <div ref={ref} className="algo-card" style={{['--accent' as string]:spec.color}}>
    {onOpen?<button type="button" onClick={onOpen} aria-label={`Mở ${spec.title}`}>{body}</button>:body}
  </div>;
}

export function AlgoAtlas({openDay,available}:{openDay:(day:number)=>void;available:(day:number)=>boolean}){
  return <section className="algo-atlas" aria-label="Bản đồ thuật toán">
    <header><span>BẢN ĐỒ THUẬT TOÁN</span><h2>18 ý tưởng cốt lõi · cách mỗi thuật toán học</h2>
      <p>Mỗi thẻ chạy chính thuật toán đó trên dữ liệu minh họa nhỏ. Bấm vào thẻ để mở buổi học tương ứng.</p></header>
    <div className="algo-atlas-grid">{ALGO_CARDS.map(c=>{const day=c.days.find(available);
      return <AlgoCard key={c.id} spec={c} onOpen={day?()=>openDay(day):undefined} status={day?`Mở ngày ${day}`:`Ngày ${c.days[0]} · đang soạn`}/>})}</div>
  </section>;
}

export const cardsForDay=(day:number)=>ALGO_CARDS.filter(c=>c.days.includes(day));
