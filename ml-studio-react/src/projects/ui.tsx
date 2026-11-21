import {useEffect,useRef,useState} from 'react';
import type {MouseEvent,ReactNode} from 'react';
import {ArrowLeft,ArrowRight,CheckCircle,Lightbulb} from 'lucide-react';
import type {Route} from '../types';
import {CHAPTERS,PROJECTS} from './chapters';
import type {ProjectId} from './ids';

/* ---------- Tiến độ dự án (lưu trên trình duyệt) ---------- */
export interface ProjectState {visited:number[];done:boolean;note:string}
export type ProjectStore=Record<string,ProjectState>;
const KEY='mlStudio-projects-v1';
export const blankProject=():ProjectState=>({visited:[],done:false,note:''});
export function loadProjects():ProjectStore{
  try{
    const raw=JSON.parse(localStorage.getItem(KEY)??'{}') as Record<string,Partial<ProjectState>>;
    const out:ProjectStore={};
    for(const [id,v] of Object.entries(raw??{}))if(v&&typeof v==='object')out[id]={visited:Array.isArray(v.visited)?v.visited.filter(n=>Number.isInteger(n)&&n>=0&&n<10):[],done:v.done===true,note:typeof v.note==='string'?v.note.slice(0,5000):''};
    return out;
  }catch{return {}}
}
export function saveProject(id:string,state:ProjectState){
  try{const all=loadProjects();all[id]=state;localStorage.setItem(KEY,JSON.stringify(all))}catch{/* bộ nhớ trình duyệt không khả dụng */}
}

/* ---------- Số ngẫu nhiên có hạt giống: mô phỏng chạy lại cho cùng kết quả ---------- */
export function rng(seed:number){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
export function shuffled<T>(items:readonly T[],seed:number):T[]{const r=rng(seed),a=[...items];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
export const gauss=(r:()=>number)=>Math.sqrt(-2*Math.log(r()||1e-9))*Math.cos(2*Math.PI*r());
export const pct=(v:number,d=0)=>(v*100).toFixed(d)+'%';
export const fmt=(v:number,d=2)=>v.toLocaleString('vi-VN',{maximumFractionDigits:d,minimumFractionDigits:d});

export const PALETTE=['#42d3e5','#f28db9','#ffd173','#84e0a8','#a8a7ff','#ff9b6a','#7fd1ff','#e3b6ff','#c3e88d','#ffb3a7'];

/* ---------- Khung 5 bước dùng chung ---------- */
export interface StepDef {label:string;title:string;intro:ReactNode;controls?:ReactNode;visual:ReactNode;takeaway:string;below?:ReactNode}

export function ProjectFrame({id,steps,step,setStep,navigate}:{id:ProjectId;steps:StepDef[];step:number;setStep:(n:number)=>void;navigate:(r:Route,anchor?:string)=>void}){
  const meta=PROJECTS[id],chapter=CHAPTERS[meta.chapter-1];
  const [state,setState]=useState<ProjectState>(()=>loadProjects()[id]??blankProject());
  const [revealed,setRevealed]=useState(false);
  useEffect(()=>{setRevealed(false);setState(old=>old.visited.includes(step)?old:{...old,visited:[...old.visited,step].sort()})},[step]);
  useEffect(()=>saveProject(id,state),[id,state]);
  const current=steps[step];
  const allVisited=steps.every((_,i)=>state.visited.includes(i));
  return <>
    <header className="page-heading">
      <button className="back-link" onClick={()=>navigate({kind:'phase',id:chapter.id})}><ArrowLeft size={14}/> Chương {chapter.id}: {chapter.title}</button>
      <span className="eyebrow">Dự án · Chương {chapter.id}</span>
      <h1>{meta.question}</h1>
      <p>{meta.summary}</p>
      <div className="chip-row">{meta.skills.map(s=><span className="chip" key={s}>{s}</span>)}</div>
      <div className="heading-actions">{meta.days.map(d=><button key={d} className="btn-outline" onClick={()=>navigate({kind:'day',id:d})}>Bài ngày {d}</button>)}</div>
    </header>
    <section className="panel project-panel">
      <div className="project-tabs" role="tablist" aria-label="Các bước dự án">
        {steps.map((s,i)=><button key={s.label} role="tab" aria-selected={i===step} onClick={()=>setStep(i)}><span>{state.visited.includes(i)&&i!==step?<CheckCircle size={13}/>:i+1}</span>{s.label}</button>)}
      </div>
      <div className="project-workspace">
        <div className="project-controls">
          <span className="eyebrow">Bước {step+1} / {steps.length}</span>
          <h2>{current.title}</h2>
          <div className="step-intro">{current.intro}</div>
          {current.controls}
          <div className="takeaway">
            {revealed?<><strong><Lightbulb size={15}/> Điều rút ra</strong><p>{current.takeaway}</p></>:
              <button className="btn-quiet" onClick={()=>setRevealed(true)}><Lightbulb size={15}/> Em thử xong rồi, xem điều rút ra</button>}
          </div>
        </div>
        <div className="project-chart">{current.visual}</div>
      </div>
      {current.below&&<div className="project-evidence">{current.below}</div>}
    </section>
    <div className="previous-next">
      {step>0?<button className="btn-outline" onClick={()=>setStep(step-1)}><ArrowLeft size={15}/> {steps[step-1].label}</button>:<span/>}
      {step<steps.length-1&&<button className="btn-primary" onClick={()=>setStep(step+1)}>Bước tiếp: {steps[step+1].label} <ArrowRight size={15}/></button>}
    </div>
    <section className="panel project-wrap">
      <div><h3>Kết luận của em</h3><p className="small-note">Viết 3–5 dòng: em đã thử gì, quan sát gì, mô hình còn yếu ở đâu.</p>
        <textarea rows={4} value={state.note} aria-label="Kết luận dự án" onChange={e=>setState({...state,note:e.target.value})} placeholder="Ví dụ: Khi hạ ngưỡng xuống 0,3 thì bỏ sót ít ca ác tính hơn nhưng báo động nhầm nhiều hơn…"/></div>
      <div className="project-wrap-side">
        <p className="small-note">Đã xem {state.visited.length}/{steps.length} bước{allVisited?'':' · xem đủ các bước để hoàn thành'}.</p>
        <label className="check-line"><input type="checkbox" disabled={!allVisited&&!state.done} checked={state.done} onChange={e=>setState({...state,done:e.target.checked})}/>Em đã làm xong dự án này và viết kết luận</label>
        {state.done&&<span className="complete-label"><CheckCircle size={16}/> Dự án hoàn thành</span>}
        <p className="small-note">Dữ liệu: {meta.data}</p>
      </div>
    </section>
  </>;
}

/* ---------- Biểu đồ SVG nhỏ ---------- */
export type Scale=(v:number)=>number;
export function Plot({x,y,xLabel,yLabel,height=340,onPick,children,aspect=1.45}:{x:[number,number];y:[number,number];xLabel?:string;yLabel?:string;height?:number;aspect?:number;onPick?:(x:number,y:number)=>void;children:(sx:Scale,sy:Scale)=>ReactNode}){
  const W=Math.round(height*aspect),H=height,m={l:46,r:14,t:12,b:40};
  const sx:Scale=v=>m.l+(v-x[0])/(x[1]-x[0])*(W-m.l-m.r),sy:Scale=v=>H-m.b-(v-y[0])/(y[1]-y[0])*(H-m.t-m.b);
  const ref=useRef<SVGSVGElement>(null);
  const ticks=(a:number,b:number)=>{const step=niceStep((b-a)/5),out:number[]=[];for(let v=Math.ceil(a/step)*step;v<=b+1e-9;v+=step)out.push(+v.toFixed(6));return out};
  const pick=(e:MouseEvent<SVGSVGElement>)=>{if(!onPick||!ref.current)return;const r=ref.current.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W,py=(e.clientY-r.top)/r.height*H;
    onPick(x[0]+(px-m.l)/(W-m.l-m.r)*(x[1]-x[0]),y[0]+(H-m.b-py)/(H-m.t-m.b)*(y[1]-y[0]))};
  return <svg ref={ref} className={'plot'+(onPick?' pickable':'')} viewBox={`0 0 ${W} ${H}`} onClick={pick} role="img">
    <defs><clipPath id={`clip-${W}-${H}`}><rect x={m.l} y={m.t} width={W-m.l-m.r} height={H-m.t-m.b}/></clipPath></defs>
    {ticks(...x).map(t=><g key={'x'+t}><line className="grid" x1={sx(t)} x2={sx(t)} y1={m.t} y2={H-m.b}/><text className="tick" x={sx(t)} y={H-m.b+16} textAnchor="middle">{fmtTick(t)}</text></g>)}
    {ticks(...y).map(t=><g key={'y'+t}><line className="grid" x1={m.l} x2={W-m.r} y1={sy(t)} y2={sy(t)}/><text className="tick" x={m.l-7} y={sy(t)+4} textAnchor="end">{fmtTick(t)}</text></g>)}
    {xLabel&&<text className="axis-label" x={(m.l+W-m.r)/2} y={H-6} textAnchor="middle">{xLabel}</text>}
    {yLabel&&<text className="axis-label" transform={`translate(13 ${(m.t+H-m.b)/2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
    <g clipPath={`url(#clip-${W}-${H})`}>{children(sx,sy)}</g>
  </svg>;
}
function niceStep(raw:number){const p=10**Math.floor(Math.log10(raw||1)),f=raw/p;return (f<1.5?1:f<3?2:f<7?5:10)*p}
const fmtTick=(v:number)=>Math.abs(v)>=1000?(v/1000)+'k':String(+v.toFixed(2)).replace('.',',');

export function LineChart({series,xLabel,yLabel,height=220,yMin}:{series:{name:string;values:number[];color:string;dashed?:boolean}[];xLabel?:string;yLabel?:string;height?:number;yMin?:number}){
  const n=Math.max(2,...series.map(s=>s.values.length)),all=series.flatMap(s=>s.values);
  let lo=yMin??Math.min(...all),hi=Math.max(...all);if(hi-lo<1e-9){hi+=1;lo-=1}const pad=(hi-lo)*.08;
  return <div><Plot x={[0,n-1]} y={[yMin??lo-pad,hi+pad]} xLabel={xLabel} yLabel={yLabel} height={height} aspect={Math.max(2,560/height)}>
    {(sx,sy)=>series.map(s=><polyline key={s.name} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dashed?'6 5':undefined} points={s.values.map((v,i)=>`${sx(i)},${sy(v)}`).join(' ')}/>)}
  </Plot><Legend items={series.map(s=>[s.name,s.color])}/></div>;
}
export function Legend({items}:{items:[string,string][]}){return <ul className="plot-legend">{items.map(([n,c])=><li key={n}><i style={{background:c}}/>{n}</li>)}</ul>}

export function Bars({rows,max,format=(v:number)=>fmt(v)}:{rows:{label:ReactNode;value:number;color?:string;note?:ReactNode}[];max?:number;format?:(v:number)=>string}){
  const m=max??Math.max(1e-9,...rows.map(r=>Math.abs(r.value)));
  return <div className="hbars">{rows.map((r,i)=><div className="hbar" key={i}><span className="hbar-label">{r.label}</span><span className="hbar-track"><i style={{width:`${Math.min(100,Math.abs(r.value)/m*100)}%`,background:r.color??'var(--accent)'}}/></span><span className="hbar-value">{format(r.value)}{r.note}</span></div>)}</div>;
}

export function Confusion({labels,matrix,highlight}:{labels:string[];matrix:number[][];highlight?:[number,number]}){
  const max=Math.max(1,...matrix.flat());
  return <div className="table-wrap"><table className="confusion"><thead><tr><th>Thật ↓ / Đoán →</th>{labels.map(l=><th key={l}>{l}</th>)}</tr></thead>
    <tbody>{matrix.map((row,i)=><tr key={i}><th>{labels[i]}</th>{row.map((v,j)=><td key={j} className={(i===j?'diag':v?'off':'')+(highlight&&highlight[0]===i&&highlight[1]===j?' hl':'')} style={{['--a' as string]:v/max}}>{v}</td>)}</tr>)}</tbody></table></div>;
}

export function Stat({label,value,tone}:{label:string;value:ReactNode;tone?:'good'|'bad'|'warn'}){return <div className={'stat '+(tone??'')}><span>{label}</span><strong>{value}</strong></div>}
export function Slider({label,value,min,max,step=1,onChange,format}:{label:string;value:number;min:number;max:number;step?:number;onChange:(v:number)=>void;format?:(v:number)=>string}){
  return <label className="range-field"><span>{label}<output>{format?format(value):String(value).replace('.',',')}</output></span><input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(+e.target.value)}/></label>;
}
export function Segmented<T extends string|number>({options,value,onChange,label}:{options:[T,string][];value:T;onChange:(v:T)=>void;label:string}){
  return <div className="segmented" role="radiogroup" aria-label={label}>{options.map(([v,t])=><button key={String(v)} role="radio" aria-checked={v===value} className={v===value?'on':''} onClick={()=>onChange(v)}>{t}</button>)}</div>;
}

/** Hẹn giờ phát/tạm dừng cho mô phỏng chạy theo khung hình. */
export function usePlayer(count:number,interval=120){
  const [frame,setFrame]=useState(0),[playing,setPlaying]=useState(false);
  useEffect(()=>{if(!playing)return;const t=setInterval(()=>setFrame(f=>{if(f>=count-1){setPlaying(false);return f}return f+1}),interval);return()=>clearInterval(t)},[playing,count,interval]);
  useEffect(()=>{setFrame(f=>Math.min(f,Math.max(0,count-1)))},[count]);
  return {frame,setFrame:(f:number)=>{setPlaying(false);setFrame(Math.max(0,Math.min(count-1,f)))},playing,toggle:()=>{if(frame>=count-1)setFrame(0);setPlaying(p=>!p)},reset:()=>{setPlaying(false);setFrame(0)}};
}
export function PlayerBar({player,count,label='Bước'}:{player:ReturnType<typeof usePlayer>;count:number;label?:string}){
  return <div className="player-bar">
    <button className="btn-outline" onClick={player.reset}>Đặt lại</button>
    <button className="btn-primary" onClick={player.toggle}>{player.playing?'Tạm dừng':'Chạy'}</button>
    <button className="btn-outline" disabled={player.frame>=count-1} onClick={()=>player.setFrame(player.frame+1)}>+1</button>
    <label className="player-seek"><span>{label} {player.frame}/{count-1}</span><input type="range" min={0} max={count-1} value={player.frame} onChange={e=>player.setFrame(+e.target.value)}/></label>
  </div>;
}
