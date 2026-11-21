import {ArrowRight,ChartLine,FlaskConical,House,Map,NotebookPen,SquareTerminal,Target} from 'lucide-react';
import {CHAPTERS,PROJECTS} from '../projects/chapters';
import type {Progress,Route} from '../types';
import {course} from '../data';
import {complete} from '../lib/domain';

/** Màu sáng cho sáu giai đoạn trên nền tối (màu trong course.json được chọn cho nền sáng). */
export const PHASE_COLORS=['#42d3e5','#84e0a8','#ffd173','#f28db9','#a8a7ff','#ff9b6a'];
const PY_KEY='mlStudioPy30-progress-v1';
const PY_BLOCKS=7;

type DayState='done'|'partial'|'todo';
function dayState(progress:Progress,id:number):DayState{
  const p=progress.days[id];
  if(complete(p))return 'done';
  return p&&(p.a||p.b||p.c||p.review||p.note.trim())?'partial':'todo';
}
function pythonDaysDone():number{
  try{
    const raw=JSON.parse(localStorage.getItem(PY_KEY)??'null') as {days?:Record<string,{blocks?:unknown[]}>}|null;
    return Object.values(raw?.days??{}).filter(d=>Array.isArray(d.blocks)&&d.blocks.length>=PY_BLOCKS&&d.blocks.every(b=>b===true)).length;
  }catch{return 0}
}

export function ProgressRing({value,total,size=132,stroke=10,label}:{value:number;total:number;size?:number;stroke?:number;label?:string}){
  const r=(size-stroke)/2,c=2*Math.PI*r,frac=total?value/total:0;
  return <svg className="progress-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${value}/${total} ${label??''}`}>
    <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke}/>
    {frac>0&&<circle cx={size/2} cy={size/2} r={r} fill="none" stroke="var(--accent)" strokeWidth={stroke} strokeLinecap="round" strokeDasharray={`${c*frac} ${c}`} transform={`rotate(-90 ${size/2} ${size/2})`}/>}
    <text x="50%" y="47%" textAnchor="middle" dominantBaseline="middle" className="ring-value" style={{fontSize:size*.24}}>{value}</text>
    {label&&<text x="50%" y="66%" textAnchor="middle" dominantBaseline="middle" className="ring-label" style={{fontSize:Math.max(10,size*.085)}}>/{total} {label}</text>}
  </svg>;
}

export function Home({progress,navigate}:{progress:Progress;navigate:(r:Route,anchor?:string)=>void}){
  const states=course.days.map(d=>dayState(progress,d.id));
  const done=states.filter(s=>s==='done').length;
  const started=states.some(s=>s!=='todo');
  const nextIndex=states.findIndex(s=>s!=='done');
  const next=nextIndex>=0?course.days[nextIndex]:null;
  const nextPhase=next?course.phases.find(p=>p.id===next.phase)!:null;
  const tasks=Object.values(progress.days).reduce((n,p)=>n+(+p.a)+(+p.b)+(+p.c),0);
  const notes=Object.values(progress.days).filter(p=>p.note.trim()).length;
  const py=pythonDaysDone();

  return <div className="home">
    <section className="home-hero">
      <div className="home-hero-copy">
        <span className="eyebrow">{started?'Tiếp tục hành trình':'Bắt đầu từ một câu hỏi'}</span>
        {next&&started?<>
          <h1><span className="hero-day" style={{color:PHASE_COLORS[next.phase-1]}}>Ngày {next.id}</span>{next.title}</h1>
          <p>{next.bullets[0]?.text.replaceAll('**','').replaceAll('`','')}</p>
          <div className="heading-actions">
            <button className="btn-primary" onClick={()=>navigate({kind:'day',id:next.id})}>{states[nextIndex]==='partial'?'Làm tiếp bài này':'Vào bài học'} <ArrowRight size={16}/></button>
            <button className="btn-outline" onClick={()=>navigate({kind:'phase',id:next.phase})}>Giai đoạn {next.phase}: {nextPhase?.title}</button>
          </div>
        </>:next?<>
          <h1>Nhà 80 m² giá bao nhiêu?</h1>
          <p>Đoán trước, rồi để mô hình kiểm chứng. Mọi khái niệm trong 30 ngày đều bắt đầu từ một thí nghiệm nhỏ như vậy.</p>
          <div className="heading-actions">
            <button className="btn-primary" onClick={()=>navigate({kind:'project'})}>Thử dự án giá nhà <ArrowRight size={16}/></button>
            <button className="btn-outline" onClick={()=>navigate({kind:'day',id:1})}>Hoặc mở Ngày 1</button>
          </div>
        </>:<>
          <h1>Em đã đi hết 30 ngày.</h1>
          <p>Giờ là lúc tự làm một dự án của riêng em, hoặc học tiếp mạng nơ-ron đồ thị.</p>
          <div className="heading-actions">
            <button className="btn-primary" onClick={()=>navigate({kind:'assessments'},'project-rubric')}>Mở rubric dự án <ArrowRight size={16}/></button>
            <button className="btn-outline" onClick={()=>navigate({kind:'gnn'})}>Học tiếp GNN</button>
          </div>
        </>}
      </div>
      <div className="home-hero-ring"><ProgressRing value={done} total={30} label="ngày"/></div>
    </section>

    <section className="home-stats" aria-label="Thống kê học tập">
      <div><Target size={18}/><strong>{done}</strong><span>ngày hoàn thành</span></div>
      <div><FlaskConical size={18}/><strong>{tasks}</strong><span>bài A/B/C đã làm</span></div>
      <div><NotebookPen size={18}/><strong>{notes}</strong><span>nhật ký đã viết</span></div>
      <div><SquareTerminal size={18}/><strong>{py}</strong><span>ngày Python → ML</span></div>
    </section>


    {next&&(()=>{const ch=CHAPTERS[next.phase-1];return <section className="home-chapter" style={{['--phase' as string]:PHASE_COLORS[ch.id-1]}}>
      <div><span className="eyebrow">Chương đang học · {ch.id}/6</span><h2>{ch.title}</h2>
        <h3 className="outcome-title"><Target size={15}/> Học xong chương này em làm được</h3>
        <ul className="outcomes">{ch.outcomes.map(o=><li key={o}>{o}</li>)}</ul></div>
      <div className="home-chapter-side">{ch.projects.map(p=><button key={p} className="chip proj-chip" onClick={()=>navigate(p==='housing'?{kind:'project'}:{kind:'proj',id:p})}><FlaskConical size={13}/>{PROJECTS[p].title}</button>)}
        <button className="btn-outline" onClick={()=>navigate({kind:'phase',id:ch.id})}>Xem cả chương <ArrowRight size={14}/></button></div>
    </section>})()}
    <section className="home-map">
      <div className="section-head"><div><span className="eyebrow">Bản đồ 6 chương · 30 ngày</span><h2>Mỗi ô là một buổi học. Bấm để mở.</h2></div>
        <ul className="map-legend"><li><i className="cell done"/>Hoàn thành</li><li><i className="cell partial"/>Đang làm</li><li><i className="cell"/>Chưa học</li></ul></div>
      <div className="phase-columns">
        {course.phases.map(p=>{const days=course.days.filter(d=>d.phase===p.id);const n=days.filter(d=>states[d.id-1]==='done').length;const color=PHASE_COLORS[p.id-1];
          return <div className="phase-col" key={p.id} style={{['--phase' as string]:color}}>
            <button className="phase-col-head" onClick={()=>navigate({kind:'phase',id:p.id})}>
              <span className="phase-dot">{p.id}</span><span><strong>{p.title}</strong><small>Ngày {p.range[0]}–{p.range[1]} · {n}/{days.length}</small></span>
            </button>
            <div className="phase-bar"><i style={{width:`${n/days.length*100}%`}}/></div>
            <div className="day-cells">{days.map(d=>{const s=states[d.id-1];return <button key={d.id} className={`cell ${s}${next?.id===d.id?' next':''}`} title={`Ngày ${d.id}: ${d.title}`} aria-label={`Ngày ${d.id}: ${d.title} (${s==='done'?'hoàn thành':s==='partial'?'đang làm':'chưa học'})`} onClick={()=>navigate({kind:'day',id:d.id})}>{d.id}</button>})}</div>
          </div>})}
      </div>
    </section>

    <section className="home-labs">
      <div className="section-head"><div><span className="eyebrow">Phòng lab</span><h2>Vào thẳng thực hành</h2></div></div>
      <div className="lab-cards">
        <button className="lab-card" onClick={()=>navigate({kind:'projects'})}><FlaskConical size={22}/><strong>9 dự án theo chương</strong><span>Khối u, email rác, chữ số viết tay, robot tự học… mỗi dự án 5 bước.</span></button>
        <button className="lab-card" onClick={()=>navigate({kind:'project'})}><House size={22}/><strong>Dự án giá nhà</strong><span>Đoán giá, vẽ đường thẳng, cho máy học, rồi thử dữ liệu thật.</span></button>
        <button className="lab-card" onClick={()=>navigate({kind:'py'})}><SquareTerminal size={22}/><strong>Python → ML 30 ngày</strong><span>Tự cài từng thuật toán bằng Python, mỗi ngày một mô phỏng.</span></button>
        <button className="lab-card" onClick={()=>navigate({kind:'exercises'})}><NotebookPen size={22}/><strong>Thư viện bài tập</strong><span>90 bài theo mức A/B/C, tìm theo điều em muốn hiểu.</span></button>
        <button className="lab-card" onClick={()=>navigate({kind:'lab'})}><ChartLine size={22}/><strong>Phòng hồi quy</strong><span>Mô phỏng hồi quy nâng cao: nhiều biến, Iris, CSV của em.</span></button>
        <button className="lab-card" onClick={()=>navigate({kind:'overview'})}><Map size={22}/><strong>6 chương học</strong><span>Mục tiêu "học xong làm được gì" và mốc kiểm tra của từng chương.</span></button>
      </div>
    </section>
  </div>;
}
