import {ArrowRight,CheckCircle,ChevronRight,Circle,ClipboardCheck,FlaskConical,Target} from 'lucide-react';
import type {Progress,Route} from '../types';
import {course} from '../data';
import {complete} from '../lib/domain';
import {PHASE_COLORS} from '../components/Home';
import {CHAPTERS,PROJECTS} from './chapters';
import type {Chapter} from './chapters';
import type {ProjectId} from './ids';
import {Segmented,loadMode,loadProjects,saveMode} from './ui';
import type {LearnMode} from './ui';
import {useState} from 'react';
import type {ProjectStore} from './ui';
import {TumorProject} from './TumorProject';
import {SpamProject} from './SpamProject';
import {IrisProject} from './IrisProject';
import {CustomersProject} from './CustomersProject';
import {DigitsProject} from './DigitsProject';
import {SentimentProject} from './SentimentProject';
import {RLProject} from './RLProject';
import {GnnProject} from './GnnProject';

type Nav=(r:Route,anchor?:string)=>void;
const openProject=(id:ProjectId|'housing',navigate:Nav)=>navigate(id==='housing'?{kind:'project'}:{kind:'proj',id});

export function chapterStatus(ch:Chapter,progress:Progress,store:ProjectStore){
  const days=course.days.filter(d=>d.id>=ch.days[0]&&d.id<=ch.days[1]);
  const daysDone=days.filter(d=>complete(progress.days[d.id])).length;
  const tracked=ch.projects.filter(p=>p!=='housing');
  const projDone=tracked.filter(p=>store[p]?.done).length;
  const total=days.length+tracked.length,doneAll=daysDone+projDone;
  return {days,daysDone,tracked,projDone,frac:total?doneAll/total:0,finished:total>0&&doneAll===total};
}

function ProjectCard({id,navigate,store}:{id:ProjectId|'housing';navigate:Nav;store:ProjectStore}){
  const m=PROJECTS[id],s=id==='housing'?undefined:store[id];
  return <button className="proj-card" onClick={()=>openProject(id,navigate)} style={{['--phase' as string]:PHASE_COLORS[m.chapter-1]}}>
    <span className="proj-card-top"><span className="eyebrow">Chương {m.chapter}</span>{s?.done?<span className="proj-status done"><CheckCircle size={14}/> Hoàn thành</span>:s&&(s.visited.length||s.visitedR.length)?<span className="proj-status">Đã xem {Math.max(s.visited.length,s.visitedR.length)} phần</span>:null}</span>
    <strong>{m.question}</strong>
    <span className="proj-title">{m.title}</span>
    <span className="proj-summary">{m.summary}</span>
    <span className="chip-row">{m.skills.slice(0,3).map(k=><span className="chip" key={k}>{k}</span>)}</span>
  </button>;
}

export function ChaptersPage({progress,navigate}:{progress:Progress;navigate:Nav}){
  const store=loadProjects();
  return <>
    <header className="page-heading"><span className="eyebrow">6 chương · 30 ngày · 9 dự án</span><h1>Học theo chương, biết rõ mình sẽ làm được gì.</h1>
      <p>Mỗi chương gồm vài buổi học ngắn và ít nhất một dự án. Đọc mục <b>"Học xong em làm được"</b> trước khi bắt đầu, rồi quay lại tự đánh dấu khi xong chương.</p></header>
    <div className="chapter-list">{CHAPTERS.map(ch=>{const st=chapterStatus(ch,progress,store),color=PHASE_COLORS[ch.id-1];
      return <article className={'chapter-card'+(st.finished?' finished':'')} key={ch.id} style={{['--phase' as string]:color}}>
        <div className="chapter-badge">{st.finished?<CheckCircle size={22}/>:ch.id}</div>
        <div className="chapter-body">
          <span className="eyebrow">Chương {ch.id} · Ngày {ch.days[0]}–{ch.days[1]}{ch.checkpoint?` · Mốc kiểm tra ngày ${ch.checkpoint}`:''}</span>
          <h2>{ch.title}</h2>
          <div className="chapter-progress"><i style={{width:`${st.frac*100}%`}}/></div>
          <small className="small-note">{st.daysDone}/{st.days.length} ngày học{st.tracked.length?` · ${st.projDone}/${st.tracked.length} dự án`:''}</small>
          <h3 className="outcome-title"><Target size={15}/> Học xong em làm được</h3>
          <ul className="outcomes">{ch.outcomes.map(o=><li key={o}>{o}</li>)}</ul>
          {ch.projects.length>0&&<div className="chapter-projects">{ch.projects.map(p=><button key={p} className="chip proj-chip" onClick={()=>openProject(p,navigate)}><FlaskConical size={13}/>{PROJECTS[p].title}</button>)}</div>}
        </div>
        <button className="btn-outline" onClick={()=>navigate({kind:'phase',id:ch.id})}>Vào chương <ChevronRight size={15}/></button>
      </article>})}</div>
  </>;
}

export function ChapterPage({id,progress,navigate}:{id:number;progress:Progress;navigate:Nav}){
  const ch=CHAPTERS[id-1],store=loadProjects(),st=chapterStatus(ch,progress,store),color=PHASE_COLORS[id-1];
  return <div style={{['--phase' as string]:color}}>
    <header className="page-heading"><button className="back-link" onClick={()=>navigate({kind:'overview'})}>← Tất cả chương</button>
      <span className="eyebrow">Chương {ch.id} · Ngày {ch.days[0]}–{ch.days[1]}</span><h1>{ch.title}</h1>
      <div className="chapter-progress big"><i style={{width:`${st.frac*100}%`}}/></div>
      <p className="small-note">{st.daysDone}/{st.days.length} ngày học · {st.projDone}/{st.tracked.length||0} dự án{st.finished?' · Đã hoàn thành chương':''}</p></header>
    <section className="outcome-panel">
      <div><h2><Target size={18}/> Học xong chương này, em làm được</h2>
        <ul className="outcomes big">{ch.outcomes.map((o,i)=><li key={o} className={st.finished?'met':''}>{st.finished?<CheckCircle size={16}/>:<Circle size={16}/>}<span><b>{i+1}.</b> {o}</span></li>)}</ul></div>
      <div className="outcome-side"><h3>Sản phẩm của chương</h3><p>{ch.product}</p>
        {ch.checkpoint&&<button className="btn-outline" onClick={()=>navigate({kind:'day',id:ch.checkpoint!},`checkpoint-${ch.checkpoint}`)}><ClipboardCheck size={15}/> Mốc kiểm tra ngày {ch.checkpoint}</button>}
        {id===6&&<button className="btn-outline" onClick={()=>navigate({kind:'assessments'},'project-rubric')}><ClipboardCheck size={15}/> Rubric dự án cuối</button>}</div>
    </section>
    {ch.projects.length>0&&<section className="chapter-section"><div className="section-head"><div><span className="eyebrow">Dự án của chương</span><h2>Học qua việc làm</h2></div></div>
      <div className="proj-grid">{ch.projects.map(p=><ProjectCard key={p} id={p} navigate={navigate} store={store}/>)}</div></section>}
    <section className="chapter-section"><div className="section-head"><div><span className="eyebrow">Các buổi học</span><h2>{st.days.length} buổi, mỗi buổi khoảng 2 giờ</h2></div></div>
      <ol className="lesson-list">{st.days.map(d=>{const done=complete(progress.days[d.id]);return <li key={d.id} className={done?'done':''}><button onClick={()=>navigate({kind:'day',id:d.id})}>
        <span className="lesson-num">{done?<CheckCircle size={18}/>:d.id}</span><span><strong>Ngày {d.id}: {d.title}</strong><small>{d.bullets[0]?.text.replaceAll('**','').replaceAll('`','')}</small></span><ArrowRight size={16}/></button></li>})}</ol></section>
    <div className="previous-next">{id>1?<button className="btn-outline" onClick={()=>navigate({kind:'phase',id:id-1})}>← Chương {id-1}</button>:<span/>}{id<6&&<button className="btn-primary" onClick={()=>navigate({kind:'phase',id:id+1})}>Chương {id+1}: {CHAPTERS[id].title} <ArrowRight size={15}/></button>}</div>
  </div>;
}

export function ProjectsHub({navigate}:{navigate:Nav}){
  const store=loadProjects(),tracked=Object.keys(PROJECTS).filter(k=>k!=='housing'),done=tracked.filter(k=>store[k]?.done).length;
  const [mode,setMode]=useState<LearnMode>(loadMode);
  return <>
    <header className="page-heading"><span className="eyebrow">Phòng lab dự án · {done}/{tracked.length} hoàn thành</span><h1>Chọn một câu hỏi để trả lời.</h1>
      <p>Mỗi dự án đi qua 5 bước: em đoán trước, tự làm bằng tay, cho máy học, thử thách mô hình, rồi rút ra kết luận. Chạy ngay trên trình duyệt, không cần cài đặt.</p></header>
    <div className="mode-switch hub"><span className="eyebrow">Cách học mặc định</span>
      <Segmented label="Cách học mặc định" options={[['forward','Học xuôi · tự xây từ đầu'],['reverse','Học ngược · tháo từ sản phẩm']]} value={mode} onChange={m=>{setMode(m);saveMode(m)}}/>
      <p className="small-note">{mode==='reverse'?'Mỗi dự án mở ra là một mô hình đã chạy tốt; em tháo dần từng lớp rồi tự lắp lại.':'Mỗi dự án bắt đầu từ trang trắng: đoán, tự làm, rồi cho máy học. Trong từng dự án em vẫn đổi được cách học.'}</p></div>
    {CHAPTERS.filter(c=>c.projects.length).map(ch=><section className="chapter-section" key={ch.id}>
      <div className="section-head"><div><span className="eyebrow" style={{color:PHASE_COLORS[ch.id-1]}}>Chương {ch.id}</span><h2>{ch.title}</h2></div><button className="btn-quiet" onClick={()=>navigate({kind:'phase',id:ch.id})}>Mục tiêu chương <ChevronRight size={14}/></button></div>
      <div className="proj-grid">{ch.projects.map(p=><ProjectCard key={p} id={p} navigate={navigate} store={store}/>)}</div></section>)}
  </>;
}

export function ProjectPage({id,navigate}:{id:ProjectId;navigate:Nav}){
  switch(id){
    case 'tumor':return <TumorProject navigate={navigate}/>;
    case 'spam':return <SpamProject navigate={navigate}/>;
    case 'iris':return <IrisProject navigate={navigate}/>;
    case 'customers':return <CustomersProject navigate={navigate}/>;
    case 'digits':return <DigitsProject navigate={navigate}/>;
    case 'sentiment':return <SentimentProject navigate={navigate}/>;
    case 'rl':return <RLProject navigate={navigate}/>;
    case 'gnn':return <GnnProject navigate={navigate}/>;
  }
}
