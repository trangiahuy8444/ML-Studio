import {SimulationStudio,SimulationPanel,SimulationCode} from '../components/SimulationStudio';
import {useEffect,useRef,useState} from 'react';
import {PythonEditor} from '../components/PythonEditor';
import {Table,Transport,usePlayback} from '../components/ObservationControls';
import {download} from '../lib/domain';
import {WeekOneLab} from './WeekOneLab';
import {discoverServer,LocalKernel,openInJupyterHref} from './kernel/client';
import type {KernelStatus} from './kernel/client';
import {extractReports,stripAnsi,toNotebook} from './kernel/protocol';
import type {CellOutput,PageCell,Report} from './kernel/protocol';
import week1 from './week1.json';
import week2 from './week2.json';
import {WeekTwoLab} from './WeekTwoLab';
import week3 from './week3.json';
import {WeekThreeLab} from './WeekThreeLab';
import week4 from './week4.json';
import {WeekFourLab} from './WeekFourLab';
import week5 from './week5.json';
import {WeekFiveLab} from './WeekFiveLab';
import {AlgoAtlas,AlgoCard,cardsForDay} from './AlgoCards';
const content=[...week1.map(l=>({...l,projectTitle:'House Prices'})),...week2,...week3,...week4,...week5];
import './python-track.css';

const storageKey='mlStudioPy30-progress-v1';
const blocks=[['Khởi động',10],['Trực quan hóa',30],['Mô phỏng code',20],['Đọc có chủ đích',20],['Lab từ đầu',50],['Dự án',40],['Tổng kết',10]] as const;
type Lesson=(typeof content)[number];
type DayProgress={blocks:boolean[];journal:string;prediction:string;grades:Record<string,boolean>};
type Progress={version:1;days:Record<string,DayProgress>};
const emptyDay=():DayProgress=>({blocks:blocks.map(()=>false),journal:'',prediction:'',grades:{}});
function progressFrom(value:unknown):Progress {
  if(!value||typeof value!=='object'||!('version'in value)||value.version!==1||
     !('days'in value)||!value.days||typeof value.days!=='object'||Array.isArray(value.days))throw Error('Tệp không phải tiến độ Python → ML phiên bản 1.');
  const days:Progress['days']={};
  for(const d of content){
    const row=(value.days as Record<string,Partial<DayProgress>>)[d.id];
    if(!row||typeof row!=='object')continue;
    days[d.id]={blocks:blocks.map((_,i)=>Array.isArray(row.blocks)&&row.blocks[i]===true),
      journal:typeof row.journal==='string'?row.journal.slice(0,10000):'',
      prediction:typeof row.prediction==='string'?row.prediction.slice(0,1000):'',
      grades:Object.fromEntries(d.exercises.filter(e=>typeof row.grades?.[e.id]==='boolean').map(e=>[e.id,row.grades![e.id]]))};
  }
  return {version:1,days};
}
function loadProgress():Progress {
  try{return progressFrom(JSON.parse(localStorage.getItem(storageKey)??'null'));}
  catch{return {version:1,days:{}};}
}
const labels:Record<KernelStatus,string>={offline:'Chưa kết nối',connecting:'Đang tìm Jupyter',starting:'Đang tạo kernel',idle:'Sẵn sàng',busy:'Đang chạy',restarting:'Đang khởi động lại',dead:'Mất kết nối'};
const errorText=(error:unknown)=>error instanceof Error?error.message:String(error);
type CellRun={output:CellOutput[];report?:Report;attempted:boolean;error?:string};

function Outputs({outputs}:{outputs:CellOutput[]}) {
  return <div className="py30-output" aria-live="polite">{outputs.map((o,i)=>{
    if(o.output_type==='stream')return <pre key={i}>{stripAnsi(o.text??'')}</pre>;
    if(o.output_type==='error')return <pre key={i} role="alert">{stripAnsi(o.traceback?.join('\n')||`${o.ename}: ${o.evalue}`)}</pre>;
    const png=o.data?.['image/png'],html=o.data?.['text/html'],plain=o.data?.['text/plain'];
    if(typeof png==='string'&&/^[\w+/=\s]+$/.test(png))return <img key={i} src={'data:image/png;base64,'+png} alt="Biểu đồ do Python tạo"/>;
    if(typeof html==='string')return <iframe key={i} title="Kết quả HTML từ Python" sandbox="allow-same-origin" referrerPolicy="no-referrer" srcDoc={'<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:"><base target="_self">'+html}/>;
    return typeof plain==='string'?<pre key={i}>{plain}</pre>:null;
  })}</div>;
}

function CodeTrace({lesson}:{lesson:Lesson}) {
  const player=usePlayback(lesson.steps.length,lesson.id);
  return <SimulationPanel title="Python · Ví dụ tính từng dòng"><Transport player={player} count={lesson.steps.length}/>
    <SimulationCode lines={lesson.steps.map(s=>s.line)} active={player.index} detail={lesson.steps[player.index].state}/>
    <Table headers={['Dòng đã xem','Giá trị sau bước']} rows={lesson.steps.slice(0,player.index+1).map(s=>[s.line,s.state])} active={player.index}/>
  </SimulationPanel>;
}

function DayPage({lesson,progress,change,onBusy,openDay}:{lesson:Lesson;progress:DayProgress;change:(patch:Partial<DayProgress>)=>void;onBusy:(b:boolean)=>void;openDay:(id?:number)=>void}) {
  const {id}=lesson, file=`Ngay_${String(id).padStart(2,'0')}.ipynb`,modelPath='py30/'+file,workPath='py30/bai_lam/'+file;
  const client=useRef<LocalKernel|null>(null), mounted=useRef(true), action=useRef(false);
  const [status,setStatus]=useState<KernelStatus>('offline'),[working,setWorking]=useState(false),[notice,setNotice]=useState(''),[hasSaved,setHasSaved]=useState(false);
  const [codes,setCodes]=useState<Record<string,string>>(()=>Object.fromEntries(lesson.exercises.map(e=>[e.id,e.starter])));
  const [runs,setRuns]=useState<Record<string,CellRun>>({});
  useEffect(()=>{
    mounted.current=true;
    return ()=>{mounted.current=false; onBusy(false); if(client.current){client.current.onStatus=()=>{};void client.current.shutdown();client.current=null;}};
  },[onBusy]);
  const runAction=async(fn:()=>Promise<void>)=>{
    if(action.current)return;
    action.current=true;setWorking(true);onBusy(true);setNotice('');
    try{await fn();}
    catch(error){if(mounted.current)setNotice(errorText(error));}
    finally{action.current=false;if(mounted.current){setWorking(false);onBusy(false);}}
  };
  async function connect() {
    if(client.current){client.current.onStatus=()=>{}; await client.current.shutdown();client.current=null;}
    setStatus('connecting');
    try{
      const server=await discoverServer();
      if(!mounted.current)return;
      const kernel=new LocalKernel(server,modelPath);
      client.current=kernel;kernel.onStatus=setStatus;
      await kernel.start();
      const saved=await kernel.loadNotebook(workPath);
      setHasSaved(saved!==null);
    }catch(error){setStatus('offline');throw error;}
  }
  async function runExercise(exercise:Lesson['exercises'][number]) {
    const k=client.current;
    if(!k)return;
    const oldOutput=runs[exercise.id]?.output??[];
    setRuns(previous=>({...previous,[exercise.id]:{output:oldOutput,attempted:true}}));
    change({grades:{...progress.grades,[exercise.id]:false}});
    const update=(output:CellOutput[],report?:Report,error?:string)=>setRuns(previous=>({...previous,[exercise.id]:{output,attempted:true,report,error}}));
    const setup=await k.execute(lesson.setup,undefined,{storeHistory:false});
    if(setup.status!=='ok'){update(setup.outputs,undefined,'Ô chuẩn bị gặp lỗi.');return;}
    // Xóa các hàm và lớp của bài khỏi kernel trước khi chạy, để kết quả chấm phản ánh đúng code đang có trong ô.
    const names=[...exercise.starter.matchAll(/^(?:def|class)\s+([A-Za-z_]\w*)/gm)].map(m=>m[1]);
    if(names.length)await k.execute(`for _n in ${JSON.stringify(names)}:\n    globals().pop(_n, None)`,undefined,{storeHistory:false});
    const result=await k.execute(codes[exercise.id],s=>update(s.outputs));
    if(result.status!=='ok'){update(result.outputs,undefined,'Ô bài làm gặp lỗi hoặc đã bị ngắt.');return;}
    const graded=await k.execute(exercise.test,undefined,{storeHistory:false});
    const parsed=extractReports(graded.outputs);
    const report=graded.status==='ok'?parsed.reports.find(r=>r.id===exercise.id):undefined;
    const complete=Boolean(report&&report.checks.length&&report.checks.every(c=>c.ok));
    update([...result.outputs,...parsed.outputs],report,report?undefined:'Ô kiểm tra không trả về kết quả hợp lệ.');
    change({grades:{...progress.grades,[exercise.id]:complete}});
  }
  async function save() {
    const cells:PageCell[]=[{kind:'markdown',source:`# Ngày ${id} · ${lesson.title}`},{kind:'code',source:lesson.setup,tag:'setup'}];
    for(const e of lesson.exercises){
      cells.push({kind:'markdown',source:e.title},{kind:'code',source:codes[e.id],tag:e.id,outputs:runs[e.id]?.output},{kind:'code',source:e.test,tag:`${e.id}-check`});
    }
    await client.current!.saveNotebook(workPath,toNotebook(cells,lesson.title));
    setHasSaved(true);setNotice(`Đã lưu ${workPath}`);
  }
  async function load() {
    const saved=await client.current!.loadNotebook(workPath);
    if(!saved){setNotice('Chưa có bài lưu cho ngày này.');return;}
    const values:Record<string,string>={};
    for(const e of lesson.exercises){
      const cell=saved.cells.find(c=>c.cell_type==='code'&&c.metadata?.tags?.includes(e.id));
      values[e.id]=cell?(Array.isArray(cell.source)?cell.source.join(''):cell.source):e.starter;
    }
    setCodes(values);setRuns({});setNotice('Đã nạp bài đã lưu.');
  }
  function edit(exerciseId:string,value:string) {
    setCodes(old=>({...old,[exerciseId]:value}));
    setRuns(old=>old[exerciseId]?{...old,[exerciseId]:{output:[],attempted:old[exerciseId].attempted}}:old);
    if(progress.grades[exerciseId])change({grades:{...progress.grades,[exerciseId]:false}});
  }
  const connected=status==='idle'||status==='busy';
  const prev=id>1?content[id-2]:undefined;
  const terminalAction=async(kind:'interrupt'|'restart')=>{
    try{await client.current?.[kind](); if(kind==='restart')setNotice('Kernel đã khởi động lại.');}
    catch(error){setNotice(errorText(error));}
  };
  return <>
    <header className="page-heading"><button className="btn-outline" onClick={()=>openDay()}>Lộ trình Python → ML</button><h1>Ngày {id} · {lesson.title}</h1>
      <ul>{lesson.goals.map(goal=><li key={goal}>{goal}</li>)}</ul></header>
    <section className="panel py30-section"><h2>Tiến độ buổi học</h2><div className="py30-checklist">{blocks.map(([label,minutes],i)=><label key={label}><input type="checkbox" checked={progress.blocks[i]} onChange={e=>change({blocks:progress.blocks.map((v,j)=>j===i?e.target.checked:v)})}/> {label} · {minutes} phút</label>)}</div></section>
    <section className="panel py30-section"><h2>Khởi động</h2>
      {prev?<ol>{prev.questions.map(q=><li key={q}>{q}</li>)}</ol>:<p>Ôn lại danh sách, mảng NumPy và cách lấy kích thước của mảng hai chiều.</p>}
    </section>
    {cardsForDay(id).length>0&&<div className="day-hero-cards">{cardsForDay(id).map(c=><AlgoCard key={c.id} spec={c}/>)}</div>}
    <SimulationStudio title="Quan sát từng phép tính" description={lesson.title}>
      <label>Dự đoán trước khi thử<textarea rows={2} value={progress.prediction} onChange={e=>change({prediction:e.target.value.slice(0,1000)})}/></label>
      {id<=7?<WeekOneLab day={id}/>:id<=14?<WeekTwoLab day={id}/>:id<=21?<WeekThreeLab day={id}/>:id<=28?<WeekFourLab day={id}/>:<WeekFiveLab day={id}/>}
      {id!==4&&id!==5&&<CodeTrace lesson={lesson}/>}
    </SimulationStudio>
    <section className="panel py30-section"><h2>Đọc có chủ đích</h2><ul>{lesson.reads.map(r=><li key={r}>{r}</li>)}</ul></section>
    <section className="panel py30-section"><h2>Lab bằng Python trên máy</h2>
      <p>Để chạy trong trang, mở <code>Mo_Jupyter_Studio.command</code> rồi truy cập địa chỉ launcher in ra. Gói thực hành có notebook và dữ liệu UCI dùng khi chưa tải Kaggle.</p>
      <div className="py30-kernel" role="group" aria-label="Điều khiển kernel">
        <strong role="status">Kernel · {labels[status]}</strong>
        <button className="btn-primary" disabled={working||connected} onClick={()=>void runAction(connect)}>Kết nối</button>
        <button className="btn-outline" disabled={!(working&&connected)} onClick={()=>void terminalAction('interrupt')}>Interrupt</button>
        <button className="btn-outline" disabled={working||!connected} onClick={()=>void runAction(()=>terminalAction('restart'))}>Khởi động lại</button>
        <button className="btn-outline" disabled={working||!connected} onClick={()=>void runAction(save)}>Lưu vào py30/bai_lam/</button>
        <button className="btn-outline" disabled={working||!connected} onClick={()=>void runAction(load)}>Nạp bài đã lưu</button>
        {connected&&<a className="btn-outline" href={openInJupyterHref(hasSaved?workPath:modelPath)} target="_blank" rel="noreferrer">Mở trong Jupyter {hasSaved?'(bài đã lưu)':'(bản mẫu)'}</a>}
      </div>
      {notice&&<p role="status">{notice}</p>}
      {lesson.exercises.map(e=>{
        const run=runs[e.id],canRun=connected&&!working;
        return <article className="py30-exercise" key={e.id}>
          <h3>{e.title}</h3><p>Trạng thái: {progress.grades[e.id]===true?'Đạt':progress.grades[e.id]===false?'Chưa đạt':'Chưa chạy'}</p>
          <PythonEditor value={codes[e.id]} onChange={v=>edit(e.id,v)} ariaLabel={'Code '+e.title} rows={Math.min(14,Math.max(5,codes[e.id].split('\n').length))} disabled={working} onRun={()=>{if(canRun)void runAction(()=>runExercise(e));}}/>
          <button className="btn-primary" disabled={!canRun} onClick={()=>void runAction(()=>runExercise(e))}>Chạy và kiểm tra</button>
          {run?.error&&<p role="alert">{run.error}</p>}{run?.output.length?<Outputs outputs={run.output}/>:null}
          {run?.report&&<ul>{run.report.checks.map((c,i)=><li key={i}>{c.ok?'Đạt':'Chưa đạt'} · {c.name}. {c.detail}</li>)}</ul>}
          <details><summary>Gợi ý</summary><p>{e.hint}</p></details>
          {run?.attempted&&<details><summary>Lời giải tham khảo</summary><pre>{e.solution}</pre></details>}
        </article>;
      })}
    </section>
    <section className="panel py30-section"><h2>Dự án · {lesson.projectTitle}</h2><p>{lesson.project}</p><h2>Tổng kết</h2>
      <ol>{lesson.questions.map(q=><li key={q}>{q}</li>)}</ol><details><summary>Đáp án tự kiểm tra</summary><ol>{lesson.answers.map(a=><li key={a}>{a}</li>)}</ol></details>
      <label>Nhật ký<textarea rows={4} value={progress.journal} onChange={e=>change({journal:e.target.value.slice(0,10000)})}/></label>
    </section>
    <div className="py30-actions">{prev&&<button className="btn-outline" onClick={()=>openDay(id-1)}>Về ngày {id-1}</button>}{id<content.length&&<button className="btn-primary" onClick={()=>openDay(id+1)}>Tiếp ngày {id+1}</button>}</div>
  </>;
}

export function PythonTrack({day,openDay,onBusy}:{day?:number;openDay:(day?:number)=>void;onBusy:(busy:boolean)=>void}) {
  const [progress,setProgress]=useState(loadProgress),[message,setMessage]=useState('');
  useEffect(()=>{try{localStorage.setItem(storageKey,JSON.stringify(progress));}catch{setMessage('Không ghi được tiến độ trong trình duyệt. Bạn có thể xuất bản JSON.');}},[progress]);
  const lesson=day?content[day-1]:undefined;
  if(lesson)return <DayPage lesson={lesson} progress={progress.days[lesson.id]??emptyDay()} onBusy={onBusy} openDay={openDay}
    change={patch=>setProgress(old=>({version:1,days:{...old.days,[lesson.id]:{...(old.days[lesson.id]??emptyDay()),...patch}}}))}/>;
  if(day)return <section className="panel py30-section"><h1>Ngày {day} · Đang trong lộ trình</h1><p>Buổi này sẽ mở sau khi nội dung tuần trước được duyệt.</p><button className="btn-primary" onClick={()=>openDay()}>Quay lại lộ trình</button></section>;
  return <><header className="page-heading"><h1>Python → ML 30 ngày</h1><p>Dành cho bạn đã vững Python và mới học Machine Learning. Mỗi buổi 180 phút, từ mô phỏng đến tự cài thuật toán và thử trên dự án.</p></header>
    <AlgoAtlas openDay={d=>openDay(d)} available={d=>d<=content.length}/>
    {[['Tuần 1 · Hồi quy và tối ưu · Dự án House Prices',1,7],['Tuần 2 · Phân loại và đánh giá · Dự án Titanic',8,14],['Tuần 3 · Cây, tổ hợp và học không nhãn · Dự án Digit Recognizer',15,21],['Tuần 4 · Mạng nơ-ron và PyTorch · Dự án CNN nhiều seed',22,28],['Tuần 5 · Dự án tổng kết trên Kaggle',29,30]].map(([title,from,to])=><section className="panel py30-section" key={title as string}><h2>{title}</h2>
      <div className="py30-days">{content.filter(d=>d.id>=(from as number)&&d.id<=(to as number)).map(d=><button key={d.id} className="btn-outline" onClick={()=>openDay(d.id)}>Ngày {d.id} · {d.title} ({progress.days[d.id]?.blocks.filter(Boolean).length??0}/7)</button>)}</div>
    </section>)}
    <div className="py30-actions"><button className="btn-outline" onClick={()=>download('Tien_do_Python_ML_30_ngay.json',JSON.stringify(progress,null,2))}>Xuất tiến độ Python → ML</button>
      <label>Nhập tiến độ Python → ML <input type="file" accept="application/json,.json" onChange={async e=>{
        const file=e.currentTarget.files?.[0];e.currentTarget.value='';
        if(!file)return;
        try{if(file.size>1024*1024)throw Error('Tệp vượt quá 1 MB.');const incoming=progressFrom(JSON.parse(await file.text()));setProgress(old=>({version:1,days:{...old.days,...incoming.days}}));setMessage('Đã nhập tiến độ Python → ML.');}
        catch(error){setMessage(errorText(error));}
      }}/></label></div>{message&&<p role="status">{message}</p>}
  </>;
}
