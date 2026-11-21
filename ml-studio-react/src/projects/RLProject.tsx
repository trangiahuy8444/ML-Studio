import {useMemo,useState} from 'react';
import type {Route} from '../types';
import {Bars,LineChart,PALETTE,PlayerBar,ProjectFrame,Segmented,Slider,Stat,fmt,pct,rng,usePlayer} from './ui';
import type {StepDef} from './ui';

const TRUE_P=[.25,.45,.62,.38];
const MACHINES=['A','B','C','D'];

function egreedy(eps:number,pulls:number,seed:number){
  const r=rng(seed),n=[0,0,0,0],q=[0,0,0,0];const hist:{arm:number;reward:number;q:number[];total:number}[]=[];let total=0;
  for(let t=0;t<pulls;t++){
    const arm=r()<eps?Math.floor(r()*4):q.indexOf(Math.max(...q));
    const reward=r()<TRUE_P[arm]?1:0;n[arm]++;q[arm]+=(reward-q[arm])/n[arm];total+=reward;
    hist.push({arm,reward,q:[...q],total});
  }
  return hist;
}

/* ---------- Thế giới lưới: vách đá trơn trượt ---------- */
const R=4,C=6,START=[3,0],GOAL=[3,5];
const isPit=(r:number,c:number)=>r===3&&c>0&&c<5;
const MOVES=[[-1,0],[0,1],[1,0],[0,-1]],ARROWS=['↑','→','↓','←'];
function qlearn(episodes:number,penalty:number,slip:number,seed=4){
  // ε giảm dần từ 0,5 về 0,05: đầu khám phá nhiều, cuối khai thác.
  const r=rng(seed),Q=Array.from({length:R*C},()=>[0,0,0,0]);const snaps:{ep:number;Q:number[][];ret:number}[]=[{ep:0,Q:Q.map(x=>[...x]),ret:0}];let avg=0;
  for(let ep=1;ep<=episodes;ep++){
    let [y,x]=START,ret=0;const eps=Math.max(.05,.5*(1-ep/episodes));
    for(let s=0;s<200;s++){
      const st=y*C+x,a=r()<eps?Math.floor(r()*4):Q[st].indexOf(Math.max(...Q[st]));
      const real=r()<slip?Math.floor(r()*4):a;const ny=Math.min(R-1,Math.max(0,y+MOVES[real][0])),nx=Math.min(C-1,Math.max(0,x+MOVES[real][1]));
      let reward=penalty,end=false;if(isPit(ny,nx)){reward=-1;end=true}else if(ny===GOAL[0]&&nx===GOAL[1]){reward=1;end=true}
      const target=reward+(end?0:.95*Math.max(...Q[ny*C+nx]));Q[st][a]+=.1*(target-Q[st][a]);ret+=reward;
      if(end)break;[y,x]=[ny,nx];
    }
    avg=avg*.95+ret*.05;
    if(ep%50===0)snaps.push({ep,Q:Q.map(q=>[...q]),ret:avg});
  }
  return snaps;
}
function greedyPath(Q:number[][]){const path:[number,number][]=[[START[0],START[1]]];let [y,x]=START;for(let s=0;s<30;s++){const a=Q[y*C+x].indexOf(Math.max(...Q[y*C+x]));y=Math.min(R-1,Math.max(0,y+MOVES[a][0]));x=Math.min(C-1,Math.max(0,x+MOVES[a][1]));path.push([y,x]);if(isPit(y,x)||(y===GOAL[0]&&x===GOAL[1]))break}return path}
function Grid({Q}:{Q:number[][]}){
  const S=64,path=greedyPath(Q),vals=Q.map(q=>Math.max(...q)),hi=Math.max(.01,...vals.map(Math.abs));
  return <svg className="rl-grid" viewBox={`0 0 ${C*S} ${R*S}`}>
    {Array.from({length:R*C},(_,i)=>{const y=Math.floor(i/C),x=i%C,pit=isPit(y,x),goal=y===GOAL[0]&&x===GOAL[1],start=y===START[0]&&x===START[1],v=vals[i];
      return <g key={i}><rect x={x*S+1} y={y*S+1} width={S-2} height={S-2} rx={6} fill={pit?'rgba(242,141,185,.55)':goal?'rgba(132,224,168,.6)':v>=0?`rgba(66,211,229,${.08+.5*v/hi})`:`rgba(242,141,185,${.08+.4*-v/hi})`} stroke="var(--line)"/>
        <text x={x*S+S/2} y={y*S+S/2+(pit||goal||start?6:9)} textAnchor="middle" className={pit||goal||start?'rl-cell rl-label':'rl-cell'}>{pit?'Vực':goal?'Đích':start?'Bắt đầu':(Q[i].some(q=>q!==0)?ARROWS[Q[i].indexOf(Math.max(...Q[i]))]:'·')}</text></g>})}
    <polyline points={path.map(([y,x])=>`${x*S+S/2},${y*S+S/2}`).join(' ')} fill="none" stroke="#ffd173" strokeWidth={4} strokeLinejoin="round" strokeDasharray="2 7" strokeLinecap="round"/>
  </svg>;
}

export function RLProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [step,setStep]=useState(0);
  const [plays,setPlays]=useState<{arm:number;reward:number}[]>([]);
  const [eps,setEps]=useState(.1);
  const run=useMemo(()=>egreedy(eps,300,42),[eps]);
  const player=usePlayer(run.length,30);
  const compare=useMemo(()=>[0,.01,.1,.3].map(e=>{const sums=Array(500).fill(0);for(let s=0;s<200;s++){egreedy(e,500,1000+s).forEach((h,t)=>sums[t]+=h.reward)}
    const avg=sums.map(v=>v/200),smooth=avg.map((_,t)=>{const a=avg.slice(Math.max(0,t-20),t+1);return a.reduce((x,y)=>x+y,0)/a.length});return {e,smooth,total:avg.reduce((a,b)=>a+b,0)}}),[]);
  const [penalty,setPenalty]=useState(-.04);
  const snaps=useMemo(()=>qlearn(3000,-.04,.2),[]);
  const snaps5=useMemo(()=>qlearn(3000,penalty,.2),[penalty]);
  const [snapI,setSnapI]=useState(snaps.length-1);

  const mine=MACHINES.map((_,a)=>{const p=plays.filter(x=>x.arm===a);return {n:p.length,avg:p.length?p.reduce((s,x)=>s+x.reward,0)/p.length:0}});
  const total=plays.reduce((s,x)=>s+x.reward,0),finished=plays.length>=30;
  const h=run[player.frame];
  const bestShare=run.slice(0,player.frame+1).filter(x=>x.arm===2).length/(player.frame+1);
  const finalPath=greedyPath(snaps5.at(-1)!.Q),route=finalPath.some(([y,x])=>y===2&&x>0&&x<5)?2:finalPath.some(([y,x])=>y===1&&x>0&&x<5)?1:0;

  const steps:StepDef[]=[
    {label:'Em chơi thử',title:'Bốn máy, em có 30 lượt.',
      intro:<p>Mỗi máy trả thưởng với một xác suất bí mật. Em muốn tổng thưởng cao nhất. Thử máy mới để biết thêm, hay chơi tiếp máy đang có vẻ tốt?</p>,
      controls:<><div className="stat-row"><Stat label="Lượt" value={`${plays.length}/30`}/><Stat label="Tổng thưởng" value={total}/></div><button className="btn-outline" onClick={()=>setPlays([])}>Chơi lại</button>
        {finished&&<p className="observation">Xác suất thật: {MACHINES.map((m,i)=>`${m} ${pct(TRUE_P[i])}`).join(' · ')}. Chơi toàn máy C, trung bình em được khoảng {Math.round(30*TRUE_P[2])} điểm.</p>}</>,
      visual:<div className="slot-row">{MACHINES.map((m,a)=><button key={m} className="slot" disabled={finished} onClick={()=>setPlays(p=>[...p,{arm:a,reward:Math.random()<TRUE_P[a]?1:0}])}>
        <strong>Máy {m}</strong><span className="slot-last">{plays.filter(x=>x.arm===a).slice(-8).map((x,i)=><i key={i} className={x.reward?'win':''}/>)}</span><small>{mine[a].n} lượt · TB {fmt(mine[a].avg)}</small></button>)}</div>,
      takeaway:'Đây là thế lưỡng nan khám phá–khai thác: chỉ khai thác thì có thể kẹt ở máy tầm thường; chỉ khám phá thì phí lượt vào máy tệ. Mọi tác tử học tăng cường đều phải cân hai việc này.'},
    {label:'Chiến lược ε-greedy',title:'Phần lớn chọn máy tốt nhất, thỉnh thoảng thử ngẫu nhiên.',
      intro:<p>Với xác suất <b>ε</b> máy chọn ngẫu nhiên (khám phá), còn lại chọn máy có điểm ước lượng cao nhất (khai thác). Thanh màu là ước lượng, vạch trắng là xác suất thật.</p>,
      controls:<><Slider label="ε (tỉ lệ khám phá)" value={eps} min={0} max={.5} step={.01} onChange={v=>{setEps(v);player.reset()}} format={v=>fmt(v)}/>
        <PlayerBar player={player} count={run.length} label="Lượt"/>
        <div className="stat-row"><Stat label="Tổng thưởng" value={h.total}/><Stat label="Chọn máy C" value={pct(bestShare)}/></div></>,
      visual:<div className="q-bars">{MACHINES.map((m,a)=><div key={m}><span>Máy {m}</span><div className="q-track"><i style={{width:pct(h.q[a]),background:PALETTE[a]}}/><b style={{left:pct(TRUE_P[a])}}/></div><small>{fmt(h.q[a])}</small></div>)}
        <LineChart height={160} xLabel="Lượt" yLabel="Tổng thưởng" series={[{name:'Tổng thưởng',values:run.slice(0,player.frame+1).map(x=>x.total),color:'#ffd173'}]}/></div>,
      takeaway:'Với ε = 0, máy có thể kẹt ở máy đầu tiên cho thưởng. Một chút ngẫu nhiên giúp nó phát hiện máy C, rồi phần lớn thời gian khai thác máy C.'},
    {label:'So sánh ε',title:'200 ván, mỗi ván 500 lượt.',
      intro:<p>Một ván có thể may hay rủi. Để công bằng, ta chạy 200 ván cho mỗi giá trị ε và lấy trung bình phần thưởng ở từng lượt.</p>,
      controls:<Bars rows={compare.map((c,i)=>({label:`ε = ${String(c.e).replace('.',',')}`,value:c.total,color:PALETTE[i]}))} format={v=>fmt(v,0)+' điểm'}/>,
      visual:<LineChart height={300} xLabel="Lượt" yLabel="Thưởng trung bình mỗi lượt" series={compare.map((c,i)=>({name:`ε = ${String(c.e).replace('.',',')}`,values:c.smooth,color:PALETTE[i]}))}/>,
      takeaway:'ε = 0 học nhanh nhưng hay kẹt; ε lớn tìm ra máy tốt nhưng phí lượt mãi về sau. Giá trị vừa phải thắng về lâu dài. Thực tế người ta còn giảm ε dần theo thời gian.'},
    {label:'Robot tìm đường',title:'Q-learning: học giá trị của từng ô.',
      intro:<p>Robot muốn tới <b>Đích</b> (+1) và tránh <b>Vực</b> (−1). Sàn trơn: 20% số bước trượt sang hướng ngẫu nhiên. Mũi tên là hướng tốt nhất robot đã học, đường vàng là lộ trình nó sẽ đi.</p>,
      controls:<><Slider label="Số lượt chơi đã học" value={snapI} min={0} max={snaps.length-1} onChange={setSnapI} format={i=>String(snaps[i].ep)}/>
        <div className="stat-row"><Stat label="Thưởng TB gần đây" value={fmt(snaps[snapI].ret)}/></div>
        <p className="small-note">Mỗi bước robot cập nhật: Q(ô, hướng) ← Q + α·(thưởng + γ·giá trị ô kế − Q). Không ai chỉ đường, nó tự học từ thưởng phạt.</p></>,
      visual:<Grid Q={snaps[snapI].Q}/>,
      takeaway:'Ban đầu robot đi lung tung và rơi vực nhiều lần. Giá trị lan dần ngược từ ô Đích về điểm xuất phát, giống gradient lan ngược trong mạng nơ-ron.'},
    {label:'Phần thưởng định hình hành vi',title:'Đổi luật chơi, robot đổi tính cách.',
      intro:<p>Mỗi bước đi bị trừ một chút điểm. Phạt nặng thì robot vội, đi sát mép vực cho nhanh. Phạt nhẹ thì robot đi vòng xa vực cho an toàn.</p>,
      controls:<><Segmented label="Phạt mỗi bước" options={[[0,'0'],[-.01,'−0,01'],[-.04,'−0,04'],[-.15,'−0,15']]} value={penalty} onChange={setPenalty}/>
        <div className="stat-row"><Stat label="Lộ trình" value={['Vòng xa nhất','Cách vực một hàng','Sát mép vực'][route]} tone={route===2?'warn':'good'}/><Stat label="Số bước" value={finalPath.length-1}/></div></>,
      visual:<Grid Q={snaps5.at(-1)!.Q}/>,
      takeaway:'Robot làm đúng điều ta thưởng, không phải điều ta muốn. Thiết kế phần thưởng là phần khó nhất của học tăng cường, và cũng là vấn đề lớn khi huấn luyện các hệ AI thật.'},
  ];
  return <ProjectFrame id="rl" steps={steps} step={step} setStep={setStep} navigate={navigate}/>;
}
