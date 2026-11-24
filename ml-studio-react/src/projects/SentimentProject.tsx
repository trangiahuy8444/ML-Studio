import {useMemo,useState} from 'react';
import type {Route} from '../types';
import {Bars,Plot,ProjectFrame,Segmented,Stat,fmt,pct} from './ui';
import type {StepDef,ReverseDef} from './ui';

type Doc={text:string;pos:boolean};
// Phản hồi tự soạn theo phong cách UIT-VSFC (phản hồi sinh viên); không trích từ bộ dữ liệu gốc.
const TRAIN:Doc[]=[
  {pos:true,text:'thầy dạy rất hay và nhiệt tình'},{pos:true,text:'bài giảng dễ hiểu, ví dụ thực tế'},{pos:true,text:'cô giảng hay, trả lời câu hỏi rất kỹ'},
  {pos:true,text:'môn học bổ ích, em học được nhiều'},{pos:true,text:'phòng học rộng và mát'},{pos:true,text:'thầy rất tận tâm với sinh viên'},
  {pos:true,text:'slide đẹp, nội dung rõ ràng'},{pos:true,text:'bài tập hay giúp em hiểu bài'},{pos:true,text:'cô vui tính, lớp học sôi nổi'},
  {pos:true,text:'giảng viên nhiệt tình, chấm bài công bằng'},{pos:true,text:'em rất thích môn này'},{pos:true,text:'tài liệu đầy đủ và dễ hiểu'},
  {pos:true,text:'thầy giải thích kỹ, không bị khó hiểu'},{pos:true,text:'môn học không khó như em nghĩ'},{pos:true,text:'lớp học không nhàm chán chút nào'},
  {pos:true,text:'cô dạy hay, em hiểu bài ngay trên lớp'},{pos:true,text:'máy chiếu tốt, âm thanh rõ'},{pos:true,text:'đề thi vừa sức, sát nội dung học'},
  {pos:false,text:'thầy dạy nhanh quá, khó hiểu'},{pos:false,text:'bài giảng nhàm chán, đọc slide'},{pos:false,text:'phòng học nóng, máy chiếu mờ'},
  {pos:false,text:'môn học khó và không thực tế'},{pos:false,text:'cô ít trả lời câu hỏi của sinh viên'},{pos:false,text:'bài tập quá nhiều, không kịp làm'},
  {pos:false,text:'tài liệu thiếu và khó hiểu'},{pos:false,text:'thầy hay đến muộn'},{pos:false,text:'chấm bài không công bằng'},
  {pos:false,text:'lớp đông, ồn ào, không nghe rõ'},{pos:false,text:'đề thi quá khó so với bài học'},{pos:false,text:'em không thích cách dạy này'},
  {pos:false,text:'giảng viên không nhiệt tình'},{pos:false,text:'bài giảng không hay, em buồn ngủ'},{pos:false,text:'slide sơ sài, chữ nhỏ'},
  {pos:false,text:'âm thanh tệ, không nghe được'},{pos:false,text:'môn học không bổ ích lắm'},{pos:false,text:'thầy giảng không rõ ràng'},
];
const TEST:Doc[]=[
  {pos:true,text:'cô dạy dễ hiểu và nhiệt tình'},{pos:false,text:'phòng học nóng và ồn ào'},{pos:true,text:'bài tập không khó, rất bổ ích'},
  {pos:false,text:'thầy dạy không hay lắm'},{pos:true,text:'nội dung rõ ràng, ví dụ hay'},{pos:false,text:'tài liệu sơ sài, khó hiểu'},
  {pos:false,text:'em không hiểu bài vì thầy dạy nhanh'},{pos:true,text:'lớp học vui, không áp lực'},{pos:false,text:'máy chiếu mờ, slide chữ nhỏ'},
  {pos:true,text:'em thích cách cô giảng bài'},{pos:false,text:'đề thi không sát nội dung học'},{pos:true,text:'giảng viên tận tâm, trả lời kỹ'},
];
const NEG=new Set(['không','chẳng','chưa','ít']);
function tokens(s:string,bigrams:boolean){
  const w=s.toLowerCase().normalize('NFC').replace(/[^\p{L}\s]/gu,' ').split(/\s+/).filter(Boolean);
  if(!bigrams)return w;
  // Ghép từ phủ định với từ ngay sau nó thành một "từ" mới, ví dụ không_hay.
  const out:string[]=[];for(let i=0;i<w.length;i++){if(NEG.has(w[i])&&i+1<w.length){out.push(w[i]+'_'+w[i+1]);i++}else out.push(w[i])}return out;
}
function trainNB(data:Doc[],bigrams:boolean){
  const pos=new Map<string,number>(),neg=new Map<string,number>();let np=0,nn=0;
  for(const d of data)for(const t of tokens(d.text,bigrams)){const m=d.pos?pos:neg;m.set(t,(m.get(t)??0)+1);d.pos?np++:nn++}
  const V=new Set([...pos.keys(),...neg.keys()]).size;
  const weight=(t:string)=>(pos.has(t)||neg.has(t))?Math.log(((pos.get(t)??0)+1)/(np+V))-Math.log(((neg.get(t)??0)+1)/(nn+V)):0;
  const score=(s:string)=>tokens(s,bigrams).reduce((a,t)=>a+weight(t),0);
  return {weight,score,size:V};
}
// Embedding 2 chiều tự đặt để minh họa: trục ngang = tích cực ↔ tiêu cực, trục dọc = nói về người dạy ↔ về cơ sở vật chất.
const EMB:Record<string,[number,number]>={
  'hay':[1.2,.9],'tuyệt vời':[2.0,.55],'xuất sắc':[2.0,1.15],'dễ hiểu':[1.0,1.45],'nhiệt tình':[1.2,2.05],'tận tâm':[2.0,1.75],'tâm huyết':[2.0,2.25],'bổ ích':[1.1,.3],'thú vị':[1.9,-.05],
  'rộng':[1.0,-1.3],'mát':[1.0,-1.9],'sạch sẽ':[1.9,-1.6],'hiện đại':[1.9,-1.05],
  'tệ':[-1.9,.2],'dở':[-1.2,.55],'nhàm chán':[-2.0,.8],'khó hiểu':[-1.2,1.3],'qua loa':[-2.0,1.6],'thờ ơ':[-1.3,2.1],
  'nóng':[-1.2,-1.3],'ồn ào':[-2.0,-1.0],'chật':[-1.3,-1.95],'mờ':[-2.1,-1.55],'xuống cấp':[-2.1,-2.15],
};
const PHRASES=Object.keys(EMB).sort((a,b)=>b.length-a.length);
function embedScore(s:string){
  let text=' '+s.toLowerCase().normalize('NFC').replace(/[^\p{L}\s]/gu,' ').replace(/\s+/g,' ')+' ';const found:{w:string;neg:boolean}[]=[];
  for(const p of PHRASES){const re=new RegExp(`(\\s(không|chẳng|chưa)(\\s\\p{L}+)?)?\\s${p}(?=\\s)`,'gu');text=text.replace(re,m=>{found.push({w:p,neg:/(không|chẳng|chưa)/u.test(m.replace(p,''))});return ' # '})}
  const v=found.length?found.reduce((a,f)=>a+(f.neg?-1:1)*EMB[f.w][0],0)/found.length:0;
  return {v,found};
}
function Chips({text,model,bigrams}:{text:string;model:ReturnType<typeof trainNB>;bigrams:boolean}){
  return <p className="word-chips">{tokens(text,bigrams).map((t,i)=>{const w=model.weight(t);return <span key={i} title={`${w>=0?'+':''}${fmt(w)}`} style={{['--w' as string]:Math.min(1,Math.abs(w)/2)}} className={w>0.05?'good':w<-0.05?'pos':'zero'}>{t.replace('_',' ')}</span>})}</p>;
}

export function SentimentProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [guess,setGuess]=useState<(boolean|null)[]>(()=>Array(6).fill(null));
  const [bigrams,setBigrams]=useState(false);
  const [word,setWord]=useState('hay');
  const [draft,setDraft]=useState('thầy dạy tuyệt vời, phòng học hơi chật');
  const uni=useMemo(()=>trainNB(TRAIN,false),[]),bi=useMemo(()=>trainNB(TRAIN,true),[]);
  const model=bigrams?bi:uni;
  const acc=(m:ReturnType<typeof trainNB>)=>TEST.filter(d=>(m.score(d.text)>0)===d.pos).length/TEST.length;
  const done=guess.every(g=>g!==null),mine=guess.filter((g,i)=>g===TEST[i].pos).length;
  const A='bài giảng hay nhưng phòng học không tốt',B='bài giảng không hay nhưng phòng học tốt';
  const bag=(s:string)=>{const m=new Map<string,number>();tokens(s,false).forEach(t=>m.set(t,(m.get(t)??0)+1));return [...m].sort((a,b)=>a[0].localeCompare(b[0]))};
  const nearest=Object.entries(EMB).filter(([w])=>w!==word).map(([w,p])=>({w,d:Math.hypot(p[0]-EMB[word][0],p[1]-EMB[word][1])})).sort((a,b)=>a.d-b.d).slice(0,4);
  const unseen='giảng viên tâm huyết, bài học thú vị';
  const e5=embedScore(draft),s5=bi.score(draft);

  const steps:StepDef[]=[
    {label:'Em đọc cảm xúc',title:'Khen hay chê? Em quyết định trước.',
      intro:<p>Sáu phản hồi của sinh viên về môn học. Em biết ngay chúng khen hay chê, nhưng thử nghĩ: em dựa vào <b>từ nào</b>?</p>,
      controls:<div className="stat-row">{done?<Stat label="Em đúng" value={`${mine}/6`}/>:<Stat label="Đã trả lời" value={`${guess.filter(g=>g!==null).length}/6`}/>}</div>,
      visual:<ol className="mail-list">{TEST.slice(0,6).map((d,i)=><li key={i} className={done?(guess[i]===d.pos?'right':'wrong'):''}><p>“{d.text}”</p><div className="segmented small">{[[true,'Khen'],[false,'Chê']].map(([v,t])=><button key={String(v)} className={guess[i]===v?'on':''} onClick={()=>setGuess(g=>g.map((x,j)=>j===i?v as boolean:x))}>{t as string}</button>)}</div></li>)}</ol>,
      takeaway:'Em hiểu "không hay" là chê dù có chữ "hay". Con người đọc cả cụm từ và ngữ cảnh. Phần còn lại của dự án là xem máy cần gì để làm được điều đó.'},
    {label:'Túi từ',title:'Hai câu ngược nghĩa, cùng một túi.',
      intro:<p>Túi từ đếm mỗi từ xuất hiện bao nhiêu lần và <b>bỏ qua thứ tự</b>. Nhìn hai câu dưới đây: nghĩa trái ngược, nhưng túi từ giống hệt nhau.</p>,
      controls:<><p className="observation">A: “{A}”<br/>B: “{B}”</p><p className="small-note">Túi của A {JSON.stringify(bag(A))===JSON.stringify(bag(B))?'trùng khớp hoàn toàn':'khác'} túi của B.</p></>,
      visual:<div className="table-wrap"><table><thead><tr><th>Từ</th><th>Câu A</th><th>Câu B</th></tr></thead><tbody>{bag(A).map(([w,c])=><tr key={w}><td>{w}</td><td>{c}</td><td>{bag(B).find(x=>x[0]===w)?.[1]??0}</td></tr>)}</tbody></table></div>,
      takeaway:'Túi từ rẻ và hiệu quả đến bất ngờ, nhưng mù với thứ tự. Mô hình nào chỉ dùng túi từ sẽ cho A và B cùng một điểm, chắc chắn sai một câu.'},
    {label:'Bigram cứu "không hay"',title:'Ghép từ phủ định với từ đứng sau.',
      intro:<p>Bật <b>bigram phủ định</b>: "không hay" thành một từ mới <code>không_hay</code>. Xem độ chính xác trên 12 phản hồi kiểm tra và màu của từng từ.</p>,
      controls:<><Segmented label="Cách tách từ" options={[['uni','Từng từ'],['bi','+ bigram phủ định']]} value={bigrams?'bi':'uni'} onChange={v=>setBigrams(v==='bi')}/>
        <div className="stat-row"><Stat label="Từng từ" value={pct(acc(uni))}/><Stat label="+ bigram" value={pct(acc(bi))} tone={acc(bi)>acc(uni)?'good':undefined}/></div></>,
      visual:<ul className="pred-list sentiment">{TEST.map((d,i)=>{const s=model.score(d.text);return <li key={i} className={(s>0)===d.pos?'right':'wrong'}><span>{s>0?'Khen':'Chê'}</span><Chips text={d.text} model={model} bigrams={bigrams}/></li>})}</ul>,
      takeaway:'Một mẹo nhỏ về biểu diễn đã sửa được nhiều lỗi mà không cần đổi thuật toán. Bài học: cách biến chữ thành số quan trọng ngang với mô hình.'},
    {label:'Embedding',title:'Từ gần nghĩa nằm gần nhau.',
      intro:<p>Mỗi từ thành một điểm trong không gian. Các từ dùng trong ngữ cảnh giống nhau thì nằm gần nhau (ý tưởng của Word2Vec). Bấm một từ để xem hàng xóm của nó.</p>,
      controls:<><div className="chip-row">{Object.keys(EMB).map(w=><button key={w} className={'chip'+(w===word?' on':'')} onClick={()=>setWord(w)}>{w}</button>)}</div>
        <p className="small-note">Gần "{word}" nhất: {nearest.map(n=>n.w).join(', ')}.</p>
        <p className="observation">Câu “{unseen}” không có từ nào trong dữ liệu huấn luyện. Túi từ chỉ thấy các từ trung tính như "giảng viên", "bài học" nên cho điểm <b>{fmt(bi.score(unseen))}</b>, gần như đoán mò; embedding cho điểm <b>{fmt(embedScore(unseen).v)}</b> vì "tâm huyết" gần "tận tâm", "thú vị" gần "hay".</p></>,
      visual:<><Plot x={[-2.5,3.1]} y={[-2.5,2.6]} xLabel="← tiêu cực · tích cực →" yLabel="← cơ sở vật chất · người dạy →" onPick={(x,y)=>{const best=Object.entries(EMB).sort((a,b)=>Math.hypot(a[1][0]-x,a[1][1]-y)-Math.hypot(b[1][0]-x,b[1][1]-y))[0];setWord(best[0])}}>{(sx,sy)=><>
        {nearest.map(n=><line key={n.w} x1={sx(EMB[word][0])} y1={sy(EMB[word][1])} x2={sx(EMB[n.w][0])} y2={sy(EMB[n.w][1])} stroke="#ffd173" strokeWidth={1.3}/>)}
        {Object.entries(EMB).map(([w,p])=><g key={w}><circle cx={sx(p[0])} cy={sy(p[1])} r={w===word?7:4.5} fill={p[0]>0?'#84e0a8':'#f28db9'}/><text x={sx(p[0])+8} y={sy(p[1])+4} className="pt-label">{w}</text></g>)}</>}</Plot>
        <p className="chart-caption">Embedding thật có hàng trăm chiều và được học từ hàng triệu câu; ở đây chỉ 2 chiều tự đặt để dễ nhìn.</p></>,
      takeaway:'Embedding cho mô hình khả năng khái quát sang từ chưa gặp. PhoBERT (bài 09 trong thư mục 04 của em) đi xa hơn: biểu diễn của một từ còn thay đổi theo cả câu xung quanh.'},
    {label:'Tự thử & giới hạn',title:'Viết bình luận của em.',
      intro:<p>Gõ một phản hồi. So hai cách: Naive Bayes + bigram (học từ 36 câu) và điểm embedding. Thử câu mỉa mai như <i>"hay quá, cả lớp ngủ ngon"</i>.</p>,
      controls:<><textarea rows={4} value={draft} onChange={e=>setDraft(e.target.value)} aria-label="Phản hồi của em"/>
        <div className="stat-row"><Stat label="Naive Bayes" value={s5>0?'Khen':s5<0?'Chê':'Không rõ'} tone={s5>0?'good':s5<0?'bad':'warn'}/><Stat label="Embedding" value={e5.v>0?'Khen':e5.v<0?'Chê':'Không rõ'} tone={e5.v>0?'good':e5.v<0?'bad':'warn'}/></div></>,
      visual:<><h4 className="col-title">Naive Bayes nhìn thấy</h4><Chips text={draft} model={bi} bigrams/>
        <h4 className="col-title">Embedding nhận ra</h4>{e5.found.length?<Bars rows={e5.found.map(f=>({label:(f.neg?'không ':'')+f.w,value:(f.neg?-1:1)*EMB[f.w][0],color:(f.neg?-1:1)*EMB[f.w][0]>0?'#84e0a8':'#f28db9'}))} format={v=>fmt(v,1)}/>:<p className="small-note">Không có từ nào trong bảng embedding nhỏ này.</p>}</>,
      takeaway:'Cả hai đều thua câu mỉa mai, vì cần hiểu ngữ cảnh và ý định. Đó là lý do các mô hình ngôn ngữ lớn dùng attention để đọc cả câu cùng lúc, chủ đề của ngày 25.'},
  ];
  const reverse:ReverseDef={layers:[
    {from:4,label:'Dùng sản phẩm',title:'Hai bộ đoán cảm xúc đã chạy.',intro:<p>Gõ một phản hồi và xem hai bộ đoán trả lời. Thử câu khen, câu chê, câu có chữ “không”, và câu mỉa mai.</p>,takeaway:'Cả hai đúng với câu thẳng thắn và lúng túng với câu mỉa mai. Tháo ra để biết vì sao.'},
    {from:2,label:'Mở hộp',title:'Bên trong: điểm của từng từ.',intro:<p>Mở bộ đoán Naive Bayes: mỗi từ, hoặc cặp “không + từ”, có một điểm. Tắt bigram để thấy bộ phận đó đóng góp gì.</p>,takeaway:'Bộ phận bigram nhỏ nhưng sửa được nhiều lỗi “không hay”. Tháo nó ra, độ chính xác tụt rõ.'},
    {from:1,label:'Lớp biểu diễn',title:'Tháo đến tận đáy: túi từ.',intro:<p>Ở đáy sản phẩm, câu được biến thành túi từ và mất thứ tự.</p>,takeaway:'Điểm yếu gốc rễ nằm ở đây: hai câu ngược nghĩa có cùng một túi từ.'},
    {from:3,label:'Thay biểu diễn',title:'Đổi túi từ bằng embedding.',intro:<p>Thay lớp biểu diễn: mỗi từ thành một điểm, từ gần nghĩa nằm gần nhau.</p>,takeaway:'Đổi lớp biểu diễn giúp đoán đúng cả câu có từ chưa gặp.'},
    {from:0,label:'So với con người',title:'Em là bộ đoán thứ ba.',intro:<p>Em tự đoán 6 phản hồi rồi so với máy.</p>,takeaway:'Em đọc được ngữ cảnh mà túi từ không thấy. Khoảng cách đó là thứ attention và các mô hình lớn cố lấp.'},
  ]};
  return <ProjectFrame id="sentiment" steps={steps} reverse={reverse} navigate={navigate}/>;
}
