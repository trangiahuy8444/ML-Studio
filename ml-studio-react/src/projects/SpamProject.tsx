import {useMemo,useState} from 'react';
import type {Route} from '../types';
import {Bars,Confusion,ProjectFrame,Slider,Stat,fmt,pct} from './ui';
import type {StepDef,ReverseDef} from './ui';

type Mail={text:string;spam:boolean};
// Email tự soạn để minh họa; không lấy từ hộp thư thật.
const TRAIN:Mail[]=[
  {spam:true,text:'Chúc mừng bạn đã trúng thưởng iPhone miễn phí, bấm vào link để nhận quà ngay'},
  {spam:true,text:'Khuyến mãi sốc giảm giá 90% chỉ hôm nay, mua ngay kẻo lỡ'},
  {spam:true,text:'Bạn được chọn nhận khoản vay không lãi suất, đăng ký ngay hôm nay'},
  {spam:true,text:'Kiếm tiền online tại nhà thu nhập 50 triệu mỗi tháng, bấm link đăng ký'},
  {spam:true,text:'Tài khoản của bạn bị khóa, xác nhận mật khẩu tại link này ngay'},
  {spam:true,text:'Nhận quà miễn phí khi giới thiệu bạn bè, ưu đãi có hạn'},
  {spam:true,text:'Giảm cân cấp tốc không cần ăn kiêng, mua ngay giảm giá 50%'},
  {spam:true,text:'Trúng thưởng tiền mặt 100 triệu, gửi thông tin tài khoản để nhận thưởng'},
  {spam:true,text:'Ưu đãi độc quyền hôm nay: mua một tặng một, bấm vào link'},
  {spam:true,text:'Đầu tư tiền ảo lợi nhuận gấp đôi mỗi tuần, đăng ký miễn phí'},
  {spam:true,text:'Bạn có một gói quà chưa nhận, xác nhận địa chỉ ngay để nhận quà'},
  {spam:true,text:'Vay tiền nhanh không cần thế chấp, giải ngân trong ngày'},
  {spam:true,text:'Sale cuối năm giảm giá toàn bộ cửa hàng, mua ngay hôm nay'},
  {spam:true,text:'Click link nhận voucher miễn phí 500 nghìn, số lượng có hạn'},
  {spam:true,text:'Cơ hội kiếm tiền dễ dàng, không cần kinh nghiệm, thu nhập cao'},
  {spam:false,text:'Chiều nay nhóm họp lúc 3 giờ ở phòng 204 để chốt báo cáo'},
  {spam:false,text:'Em gửi thầy bài tập tuần này, nhờ thầy xem giúp phần code'},
  {spam:false,text:'Mai mình đi ăn trưa nhé, quán cơm gần trường'},
  {spam:false,text:'Lịch thi cuối kỳ đã cập nhật trên cổng thông tin sinh viên'},
  {spam:false,text:'Cảm ơn bạn đã gửi tài liệu, mình đọc xong sẽ phản hồi'},
  {spam:false,text:'Báo cáo tiến độ dự án tuần này đính kèm trong file'},
  {spam:false,text:'Con nhớ mặc áo ấm, tối nay trời lạnh đấy'},
  {spam:false,text:'Nhắc lịch họp phụ huynh vào sáng thứ bảy tuần sau'},
  {spam:false,text:'Mình đã sửa lỗi trong notebook, bạn kéo bản mới về nhé'},
  {spam:false,text:'Hóa đơn tiền điện tháng này đã được thanh toán thành công'},
  {spam:false,text:'Thầy dời buổi học sang thứ năm vì trùng lịch hội thảo'},
  {spam:false,text:'Gửi bạn slide thuyết trình, góp ý giúp mình phần kết luận'},
  {spam:false,text:'Tối nay cả nhà ăn cơm lúc 7 giờ nhé'},
  {spam:false,text:'Nhóm mình chia việc: Lan làm dữ liệu, Minh viết báo cáo'},
  {spam:false,text:'Đơn hàng sách của bạn sẽ được giao vào ngày mai'},
];
const TEST:Mail[]=[
  {spam:true,text:'Nhận ngay quà tặng miễn phí, bấm link xác nhận trong hôm nay'},
  {spam:false,text:'Nhắc bạn nộp bài tập trước thứ sáu nhé'},
  {spam:true,text:'Giảm giá 70% cho khách hàng thân thiết, mua ngay'},
  {spam:false,text:'Mình gửi file dữ liệu cho dự án, bạn kiểm tra giúp'},
  {spam:true,text:'Vay tiền online lãi suất 0%, đăng ký nhận tiền ngay'},
  {spam:false,text:'Thầy gửi điểm giữa kỳ, các em xem trên cổng thông tin'},
  {spam:true,text:'Bạn đã trúng thưởng, cung cấp số tài khoản để nhận tiền'},
  {spam:false,text:'Hôm nay cửa hàng sách giảm giá, mình mua cho bạn một cuốn nhé'},
  {spam:false,text:'Họp nhóm online tối nay lúc 8 giờ, link gửi sau'},
  {spam:true,text:'Cơ hội đầu tư lợi nhuận cao, thu nhập thụ động mỗi tháng'},
];
const tokens=(s:string)=>s.toLowerCase().normalize('NFC').replace(/[^\p{L}\p{N}%\s]/gu,' ').split(/\s+/).filter(Boolean);

interface Model {weight:(w:string)=>number;prior:number;spamCount:Map<string,number>;hamCount:Map<string,number>;vocab:string[]}
function trainNB(data:Mail[],alpha:number):Model{
  const spamCount=new Map<string,number>(),hamCount=new Map<string,number>();let ns=0,nh=0,ms=0;
  for(const m of data){const c=m.spam?spamCount:hamCount;if(m.spam)ms++;for(const t of tokens(m.text)){c.set(t,(c.get(t)??0)+1);m.spam?ns++:nh++}}
  const vocab=[...new Set([...spamCount.keys(),...hamCount.keys()])],V=vocab.length;
  const weight=(w:string)=>Math.log(((spamCount.get(w)??0)+alpha)/(ns+alpha*V))-Math.log(((hamCount.get(w)??0)+alpha)/(nh+alpha*V));
  return {weight,prior:Math.log(ms/(data.length-ms)),spamCount,hamCount,vocab};
}
const score=(m:Model,text:string)=>m.prior+tokens(text).reduce((s,t)=>s+m.weight(t),0);
const prob=(m:Model,text:string)=>1/(1+Math.exp(-score(m,text)));

function Chips({model,text}:{model:Model;text:string}){
  return <p className="word-chips">{tokens(text).map((t,i)=>{const w=model.weight(t);return <span key={i} title={`${t}: ${w>=0?'+':''}${fmt(w)}`} style={{['--w' as string]:Math.min(1,Math.abs(w)/2.5)}} className={w>0.05?'pos':w<-0.05?'neg':'zero'}>{t}</span>})}</p>;
}

export function SpamProject({navigate}:{navigate:(r:Route,anchor?:string)=>void}){
  const [probe,setProbe]=useState('Chúc mừng, bạn được nhận quà miễn phí từ cửa hàng sách');
  const [guesses,setGuesses]=useState<(boolean|null)[]>(()=>TEST.slice(0,8).map(()=>null));
  const [pick,setPick]=useState(0),[alpha,setAlpha]=useState(1);
  const [draft,setDraft]=useState('Thầy ơi, em gửi bài tập. Nhân tiện có chương trình nhận quà miễn phí, bấm link nhé');
  const [thr,setThr]=useState(.5);
  const model=useMemo(()=>trainNB(TRAIN,alpha),[alpha]);
  const base=useMemo(()=>trainNB(TRAIN,1),[]);
  const answered=guesses.every(g=>g!==null),myScore=guesses.filter((g,i)=>g===TEST[i].spam).length;
  const topSpam=[...base.spamCount].sort((a,b)=>b[1]-a[1]).slice(0,10),topHam=[...base.hamCount].sort((a,b)=>b[1]-a[1]).slice(0,10);
  const weights=model.vocab.map(w=>({w,v:model.weight(w)})).sort((a,b)=>b.v-a.v);
  const all=[...TRAIN,...TEST];
  const p4=prob(model,draft);
  const preds=TEST.map(m=>({m,p:prob(model,m.text)}));
  const tp=preds.filter(x=>x.p>=thr&&x.m.spam).length,fp=preds.filter(x=>x.p>=thr&&!x.m.spam).length,fn=preds.filter(x=>x.p<thr&&x.m.spam).length,tn=TEST.length-tp-fp-fn;

  const steps:StepDef[]=[
    {label:'Em là bộ lọc',title:'Em đoán trước khi máy đoán.',
      intro:<p>Đọc 8 email và quyết định cái nào là rác. Để ý xem em dựa vào <b>từ ngữ nào</b> để quyết định.</p>,
      controls:<div className="stat-row">{answered?<Stat label="Em đoán đúng" value={`${myScore}/8`} tone={myScore>=7?'good':'warn'}/>:<Stat label="Đã trả lời" value={`${guesses.filter(g=>g!==null).length}/8`}/>}</div>,
      visual:<ol className="mail-list">{TEST.slice(0,8).map((m,i)=><li key={i} className={answered?(guesses[i]===m.spam?'right':'wrong'):''}><p>{m.text}</p><div className="segmented small">{[[true,'Rác'],[false,'Bình thường']].map(([v,t])=><button key={String(v)} className={guesses[i]===v?'on':''} onClick={()=>setGuesses(g=>g.map((x,j)=>j===i?v as boolean:x))}>{t as string}</button>)}</div>{answered&&<small>{m.spam?'Thật ra là rác':'Thật ra là email bình thường'}</small>}</li>)}</ol>,
      takeaway:'Em vừa dùng một bộ quy tắc trong đầu: "miễn phí", "bấm link", "trúng thưởng" thì nghi ngờ. Máy cũng làm vậy, nhưng nó tự đếm từ trên dữ liệu thay vì được dạy quy tắc.'},
    {label:'Đếm từ',title:'Từ nào hay xuất hiện ở đâu?',
      intro:<p>Máy chỉ nhìn thấy <b>túi từ</b>: mỗi email thành danh sách các từ, bỏ qua thứ tự. Đây là các từ hay gặp nhất trong 30 email huấn luyện.</p>,
      controls:<p className="small-note">Có những từ xuất hiện ở cả hai phía (như "bạn", "ngay"). Từ hữu ích là từ <b>lệch hẳn</b> về một phía.</p>,
      visual:<div className="two-col"><div><h4 className="col-title" style={{color:'var(--pink)'}}>Trong thư rác</h4><Bars rows={topSpam.map(([w,c])=>({label:w,value:c,color:'var(--pink)'}))} format={v=>String(v)}/></div><div><h4 className="col-title">Trong thư thường</h4><Bars rows={topHam.map(([w,c])=>({label:w,value:c}))} format={v=>String(v)}/></div></div>,
      takeaway:'Biến chữ thành con số đơn giản nhất là đếm. Mất thứ tự từ là cái giá phải trả, nhưng với lọc thư rác như vậy đã đủ dùng.'},
    {label:'Máy học trọng số',title:'Mỗi từ được một điểm nghi ngờ.',
      intro:<p>Naive Bayes so tần suất từ ở thư rác và thư thường. Điểm dương nghiêng về rác, điểm âm nghiêng về thư thường. Điểm cả email bằng tổng điểm các từ.</p>,
      controls:<><Slider label="Làm mượt α (Laplace)" value={alpha} min={.1} max={5} step={.1} onChange={setAlpha} format={v=>fmt(v,1)}/>
        <label className="select-field">Xem một email<select value={pick} onChange={e=>setPick(+e.target.value)}>{all.map((m,i)=><option key={i} value={i}>{m.text.slice(0,48)}…</option>)}</select></label>
        <Chips model={model} text={all[pick].text}/><div className="stat-row"><Stat label="Xác suất rác" value={pct(prob(model,all[pick].text))}/></div></>,
      visual:<div className="two-col"><div><h4 className="col-title" style={{color:'var(--pink)'}}>Nghi ngờ nhất</h4><Bars rows={weights.slice(0,10).map(x=>({label:x.w,value:x.v,color:'var(--pink)'}))}/></div><div><h4 className="col-title">Đáng tin nhất</h4><Bars rows={weights.slice(-10).reverse().map(x=>({label:x.w,value:-x.v}))}/></div></div>,
      takeaway:'α tránh chia cho 0 khi một từ chưa từng gặp ở một phía. α lớn làm các điểm bớt cực đoan, mô hình bớt tự tin vào vài từ hiếm.'},
    {label:'Lừa bộ lọc',title:'Em thử viết một email qua mặt máy.',
      intro:<p>Gõ một email lừa đảo nhưng tìm cách để xác suất rác dưới 50%. Màu từng từ cho biết nó kéo điểm về phía nào.</p>,
      controls:<><textarea rows={5} value={draft} onChange={e=>setDraft(e.target.value)} aria-label="Email của em"/>
        <div className="stat-row"><Stat label="Xác suất rác" value={pct(p4)} tone={p4>=.5?'bad':'good'}/><Stat label="Kết luận" value={p4>=.5?'Vào thư rác':'Lọt vào hộp thư'}/></div></>,
      visual:<><Chips model={model} text={draft}/><div className="prob-meter"><i style={{width:pct(p4,1)}}/></div><p className="chart-caption">Hồng: kéo về phía rác · xanh: kéo về phía thư thường · xám: chưa gặp hoặc trung tính.</p></>,
      takeaway:'Chèn nhiều "từ sạch" như "thầy", "bài tập", "nhóm" có thể kéo điểm xuống: đó là cách kẻ gửi rác né bộ lọc. Vì vậy bộ lọc thật cần dữ liệu mới liên tục và nhiều tín hiệu khác ngoài từ ngữ.'},
    {label:'Đánh giá',title:'Trên 10 email máy chưa thấy.',
      intro:<p>Gắn nhầm email của thầy vào thư rác tệ hơn để lọt một quảng cáo. Kéo ngưỡng và xem ô <b>báo nhầm</b>.</p>,
      controls:<><Slider label="Ngưỡng xác suất" value={thr} min={.1} max={.95} step={.05} onChange={setThr} format={v=>fmt(v)}/>
        <div className="stat-row"><Stat label="Đúng" value={pct((tp+tn)/TEST.length)}/><Stat label="Precision" value={pct(tp/Math.max(1,tp+fp))}/><Stat label="Email thật bị chặn" value={fp} tone={fp?'bad':'good'}/></div></>,
      visual:<><Confusion labels={['Thường','Rác']} matrix={[[tn,fp],[fn,tp]]} highlight={[0,1]}/>
        <ul className="pred-list">{preds.map((x,i)=><li key={i} className={(x.p>=thr)===x.m.spam?'right':'wrong'}><span>{pct(x.p)}</span>{x.m.text}</li>)}</ul></>,
      takeaway:'Email "cửa hàng sách giảm giá" của bạn bè dễ bị chặn nhầm vì chứa từ quảng cáo. Với lọc thư, ta thường ưu tiên precision cao: thà lọt vài thư rác còn hơn mất thư quan trọng.'},
  ];
  const reverse:ReverseDef={product:{label:'Dùng sản phẩm',title:'Một bộ lọc thư đã chạy.',
    intro:<p>Bộ lọc đã học xong tự xếp 10 email mới vào <b>Hộp thư</b> hoặc <b>Thư rác</b>. Gõ thêm một email ở dưới để thử. Đừng vội nhìn vào bên trong.</p>,
    controls:<><textarea rows={4} value={probe} onChange={e=>setProbe(e.target.value)} aria-label="Email thử"/><div className="stat-row"><Stat label="Xác suất rác" value={pct(prob(base,probe))} tone={prob(base,probe)>=.5?'bad':'good'}/><Stat label="Xếp vào" value={prob(base,probe)>=.5?'Thư rác':'Hộp thư'}/></div></>,
    visual:<div className="two-col"><div><h4 className="col-title">Hộp thư</h4><ul className="pred-list">{TEST.filter(m=>prob(base,m.text)<.5).map((m,i)=><li key={i} className={m.spam?'wrong':'right'}><span>{pct(prob(base,m.text))}</span>{m.text}</li>)}</ul></div><div><h4 className="col-title" style={{color:'var(--pink)'}}>Thư rác</h4><ul className="pred-list">{TEST.filter(m=>prob(base,m.text)>=.5).map((m,i)=><li key={i} className={m.spam?'right':'wrong'}><span>{pct(prob(base,m.text))}</span>{m.text}</li>)}</ul></div></div>,
    takeaway:'Bộ lọc xếp gần đúng hết, nhưng có ít nhất một email bị xếp nhầm (viền hồng). Câu hỏi để tháo: nó dựa vào đâu để quyết định?'},
  layers:[
    {from:2,label:'Mở hộp: điểm từng từ',title:'Bên trong là một bảng điểm từ.',intro:<p>Mở bộ lọc ra: mỗi từ có một điểm nghi ngờ, điểm email bằng tổng điểm các từ. Chọn một email để thấy từ nào kéo nó về phía rác.</p>,takeaway:'Toàn bộ bộ lọc là một bảng điểm cho vài trăm từ cộng một điểm gốc. Không có quy tắc nào được viết tay.'},
    {from:1,label:'Điểm đến từ đâu',title:'Đi ngược về dữ liệu: đếm từ.',intro:<p>Bảng điểm được tính từ việc đếm: từ nào hay xuất hiện trong 15 email rác mẫu, từ nào trong 15 email thường.</p>,takeaway:'Điểm của một từ chỉ là tỉ lệ tần suất ở hai phía. Đổi dữ liệu mẫu là đổi bảng điểm, nên chất lượng bộ lọc nằm ở dữ liệu.'},
    {from:4,label:'Quy tắc quyết định',title:'Ngưỡng chia hộp thư.',intro:<p>Bộ lọc đổi điểm thành xác suất rồi so với ngưỡng 0,5. Đổi ngưỡng và xem email nào chuyển hộp.</p>,takeaway:'Ngưỡng là chỗ con người quyết định: thà lọt thư rác còn hơn mất thư quan trọng.'},
    {from:3,label:'Thử phá',title:'Tìm điểm yếu của sản phẩm.',intro:<p>Biết bên trong là phép cộng điểm từ, em có thể lừa nó. Viết một email lừa đảo mà xác suất rác dưới 50%.</p>,takeaway:'Hiểu cơ chế giúp tìm ra lỗ hổng: chèn từ “sạch” để pha loãng điểm. Kẻ gửi thư rác thật cũng làm vậy.'},
    {from:0,label:'Tự lắp lại',title:'Em làm bộ lọc bằng đầu.',intro:<p>Giờ em là bộ lọc. Phân loại 8 email bằng quy tắc của riêng em rồi so với máy.</p>,takeaway:'Quy tắc trong đầu em và bảng điểm của máy giống nhau đến bất ngờ: cả hai đều dựa vào những từ đặc trưng.'},
  ]};
  return <ProjectFrame id="spam" steps={steps} reverse={reverse} navigate={navigate}/>;
}
