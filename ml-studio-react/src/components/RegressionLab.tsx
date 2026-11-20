import {useEffect, useRef, useState} from 'react';
import {ArrowRight, BookOpen, ChartLine, FlaskConical} from 'lucide-react';
import regressionData from '../data/regression.json';
import {leastSquares, validateNotebook} from '../lib/domain';
import type {PlotTrace} from '../types';
import {Chart} from './Chart';
import {Notebook} from './Notebook';

const hours = [1, 2, 3, 4, 6, 7, 8];
const scores = [43, 51, 57, 68, 77, 87, 91];
const cases = [{hour: 5, score: 74}, {hour: 3.5, score: 59}, {hour: 7.5, score: 90}];
const format = new Intl.NumberFormat('vi-VN');
const [intercept, slope] = leastSquares(hours.map(hour => [1, hour]), scores);
const notebook = validateNotebook(regressionData);
const graph = {
  margin: {l: 48, r: 21, t: 24, b: 50},
  font: {family: 'system-ui, sans-serif', size: 12, color: '#d6e1ff'},
  xaxis: {title: {text: 'Giờ ôn tập'}, range: [0.5, 8.5], dtick: 1, gridcolor: '#303a60', fixedrange: true},
  yaxis: {title: {text: 'Điểm / 100'}, range: [28, 102], dtick: 20, gridcolor: '#303a60', fixedrange: true},
  showlegend: false,
};
const observed: PlotTrace = {
  type: 'scatter', mode: 'markers', x: hours, y: scores,
  marker: {size: 10, color: '#b2aaff'},
  hovertemplate: '%{x} giờ · %{y} điểm<extra></extra>',
};

function RegressionSimulator() {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(760);

  useEffect(() => {
    function resize(event: MessageEvent) {
      if (event.source !== frame.current?.contentWindow || event.data?.type !== 'ml-regression-height') return;
      const next = event.data.height;
      if (Number.isInteger(next) && next > 100 && next < 50000) setHeight(next);
    }
    window.addEventListener('message', resize);
    return () => window.removeEventListener('message', resize);
  }, []);

  return <iframe ref={frame} className="regression-frame" style={{height}} title="Mô phỏng hồi quy một biến, nhiều biến và CSV"
    src={window.ML_LEGACY_HTML ? undefined : './legacy/index.html'} srcDoc={window.ML_LEGACY_HTML}/>;
}

export function RegressionLab({onBusy}: {onBusy: (value: boolean) => void}) {
  const [caseIndex, setCaseIndex] = useState(0);
  const [guess, setGuess] = useState(65);
  const {hour: hiddenHours, score: hiddenScore} = cases[caseIndex];
  const forecast = Math.round(intercept + slope * hiddenHours);
  const hourLabel = format.format(hiddenHours);
  const [revealed, setRevealed] = useState(false);
  const [openSimulator, setOpenSimulator] = useState(false);
  const [openNotebook, setOpenNotebook] = useState(false);
  const traces: PlotTrace[] = [observed, {
    type: 'scatter', mode: 'markers', x: [hiddenHours], y: [guess],
    marker: {size: 15, color: '#ffd48f', symbol: 'diamond', line: {width: 2, color: '#171f3d'}},
    hovertemplate: 'Em đoán %{y} điểm<extra></extra>',
  }];
  if (revealed) traces.push({
    type: 'scatter', mode: 'lines', x: [1, 8], y: [intercept + slope, intercept + 8 * slope],
    line: {color: '#82e5e1', width: 3}, hoverinfo: 'skip',
  }, {
    type: 'scatter', mode: 'markers', x: [hiddenHours], y: [hiddenScore],
    marker: {size: 19, color: '#fff', symbol: 'circle-open', line: {width: 2}},
    hovertemplate: 'Điểm quan sát %{y}<extra></extra>',
  });

  return <>
    <section className="regression-challenge simulation-studio" aria-labelledby="regression-title">
      <div className="challenge-copy">
        <p className="challenge-kicker"><FlaskConical size={16}/> Phòng hồi quy</p>
        <h1 id="regression-title">Ôn {hourLabel} giờ,<br/> được bao nhiêu điểm?</h1>
        <p>Bảy bạn đã có điểm. Dựa vào biểu đồ, em thử đoán điểm của bạn ôn {hourLabel} giờ.</p>
        <label className="challenge-range">
          <span>Điểm em đoán <strong>{guess}<small> / 100</small></strong></span>
          <input type="range" min={30} max={100} value={guess} disabled={revealed} onChange={event => setGuess(Number(event.target.value))}/>
        </label>
        <button className="btn-primary" onClick={() => {
          if (!revealed) {
            setRevealed(true);
            return;
          }
          setCaseIndex(index => (index + 1) % cases.length);
          setGuess(65);
          setRevealed(false);
        }}>{!revealed ? 'Kiểm tra dự đoán' : caseIndex === cases.length - 1 ? 'Làm lại từ đầu' : 'Thử trường hợp tiếp theo'} <ArrowRight size={16}/></button>
        <div className="challenge-feedback" role="status">{revealed && <>
          <strong>{guess === hiddenScore ? `Em đoán đúng ${hiddenScore} điểm.` : `Em lệch ${Math.abs(guess - hiddenScore)} điểm so với kết quả ${hiddenScore}.`}</strong>
          <p>Mô hình đoán khoảng {forecast} điểm. Đường xanh được khớp từ bảy điểm đã biết, trước khi xem điểm của bạn ôn {hourLabel} giờ.</p>
        </>}</div>
      </div>
      <figure className="challenge-graph">
        <Chart traces={traces} height={342} xTitle="Giờ ôn tập" yTitle="Điểm / 100" layout={graph}/>
        <figcaption><span className="graph-known">Điểm đã biết</span><span className="graph-guess">Em đoán</span>
          {revealed && <><span className="graph-model">Mô hình</span><span className="graph-answer">Điểm công bố</span></>}
        </figcaption>
        <details>
          <summary>Xem dữ liệu của bảy bạn</summary>
          <ul>{hours.map((hour, index) => <li key={hour}>{hour} giờ: {scores[index]} điểm</li>)}</ul>
        </details>
        <p>Dữ liệu giả lập để luyện hồi quy.</p>
      </figure>
    </section>
    <section className="regression-tools" aria-label="Thực hành hồi quy">
      <details onToggle={event => {if (event.currentTarget.open) setOpenSimulator(true)}}>
        <summary><ChartLine size={22}/><span><strong>Cho máy tự tìm đường thẳng</strong><small>Huấn luyện từng bước với một biến, nhiều biến, Iris hoặc CSV.</small></span></summary>
        {openSimulator && <RegressionSimulator/>}
      </details>
      <details onToggle={event => {if (event.currentTarget.open) setOpenNotebook(true)}}>
        <summary><BookOpen size={22}/><span><strong>Thử bằng Python</strong><small>Chạy notebook và xem phép tính phía sau mô hình.</small></span></summary>
        {openNotebook && <Notebook book={notebook} bookId="regression" title="Notebook hồi quy tuyến tính" onBusy={onBusy}/>}
      </details>
    </section>
  </>;
}
