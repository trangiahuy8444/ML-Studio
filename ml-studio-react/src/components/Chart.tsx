import {useEffect,useRef} from 'react';
import type {PlotTrace} from '../types';

const darkColors=['#42d3e5','#f28db9','#84e0a8','#ffd173','#a8a7ff'];
const brightColors:Record<string,string>={'#277d9b':darkColors[0],'#be5c82':darkColors[1],'#397750':darkColors[2],'#c18b22':darkColors[3],'#e85d35':'#ff9b80','#855923':darkColors[3]};
function chartTrace(trace:PlotTrace):PlotTrace {
  const result={...trace};
  for(const key of ['marker','line']){
    const value=result[key];
    if(value&&typeof value==='object'&&!Array.isArray(value)){
      const style=value as Record<string,unknown>;
      if(typeof style.color==='string'&&brightColors[style.color.toLowerCase()])result[key]={...style,color:brightColors[style.color.toLowerCase()]};
    }
  }
  return result;
}
type Layout=Record<string,unknown>;
const object=(entry:unknown)=>entry&&typeof entry==='object'?entry as Layout:{};

export function Chart({traces,xTitle='',yTitle='',height=350,layout}:{traces:PlotTrace[];xTitle?:string;yTitle?:string;height?:number;layout?:Layout}){
  const ref=useRef<HTMLDivElement>(null);
  const resizeReady=useRef(false);
  useEffect(()=>{
    const el=ref.current;
    if(!el)return;
    const resize=new ResizeObserver(()=>{if(el.isConnected&&resizeReady.current&&el.offsetWidth>0&&el.offsetHeight>0)void Promise.resolve(window.Plotly.Plots.resize(el)).catch(()=>{});});
    resize.observe(el);
    return()=>{resizeReady.current=false;resize.disconnect();window.Plotly.purge(el);};
  },[]);
  useEffect(()=>{
    const el=ref.current;
    if(!el)return;
    let cancelled=false;
    const dark=!!el.closest('.simulation-studio');
    const grid=dark?'#2b405a':'#e4ebe3';
    const axis=(value:unknown,title='')=>({...{title:{text:title},gridcolor:grid,automargin:true},...object(value),...(dark?{gridcolor:grid,zerolinecolor:'#41536a',linecolor:'#39536f'}:{})});
    const config:Layout={paper_bgcolor:'transparent',plot_bgcolor:'transparent',
      font:{family:'Arial, sans-serif',size:12,color:dark?'#c7d9ec':'#355044'},margin:dark?{l:55,r:18,t:46,b:50}:{l:55,r:18,t:15,b:65},
      // Studio tối: chú giải đặt phía trên vùng vẽ để không đè lên tiêu đề trục x ở biểu đồ thấp.
      legend:dark?{orientation:'h',x:0,xanchor:'left',y:1.02,yanchor:'bottom'}:{orientation:'h',y:-.23},...(dark?{colorway:darkColors,hoverlabel:{bgcolor:'#182944',font:{color:'#eff8ff'}}}:{}),...layout,
      xaxis:axis(layout?.xaxis,xTitle),yaxis:axis(layout?.yaxis,yTitle)};
    if(dark&&layout?.scene){
      const scene=object(layout.scene);
      config.scene={...scene,bgcolor:'#111d31',xaxis:axis(scene.xaxis),yaxis:axis(scene.yaxis),zaxis:axis(scene.zaxis)};
    }
    void window.Plotly.react(el,dark?traces.map(chartTrace):traces,config,{responsive:true,displayModeBar:false})
      .then(()=>{if(!cancelled)resizeReady.current=true;})
      .catch(()=>{if(!cancelled)el.textContent='Không vẽ được biểu đồ. Hãy tải lại trang.';});
    return()=>{cancelled=true;};
  },[traces,xTitle,yTitle,layout]);
  return <div ref={ref} style={{height,width:'100%'}} role="img" aria-label={xTitle||yTitle?`${yTitle} theo ${xTitle}`:'Biểu đồ mô phỏng'}/>;
}
