import {highlightPython} from '../lib/highlight';
import type {ReactNode} from 'react';

export function SimulationStudio({title,description,children}:{title:string;description?:string;children:ReactNode}) {
  return <section className="simulation-studio" aria-label={title}>
    <header className="simulation-studio-heading">
      <h3>{title}</h3>
      {description&&<p>{description}</p>}
    </header>
    {children}
  </section>;
}

export function SimulationPanel({title,children}:{title:string;children:ReactNode}) {
  return <div className="simulation-studio-card"><h4>{title}</h4>{children}</div>;
}

export function SimulationCode({lines,active,detail,file='thuat_toan.py'}:{lines:string[];active:number;detail?:ReactNode;file?:string}) {
  return <>
    <div className="code-window">
      <div className="code-window-bar" aria-hidden="true"><i/><i/><i/><b>{file}</b><em>python</em></div>
      <pre className="algo-code simulation-code" aria-label="Code tương ứng với bước đang xem">
        {lines.map((line,index)=><span key={index} className={active===index?'active':''} aria-current={active===index?'step':undefined}>
          <i aria-hidden="true">{String(index+1).padStart(2,'0')}</i><code dangerouslySetInnerHTML={{__html:highlightPython(line)}}/>
        </span>)}
      </pre>
    </div>
    {detail&&<div className="simulation-code-detail" role="status">{detail}</div>}
  </>;
}
