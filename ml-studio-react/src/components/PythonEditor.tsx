import {useLayoutEffect,useMemo,useRef} from 'react';
import {highlightPython} from '../lib/highlight';
interface Props {value:string;onChange:(value:string)=>void;ariaLabel:string;disabled?:boolean;rows:number;onRun?:()=>void}
/** Native textarea editing/IME/accessibility, with a synchronized Python syntax layer. */
export function PythonEditor({value,onChange,ariaLabel,disabled=false,rows,onRun}:Props){
 const layer=useRef<HTMLPreElement>(null),editor=useRef<HTMLTextAreaElement>(null);
 const html=useMemo(()=>highlightPython(value+'\n'),[value]);
 function syncScroll(){if(layer.current&&editor.current){layer.current.scrollTop=editor.current.scrollTop;layer.current.scrollLeft=editor.current.scrollLeft}}
 useLayoutEffect(syncScroll,[value,rows]);
 return <div className={'python-editor '+(disabled?'is-disabled':'')} data-language="python"><pre ref={layer} className="syntax-layer" aria-hidden="true"><code dangerouslySetInnerHTML={{__html:html}}/></pre><textarea ref={editor} aria-label={ariaLabel} value={value} spellCheck={false} autoCapitalize="off" autoCorrect="off" disabled={disabled} rows={rows} wrap="off" onScroll={syncScroll} onChange={e=>onChange(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&e.shiftKey&&onRun&&!e.nativeEvent.isComposing){e.preventDefault();onRun();return}if(e.key==='Tab'){e.preventDefault();const el=e.currentTarget,start=el.selectionStart;onChange(el.value.slice(0,start)+'    '+el.value.slice(el.selectionEnd));requestAnimationFrame(()=>el.selectionStart=el.selectionEnd=start+4)}}}/></div>
}
