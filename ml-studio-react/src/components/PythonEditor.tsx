import {useLayoutEffect,useMemo,useRef} from 'react';
import {highlightPython} from '../lib/highlight';
interface Props {value:string;onChange:(value:string)=>void;ariaLabel:string;disabled?:boolean;rows:number;onRun?:()=>void}

const EDITOR_CSS = `
.python-editor {
  position: relative !important;
  min-width: 0 !important;
  isolation: isolate !important;
  background: #08101c !important;
  border-radius: 4px !important;
  --python-code-font: 12.5px;
  --python-code-line: 1.6;
}
.python-editor .syntax-layer,
.python-editor textarea {
  box-sizing: border-box !important;
  margin: 0 !important;
  padding: 8px 10px !important;
  border: 1px solid #2a3e66 !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
  font-size: var(--python-code-font) !important;
  line-height: var(--python-code-line) !important;
  letter-spacing: 0 !important;
  word-spacing: normal !important;
  text-align: left !important;
  text-indent: 0 !important;
  white-space: pre !important;
  tab-size: 4 !important;
  font-variant-ligatures: none !important;
  overflow-wrap: normal !important;
  word-break: normal !important;
}
.python-editor .syntax-layer {
  position: absolute !important;
  inset: 0 !important;
  z-index: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
  color: #e6eef8 !important;
  max-height: none !important;
  background: transparent !important;
  border-color: transparent !important;
}
.python-editor .syntax-layer code {
  display: block !important;
  font: inherit !important;
  letter-spacing: inherit !important;
  white-space: inherit !important;
  color: #e6eef8 !important;
  background: transparent !important;
  padding: 0 !important;
  border: 0 !important;
}
.python-editor textarea {
  position: relative !important;
  z-index: 1 !important;
  display: block !important;
  width: 100% !important;
  background: transparent !important;
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
  caret-color: #e6eef8 !important;
  resize: vertical !important;
  overflow: auto !important;
  min-height: 44px !important;
}
.python-editor textarea:disabled {
  background: transparent !important;
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
}
.python-editor textarea::selection {
  background: rgba(147, 197, 253, 0.35) !important;
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
}
.python-editor .syntax-layer * {
  color: #e6eef8;
}
.python-editor .syntax-layer .hljs-keyword,
.python-editor .syntax-layer .hljs-literal {
  color: #c792ea !important;
  font-weight: 600 !important;
}
.python-editor .syntax-layer .hljs-built_in {
  color: #82aaff !important;
}
.python-editor .syntax-layer .hljs-string,
.python-editor .syntax-layer .hljs-regexp {
  color: #c3e88d !important;
}
.python-editor .syntax-layer .hljs-number {
  color: #f78c6c !important;
}
.python-editor .syntax-layer .hljs-comment {
  color: #6e83a0 !important;
  font-style: italic !important;
}
.python-editor .syntax-layer .hljs-title,
.python-editor .syntax-layer .hljs-function .hljs-title,
.python-editor .syntax-layer .hljs-class .hljs-title {
  color: #82aaff !important;
}
.python-editor .syntax-layer .hljs-operator {
  color: #89ddff !important;
}
.python-editor .syntax-layer .hljs-meta {
  color: #ffd166 !important;
}
.python-editor .syntax-layer .hljs-params,
.python-editor .syntax-layer .hljs-subst,
.python-editor .syntax-layer .hljs-variable,
.python-editor .syntax-layer .hljs-punctuation {
  color: #e6eef8 !important;
}
`;

/** Native textarea editing/IME/accessibility, with a synchronized Python syntax layer. */
export function PythonEditor({value,onChange,ariaLabel,disabled=false,rows,onRun}:Props){
 const layer=useRef<HTMLPreElement>(null),editor=useRef<HTMLTextAreaElement>(null);
 const html=useMemo(()=>highlightPython(value+'\n'),[value]);
 function syncScroll(){if(layer.current&&editor.current){layer.current.scrollTop=editor.current.scrollTop;layer.current.scrollLeft=editor.current.scrollLeft}}
 useLayoutEffect(syncScroll,[value,rows]);
 return <div className={'python-editor '+(disabled?'is-disabled':'')} data-language="python">
  <style>{EDITOR_CSS}</style>
  <pre ref={layer} className="syntax-layer" aria-hidden="true"><code dangerouslySetInnerHTML={{__html:html}}/></pre>
  <textarea ref={editor} aria-label={ariaLabel} value={value} spellCheck={false} autoCapitalize="off" autoCorrect="off" disabled={disabled} rows={rows} wrap="off" onScroll={syncScroll} onChange={e=>onChange(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&e.shiftKey&&onRun&&!e.nativeEvent.isComposing){e.preventDefault();onRun();return}if(e.key==='Tab'){e.preventDefault();const el=e.currentTarget,start=el.selectionStart;onChange(el.value.slice(0,start)+'    '+el.value.slice(el.selectionEnd));requestAnimationFrame(()=>el.selectionStart=el.selectionEnd=start+4)}}}/>
 </div>
}
