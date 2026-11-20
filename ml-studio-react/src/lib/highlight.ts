/// <reference types="highlight.js" />
import hljs from 'highlight.js/lib/core.js';
import python from 'highlight.js/lib/languages/python.js';
const language=python(hljs);
language.contains??=[];
language.contains.push({className:'operator',begin:/[-+*/%=<>!~^|&]+/});
hljs.registerLanguage('python',()=>language);
export function highlightPython(source:string):string {
 try{return hljs.highlight(source,{language:'python',ignoreIllegals:true}).value}
 catch{return source.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
}
if(typeof window!=='undefined')window.ML_HIGHLIGHT_PYTHON=highlightPython;
