import fs from 'node:fs';
const root=new URL('../',import.meta.url),dist=new URL('dist/',root);let html=fs.readFileSync(new URL('index.html',dist),'utf8'),entry='';
const safe=s=>s.replace(/<\/script/gi,'<\\/script');
html=html.replace(/<script\b[^>]*src="([^"]+)"[^>]*><\/script>/g,(_,src)=>{const script='<script>\n'+safe(fs.readFileSync(new URL(src.replace(/^\.\//,''),dist),'utf8'))+'\n</script>';if(src.includes('/assets/')||src.startsWith('./assets/')){entry+=script;return ''}return script});
html=html.replace(/<link\b[^>]*href="([^"]+\.css)"[^>]*>/g,(_,src)=>'<style>'+fs.readFileSync(new URL(src.replace(/^\.\//,''),dist),'utf8')+'</style>');
const legacy=fs.readFileSync(new URL('legacy/index.html',dist),'utf8');html=html.replace('<body>',()=>'<body><script>window.ML_LEGACY_HTML='+safe(JSON.stringify(legacy))+';</script>');const end=html.lastIndexOf('</body>');html=html.slice(0,end)+entry+html.slice(end);
fs.writeFileSync(new URL('ML_Studio_React.html',root),html);console.log('Standalone React:',Buffer.byteLength(html),'bytes');
let distIndex=fs.readFileSync(new URL('index.html',dist),'utf8');
const version=Date.now().toString(36);
distIndex=distIndex.replace(/(src="(?:\.\/)?assets\/app\.js)(")/g,`$1?v=${version}$2`);
fs.writeFileSync(new URL('index.html',dist),distIndex);

