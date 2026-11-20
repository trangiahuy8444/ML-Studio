import './legacy-embed.mjs';
import fs from 'node:fs';import vm from 'node:vm';
const old=new URL('../../linear-regression-lab/dist/',import.meta.url),root=new URL('../',import.meta.url);let ctx={window:{}};vm.createContext(ctx);
for(const name of ['course-data.js','housing-project-data.js'])vm.runInContext(fs.readFileSync(new URL(name,old),'utf8'),ctx);
fs.writeFileSync(new URL('src/data/course.json',root),JSON.stringify(ctx.window.AI_COURSE));
fs.writeFileSync(new URL('src/data/housing.json',root),JSON.stringify(ctx.window.HOUSING_PROJECT_DATA));
for(const [src,dest] of [['vendor/plotly.min.js','plotly.min.js'],['iris-data.js','iris-data.js'],['python-kernel.js','python-kernel.js'],['simulations.css','simulations.css']])fs.copyFileSync(new URL(src,old),new URL('public/runtime/'+dest,root));
let sim=fs.readFileSync(new URL('lesson-simulations.js',old),'utf8').replace("window.addEventListener('hashchange',cleanup);","window.cleanupLessonSimulations=cleanup;window.addEventListener('hashchange',cleanup);");fs.writeFileSync(new URL('public/runtime/lesson-simulations.js',root),sim);
