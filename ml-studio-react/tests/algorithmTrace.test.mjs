import {test} from 'node:test';
import assert from 'node:assert/strict';
import {housingSamples,standardize,objective,learningTrace,originalCoefficients,clusterTrace,clusterPoints,nearestNeighbors,sigmoid} from '../src/lib/algorithmTrace.ts';
const close=(a,b,tol=1e-7)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);
test('four observation phases preserve old parameters until simultaneous update',()=>{
 const rows=housingSamples('multiple'),frames=learningTrace(rows,'multiple',.15,0,-1,2);
 assert.deepEqual(frames.map(f=>f.phase),['predict','loss','gradient','update','predict','loss','gradient','update']);
 for(let i=0;i<3;i++)assert.deepEqual(frames[i].theta,[0,0,0]);
 frames[3].theta.forEach((v,j)=>close(v,frames[3].before[j]-.15*frames[3].gradient[j]));
 assert.deepEqual(frames[4].before,frames[3].theta);assert.notEqual(frames[0].theta,frames[1].theta);
 close(frames[3].loss,objective(standardize(rows).X,rows.map(r=>r.y),frames[3].theta,'multiple').loss);
});
test('linear, multivariate and BCE gradients match central finite differences; row contributions sum correctly',()=>{
 for(const kind of ['linear','multiple','logistic']){
  const rows=housingSamples(kind),{X}=standardize(rows),y=rows.map(r=>r.y),frame=learningTrace(rows,kind,.15,0,-1,1)[2];
  frame.gradient.forEach((g,j)=>{
   const plus=[...frame.before],minus=[...frame.before];plus[j]+=1e-5;minus[j]-=1e-5;
   close(g,(objective(X,y,plus,kind).loss-objective(X,y,minus,kind).loss)/2e-5,1e-6);
   close(g,frame.contributions.reduce((s,row)=>s+row[j],0));
  });
 }
});
test('batch/mini-batch/SGD use exactly their chosen samples and count sample reads',()=>{
 const rows=housingSamples('linear');
 for(const size of [0,3,1]){const frames=learningTrace(rows,'linear',.15,size,-1,2),n=size||6;
  assert.equal(frames[3].reads,n);assert.equal(frames[7].reads,n*2);assert.equal(frames[2].used.length,n);
  frames[2].contributions.forEach((row,i)=>{if(!frames[2].used.includes(i))assert(row.every(v=>v===0))});
 }
});
test('stable learning reduces true loss; adding gradient or an oversized rate can increase it',()=>{
 for(const kind of ['linear','multiple','logistic']){const frames=learningTrace(housingSamples(kind),kind);let previous=frames[0].loss;for(const frame of frames.filter(f=>f.phase==='update')){assert(frame.loss<=previous+1e-10);previous=frame.loss}assert(frames.at(-1).loss<frames[0].loss)}
 const rows=housingSamples('linear');for(const [rate,direction] of [[.15,1],[1.2,-1]]){const frames=learningTrace(rows,'linear',rate,0,direction,2);assert(frames[3].loss>frames[0].loss)}
});
test('converted coefficients produce identical predictions in original units, including constant columns',()=>{
 const rows=housingSamples('multiple'),{X,mean,scale}=standardize(rows),theta=[1.2,.5,-.7],raw=originalCoefficients(theta,mean,scale);
 rows.forEach((r,i)=>close(raw[0]+raw[1]*r.x[0]+raw[2]*r.x[1],X[i].reduce((s,x,j)=>s+x*theta[j],0)));
 assert.deepEqual(standardize([{id:1,x:[2],y:1},{id:2,x:[2],y:3}]).scale,[1]);
});
test('logistic starts at P=.5 and BCE log2; stable sigmoid supports extreme logits',()=>{
 const frame=learningTrace(housingSamples('logistic'),'logistic')[0];assert(frame.predictions.every(p=>p===.5));close(frame.loss,Math.log(2));
 close(sigmoid(10000),1);close(sigmoid(-10000),0);assert(Number.isFinite(objective([[1],[1]],[0,1],[10000],'logistic').loss));
});
test('K-means assignment/move alternation exposes real distances, means and nonincreasing inertia',()=>{
 const frames=clusterTrace(clusterPoints,3);
 frames.forEach((frame,i)=>{
  assert.equal(frame.phase,i%2?'move':'assign');
  if(i)assert(frame.inertia<=frames[i-1].inertia+1e-10);
  frame.distances.forEach((row,j)=>assert.equal(frame.labels[j],row.indexOf(Math.min(...row))));
  close(frame.inertia,clusterPoints.reduce((s,p,j)=>s+(p.x-frame.centers[frame.labels[j]].x)**2+(p.y-frame.centers[frame.labels[j]].y)**2,0));
  if(frame.phase==='move')frame.centers.forEach((c,j)=>{const pts=clusterPoints.filter((_,i)=>frame.labels[i]===j);if(pts.length){close(c.x,pts.reduce((s,p)=>s+p.x,0)/pts.length);close(c.y,pts.reduce((s,p)=>s+p.y,0)/pts.length)}});
 });assert(frames.at(-1).stable);
});
test('empty clusters retain their centers rather than produce NaN',()=>{
 const frames=clusterTrace([{x:0,y:0},{x:0,y:0},{x:2,y:0}],3);
 assert(frames.some(f=>f.empty.length));assert(frames.every(f=>f.centers.every(c=>Number.isFinite(c.x)&&Number.isFinite(c.y))));
});
test('KNN sorts distances deterministically, exposes votes, and scaling cancels an artificial unit conversion',()=>{
 const points=[{x:-2,y:0,label:0},{x:1,y:2,label:1},{x:3,y:0,label:1}],q={x:.5,y:0};
 const one=nearestNeighbors(points,q,1);assert.equal(one.prediction,1);assert.equal(one.rows[0].i,1);
 const all=nearestNeighbors(points,q,3);assert.deepEqual(all.votes,[1,2]);assert.equal(all.prediction,1);
 const a=nearestNeighbors(points,q,3,1,true),b=nearestNeighbors(points,q,3,100,true);a.rows.forEach((r,i)=>close(r.distance,b.rows[i].distance));
 const tied=nearestNeighbors([{x:-1,y:0,label:0},{x:1,y:0,label:1}],{x:0,y:0},2);assert.equal(tied.rows[0].i,0);assert(tied.tie);assert.equal(tied.prediction,0);assert.throws(()=>nearestNeighbors(points,q,0));
});
