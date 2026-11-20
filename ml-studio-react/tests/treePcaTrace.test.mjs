import {test} from 'node:test';import assert from 'node:assert/strict';
import {treeSamples,gini,splitCandidates,decisionTreeTrace,treePredict,pcaSamples,pca2D,pcaProjection} from '../src/lib/treePcaTrace.ts';
const close=(a,b,tol=1e-8)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);
test('Gini weights child sample counts; midpoints include only distinct consecutive values',()=>{
 close(gini([0,0]),0);close(gini([5,5]),.5);close(gini([4,0]),0);
 const rows=[{id:1,x:[0],y:0},{id:2,x:[0],y:1},{id:3,x:[2],y:1},{id:4,x:[4],y:1}],c=splitCandidates(rows,[0,1,2,3],[0],2);
 assert.deepEqual(c.map(c=>c.threshold),[1,3]);assert(c[0].valid);assert(!c[1].valid);close(c[0].weighted,.25);close(c[0].gain,.125);
});
test('CART visits every midpoint, respects min-leaf and depth, and predicts without using labels at query time',()=>{
 const rows=treeSamples(),trace=decisionTreeTrace(rows,2,1);
 const root=trace.frames.find(f=>f.phase==='choose'&&f.nodeId==='r');assert(root);
 const candidates=root.candidates.filter(c=>c.valid);close(root.candidates[root.best].gain,Math.max(...candidates.map(c=>c.gain)));
 assert(trace.nodes.every(n=>n.depth<=2));assert(trace.nodes.filter(n=>n.status==='leaf').every(n=>n.indices.length>=1));
 assert.deepEqual(rows.map(r=>treePredict(trace.nodes,r.x).prediction),rows.map(r=>r.y));
 const constrained=decisionTreeTrace(rows,3,3);assert(constrained.nodes.filter(n=>n.status==='leaf').every(n=>n.indices.length>=3));
});
test('tree frames keep pending siblings visible and snapshots independent; partial queries stop provisionally',()=>{
 const trace=decisionTreeTrace(treeSamples(),3,1),split=trace.frames.find(f=>f.phase==='split'&&f.nodeId==='r');
 assert.deepEqual(split.nodes.map(n=>n.id),['r','rL','rR']);assert(split.nodes.slice(1).every(n=>n.status==='pending'));
 const following=trace.frames[trace.frames.indexOf(split)+1];assert(following.nodes.some(n=>n.id==='rR'));
 const root=split.nodes[0],x=root.feature===0?[root.threshold+1,12]:[95,root.threshold+1];assert(treePredict(split.nodes,x).provisional);
 assert(split.nodes.slice(1).every(n=>n.status==='pending'));assert(trace.nodes.every(n=>n.status!=='pending'));
});
test('pure/depth-zero/no-gain nodes stop; equal gains choose first feature then first threshold',()=>{
 const pure=decisionTreeTrace([{id:1,x:[1,1],y:0},{id:2,x:[2,2],y:0}]);assert.equal(pure.frames.length,1);assert.equal(pure.nodes[0].status,'leaf');
 assert.equal(decisionTreeTrace(treeSamples(),0).nodes.length,1);
 const same=decisionTreeTrace([{id:1,x:[1,1],y:0},{id:2,x:[1,1],y:1}]);assert.equal(same.nodes.length,1);assert.equal(same.nodes[0].prediction,0);
 const tied=decisionTreeTrace([{id:1,x:[0,0],y:0},{id:2,x:[1,1],y:1}]);assert.equal(tied.nodes[0].feature,0);close(tied.nodes[0].threshold,.5);
});
test('PCA centers samples, uses sample covariance and returns sorted orthonormal eigenvectors',()=>{
 const p=pca2D(pcaSamples());close(p.centered.reduce((s,v)=>s+v.x,0),0);close(p.centered.reduce((s,v)=>s+v.y,0),0);assert(p.values[0]>=p.values[1]);close(p.ratio[0]+p.ratio[1],1);
 for(let i=0;i<2;i++){const u=p.vectors[i];close(Math.hypot(u.x,u.y),1);close(p.covariance[0][0]*u.x+p.covariance[0][1]*u.y,p.values[i]*u.x);close(p.covariance[1][0]*u.x+p.covariance[1][1]*u.y,p.values[i]*u.y)}
 close(p.vectors[0].x*p.vectors[1].x+p.vectors[0].y*p.vectors[1].y,0);
});
test('first PCA direction maximizes projected variance; rank-one residual matches discarded eigenvalue',()=>{
 const p=pca2D(pcaSamples()),projection=pcaProjection(p.centered,p.vectors[0]);close(projection.variance,p.values[0]);close(projection.mse,p.values[1]*(p.centered.length-1)/p.centered.length);
 for(let d=-90;d<=180;d+=10){const a=d*Math.PI/180;assert(pcaProjection(p.centered,{x:Math.cos(a),y:Math.sin(a)}).variance<=projection.variance+1e-9)}
 close(pcaProjection(p.centered,p.vectors[0],2).mse,0);
});
test('standardization cancels a unit conversion; raw covariance PCA changes with feature units',()=>{
 const a=pca2D(pcaSamples(1),true),b=pca2D(pcaSamples(100),true);a.covariance.forEach((row,i)=>row.forEach((v,j)=>close(v,b.covariance[i][j])));close(a.ratio[0],b.ratio[0]);
 assert(pca2D(pcaSamples(100)).ratio[0]>pca2D(pcaSamples(1)).ratio[0]);
});
test('PCA handles constant/collinear data and eigenvector sign; rejects invalid projections',()=>{
 const line=pca2D([{x:-1,y:-1},{x:0,y:0},{x:1,y:1}]);close(line.ratio[0],1);close(pcaProjection(line.centered,line.vectors[0]).mse,0);
 const p=pca2D([{x:2,y:3},{x:2,y:3}],true);assert.deepEqual(p.values,[0,0]);assert.deepEqual(p.ratio,[0,0]);assert(p.centered.every(v=>v.x===0&&v.y===0));
 const q=pca2D(pcaSamples()),a=pcaProjection(q.centered,q.vectors[0]),b=pcaProjection(q.centered,{x:-q.vectors[0].x,y:-q.vectors[0].y});a.reconstructed.forEach((v,i)=>{close(v.x,b.reconstructed[i].x);close(v.y,b.reconstructed[i].y)});assert.throws(()=>pcaProjection(q.centered,{x:0,y:0}));
});
