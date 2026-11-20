import {test} from 'node:test';import assert from 'node:assert/strict';
import {initializeNetwork,neuralSamples,forwardNetwork,networkStats,backwardNetwork,updateNetwork,parameterEntries,perturbParameter,checkNetworkGradient,neuralTrace,networkPhases} from '../src/lib/neuralTrace.ts';
const close=(a,b,tol=1e-7)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);
test('seeded initialization is reproducible, correctly shaped, and does not initialize all hidden nodes identically',()=>{
 const p=initializeNetwork(4,42);assert.deepEqual(p,initializeNetwork(4,42));assert.notDeepEqual(p,initializeNetwork(4,43));assert.equal(parameterEntries(p).length,17);assert.equal(new Set(p.W[0]).size,4);assert.throws(()=>initializeNetwork(0));
});
test('forward sums and sigmoid/BCE match a hand-worked two-hidden network',()=>{
 const p={W:[[.5,-.2],[.1,.3]],b:[.1,-.1],v:[.8,-.6],c:.2},x=[1,2];
 const f=forwardNetwork(p,x);close(f.z[0],.8);close(f.z[1],.3);close(f.a[0],Math.tanh(.8));const z=Math.tanh(.8)*.8-Math.tanh(.3)*.6+.2;close(f.logit,z);close(f.p,1/(1+Math.exp(-z)));close(networkStats(p,[{id:1,x,y:1}]).loss,Math.log1p(Math.exp(-z)));
});
test('all weights and both bias layers match central finite-difference BCE gradients for tanh and linear activations',()=>{
 for(const activation of ['tanh','linear'])for(const dataset of ['xor','moons']){
  const p=initializeNetwork(4,42),samples=neuralSamples(dataset);
  for(const entry of parameterEntries(p)){const check=checkNetworkGradient(p,samples,activation,entry.key);close(check.analytic,check.numeric,1e-7)}
 }
});
test('per-sample gradient contributions sum to the batch mean and duplicate data leave gradients unchanged',()=>{
 const p=initializeNetwork(),samples=neuralSamples('moons'),g=backwardNetwork(p,samples),duplicated=backwardNetwork(p,[...samples,...samples]);
 parameterEntries(g.total).forEach((entry,j)=>{close(entry.value,parameterEntries(duplicated.total)[j].value);close(entry.value,g.rows.reduce((sum,row)=>sum+parameterEntries(row)[j].value,0))});
});
test('seven trace phases preserve old parameters until a simultaneous update, with coherent probabilities and loss',()=>{
 const data=neuralSamples('xor'),frames=neuralTrace(data,'tanh',4,.3,42,2);assert.equal(frames.length,14);assert.deepEqual(frames.slice(0,7).map(f=>f.phase),networkPhases);
 for(const f of frames.slice(0,6))assert.deepEqual(f.parameters,frames[0].before);
 assert.deepEqual(frames[6].parameters,updateNetwork(frames[6].before,frames[6].gradient.total,.3));assert.deepEqual(frames[7].before,frames[6].parameters);close(frames[6].stats.loss,networkStats(frames[6].parameters,data).loss);
 assert.notEqual(frames[0].before.W[0],frames[1].before.W[0]);assert.notEqual(frames[0].gradient.rows,frames[1].gradient.rows);
});
test('default tanh network learns XOR while a linear hidden layer remains a linear classifier',()=>{
 const data=neuralSamples('xor'),a=neuralTrace(data),b=neuralTrace(data,'linear');assert(a.at(-1).stats.loss<a[0].stats.loss*.2);assert(a.at(-1).stats.rows.every((r,i)=>+(r.p>=.5)===data[i].y));assert(b.at(-1).stats.loss>.69);
});
test('changing a decision threshold does not change probabilities, BCE or gradients',()=>{
 const p=initializeNetwork(),data=neuralSamples('moons'),stats=networkStats(p,data),g=backwardNetwork(p,data);const low=stats.rows.map(r=>+(r.p>=.2)),high=stats.rows.map(r=>+(r.p>=.8));assert.notDeepEqual(low,high);assert.deepEqual(stats,networkStats(p,data));assert.deepEqual(g,backwardNetwork(p,data));
});
test('stable BCE handles extreme logits and perturbation copies protect recorded parameters',()=>{
 const p=initializeNetwork();p.c=10000;assert(Number.isFinite(networkStats(p,neuralSamples('xor')).loss));const changed=perturbParameter(p,'W_0_0',.2);close(changed.W[0][0],p.W[0][0]+.2);assert.notEqual(changed.W[0],p.W[0]);assert.throws(()=>perturbParameter(p,'unknown',1));assert.throws(()=>neuralTrace(neuralSamples('xor'),'tanh',4,0));
});
