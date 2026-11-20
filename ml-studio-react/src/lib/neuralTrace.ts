const sigmoid=(v:number)=>v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v));
export type Activation='tanh'|'linear';
export interface NeuralSample {id:number;x:number[];y:0|1}
export interface NetworkParameters {W:number[][];b:number[];v:number[];c:number}
export interface ForwardRow {z:number[];a:number[];logit:number;p:number;loss:number}
export interface NetworkStats {rows:ForwardRow[];loss:number}
export interface BackwardRow {deltaOutput:number;deltaHidden:number[];W:number[][];b:number[];v:number[];c:number}
export interface NetworkGradient {total:NetworkParameters;rows:BackwardRow[]}
export type NetworkPhase='input'|'hidden'|'output'|'loss'|'backward-output'|'backward-hidden'|'update';
export const networkPhases:NetworkPhase[]=['input','hidden','output','loss','backward-output','backward-hidden','update'];
export interface NeuralFrame {epoch:number;phase:NetworkPhase;before:NetworkParameters;parameters:NetworkParameters;beforeStats:NetworkStats;stats:NetworkStats;gradient:NetworkGradient;stopped:boolean}
export function neuralSamples(kind:'xor'|'moons'):NeuralSample[]{
 if(kind==='xor')return [{id:1,x:[-1,-1],y:0},{id:2,x:[-1,1],y:1},{id:3,x:[1,-1],y:1},{id:4,x:[1,1],y:0}];
 return Array.from({length:40},(_,i)=>{const c=i<20?0:1,t=(i%20)*Math.PI/19,n=.055*Math.sin(i*2.3);return {id:i+1,x:[c?1-Math.cos(t)+n:Math.cos(t)+n,c?.5-Math.sin(t)-n:Math.sin(t)-n],y:c as 0|1}});
}
export function initializeNetwork(hidden=4,seed=42):NetworkParameters{
 if(!Number.isInteger(hidden)||hidden<1||hidden>12)throw Error('Invalid hidden size');
 let state=seed>>>0;const random=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/2**32};
 return {W:Array.from({length:2},()=>Array.from({length:hidden},()=>(2*random()-1)*.9)),b:Array.from({length:hidden},()=>(2*random()-1)*.15),v:Array.from({length:hidden},()=>(2*random()-1)/Math.sqrt(hidden)),c:0};
}
export const copyNetwork=(p:NetworkParameters):NetworkParameters=>({W:p.W.map(r=>[...r]),b:[...p.b],v:[...p.v],c:p.c});
export function forwardNetwork(parameters:NetworkParameters,x:number[],activation:Activation='tanh'):ForwardRow{
 const z=parameters.b.map((b,j)=>b+x[0]*parameters.W[0][j]+x[1]*parameters.W[1][j]),a=z.map(v=>activation==='tanh'?Math.tanh(v):v),logit=parameters.c+a.reduce((s,v,j)=>s+v*parameters.v[j],0);
 return {z,a,logit,p:sigmoid(logit),loss:0};
}
export function networkStats(parameters:NetworkParameters,samples:NeuralSample[],activation:Activation='tanh'):NetworkStats{
 if(!samples.length)throw Error('Empty training set');
 const rows=samples.map(s=>{const f=forwardNetwork(parameters,s.x,activation);return {...f,loss:Math.max(f.logit,0)-s.y*f.logit+Math.log1p(Math.exp(-Math.abs(f.logit)))}});
 return {rows,loss:rows.reduce((s,r)=>s+r.loss,0)/rows.length};
}
export function backwardNetwork(parameters:NetworkParameters,samples:NeuralSample[],activation:Activation='tanh',stats=networkStats(parameters,samples,activation)):NetworkGradient{
 const n=samples.length,h=parameters.b.length,total:NetworkParameters={W:[Array(h).fill(0),Array(h).fill(0)],b:Array(h).fill(0),v:Array(h).fill(0),c:0};
 const rows=samples.map((s,i)=>{
  const f=stats.rows[i],deltaOutput=(f.p-s.y)/n,deltaHidden=f.a.map((a,j)=>deltaOutput*parameters.v[j]*(activation==='tanh'?1-a*a:1));
  const row:BackwardRow={deltaOutput,deltaHidden,W:s.x.map(x=>deltaHidden.map(d=>x*d)),b:[...deltaHidden],v:f.a.map(a=>a*deltaOutput),c:deltaOutput};
  total.c+=row.c;for(let j=0;j<h;j++){total.b[j]+=row.b[j];total.v[j]+=row.v[j];for(let d=0;d<2;d++)total.W[d][j]+=row.W[d][j]}
  return row;
 });
 return {total,rows};
}
export function updateNetwork(p:NetworkParameters,g:NetworkParameters,rate:number):NetworkParameters{return {W:p.W.map((r,i)=>r.map((v,j)=>v-rate*g.W[i][j])),b:p.b.map((v,j)=>v-rate*g.b[j]),v:p.v.map((v,j)=>v-rate*g.v[j]),c:p.c-rate*g.c}}
export function parameterEntries(p:NetworkParameters,g?:NetworkParameters){
 return [...p.W.flatMap((r,i)=>r.map((value,j)=>({key:`W_${i}_${j}`,label:`W[${i},${j}]`,value,gradient:g?.W[i][j]??0}))),...p.b.map((value,j)=>({key:`b_${j}`,label:`b[${j}]`,value,gradient:g?.b[j]??0})),...p.v.map((value,j)=>({key:`v_${j}`,label:`v[${j}]`,value,gradient:g?.v[j]??0})),{key:'c',label:'c',value:p.c,gradient:g?.c??0}];
}
export function perturbParameter(p:NetworkParameters,key:string,amount:number){
 const q=copyNetwork(p);if(!parameterEntries(q).some(e=>e.key===key))throw Error('Unknown parameter');
 const [kind,i,j]=key.split('_');if(kind==='W')q.W[+i][+j]+=amount;else if(kind==='b')q.b[+i]+=amount;else if(kind==='v')q.v[+i]+=amount;else q.c+=amount;return q;
}
export function checkNetworkGradient(p:NetworkParameters,samples:NeuralSample[],activation:Activation,key:string,eps=1e-5){
 const analytic=parameterEntries(p,backwardNetwork(p,samples,activation).total).find(e=>e.key===key);if(!analytic)throw Error('Unknown parameter');
 const numeric=(networkStats(perturbParameter(p,key,eps),samples,activation).loss-networkStats(perturbParameter(p,key,-eps),samples,activation).loss)/(2*eps);
 return {analytic:analytic.gradient,numeric,error:Math.abs(analytic.gradient-numeric)};
}
export function neuralTrace(samples:NeuralSample[],activation:Activation='tanh',hidden=4,rate=.3,seed=42,epochs=150):NeuralFrame[]{
 if(!Number.isFinite(rate)||rate<=0||!Number.isInteger(epochs)||epochs<1)throw Error('Invalid training configuration');
 let parameters=initializeNetwork(hidden,seed);const frames:NeuralFrame[]=[];
 for(let epoch=0;epoch<epochs;epoch++){
  const before=copyNetwork(parameters),beforeStats=networkStats(before,samples,activation),gradient=backwardNetwork(before,samples,activation,beforeStats),next=updateNetwork(before,gradient.total,rate);
  const stopped=parameterEntries(next).some(e=>!Number.isFinite(e.value)||Math.abs(e.value)>1e6),after=stopped?before:next,afterStats=stopped?beforeStats:networkStats(after,samples,activation);
  for(const phase of networkPhases)frames.push({epoch,phase,before:copyNetwork(before),parameters:copyNetwork(phase==='update'?after:before),beforeStats:structuredClone(beforeStats),stats:structuredClone(phase==='update'?afterStats:beforeStats),gradient:structuredClone(gradient),stopped:phase==='update'&&stopped});
  if(stopped)break;parameters=next;
 }
 return frames;
}
