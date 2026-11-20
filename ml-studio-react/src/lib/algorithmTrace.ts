export type LearningKind='linear'|'multiple'|'logistic';
export interface Sample {id:number;x:number[];y:number}
export type LearningPhase='predict'|'loss'|'gradient'|'update';
export interface LearningFrame {iteration:number;phase:LearningPhase;theta:number[];before:number[];gradient:number[];predictions:number[];errors:number[];contributions:number[][];loss:number;used:number[];reads:number;stopped:boolean}
export const sigmoid=(v:number)=>v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v));
const dot=(a:number[],b:number[])=>a.reduce((v,x,i)=>v+x*b[i],0);
export function housingSamples(kind:LearningKind,outlier=false):Sample[]{
 const areas=[40,55,65,90,110,130],ages=[30,5,20,8,35,12];
 return areas.map((area,i)=>({id:i+1,x:kind==='linear'?[area]:[area,ages[i]],y:kind==='logistic'?+(area-ages[i]*.5>72):kind==='linear'?[1.55,2.05,2.4,3.15,3.8,4.35][i]:.8+.03*area-.015*ages[i]+[.05,-.1,.08,-.06,.1,-.05][i]})).map((r,i)=>outlier&&i===5?{...r,y:kind==='logistic'?1-r.y:r.y+3}:r);
}
export function standardize(samples:Sample[]){
 const n=samples.length,d=samples[0].x.length;
 const mean=Array.from({length:d},(_,j)=>samples.reduce((s,r)=>s+r.x[j],0)/n);
 const scale=mean.map((m,j)=>Math.sqrt(samples.reduce((s,r)=>s+(r.x[j]-m)**2,0)/n)||1);
 return {mean,scale,X:samples.map(r=>[1,...r.x.map((x,j)=>(x-mean[j])/scale[j])])};
}
export function objective(X:number[][],y:number[],theta:number[],kind:LearningKind){
 const scores=X.map(r=>dot(r,theta)),predictions=scores.map(z=>kind==='logistic'?sigmoid(z):z);
 // Softplus form keeps BCE finite even for large logits.
 const loss=scores.reduce((s,z,i)=>s+(kind==='logistic'?Math.max(z,0)-y[i]*z+Math.log1p(Math.exp(-Math.abs(z))):(z-y[i])**2),0)/y.length;
 return {loss,predictions,errors:predictions.map((p,i)=>p-y[i])};
}
export function learningTrace(samples:Sample[],kind:LearningKind,rate=.15,batch=0,direction=-1,iterations=40):LearningFrame[]{
 const {X}=standardize(samples),y=samples.map(r=>r.y),n=y.length,frames:LearningFrame[]=[];
 let theta=X[0].map(()=>0),reads=0;
 for(let iteration=0;iteration<iterations;iteration++){
  const size=batch===0?n:Math.min(batch,n),used=Array.from({length:size},(_,i)=>(iteration*size+i)%n),before=[...theta];
  const old=objective(X,y,theta,kind),factor=kind==='logistic'?1:2;
  const contributions=X.map((row,i)=>row.map(x=>used.includes(i)?factor*old.errors[i]*x/size:0));
  const gradient=theta.map((_,j)=>contributions.reduce((s,row)=>s+row[j],0));
  const next=theta.map((v,j)=>v+direction*rate*gradient[j]);
  const stopped=next.some(v=>!Number.isFinite(v)||Math.abs(v)>1e7);
  for(const phase of ['predict','loss','gradient','update'] as const){
   const shown=phase==='update'&&!stopped?next:before,stats=objective(X,y,shown,kind);
   frames.push({iteration,phase,theta:[...shown],before:[...before],gradient:[...gradient],...stats,contributions:contributions.map(row=>[...row]),used:[...used],reads:reads+(phase==='update'?size:0),stopped:phase==='update'&&stopped});
  }
  reads+=size;if(stopped)break;theta=next;
 }
 return frames;
}
export function originalCoefficients(theta:number[],mean:number[],scale:number[]){
 const weights=theta.slice(1).map((v,j)=>v/scale[j]);
 return [theta[0]-weights.reduce((s,v,j)=>s+v*mean[j],0),...weights];
}
export interface Point {x:number;y:number;label?:number}
export const clusterPoints:Point[]=Array.from({length:18},(_,i)=>{const c=i%3,t=i*1.7;return {x:[-2.2,0,2.2][c]+.45*Math.cos(t),y:[-1,1.7,-.8][c]+.4*Math.sin(t)}});
export interface ClusterFrame {iteration:number;phase:'assign'|'move';centers:Point[];before:Point[];labels:number[];distances:number[][];inertia:number;empty:number[];stable:boolean}
const distance2=(a:Point,b:Point)=>(a.x-b.x)**2+(a.y-b.y)**2;
export function clusterTrace(points:Point[],k=3,alternate=false,iterations=12):ClusterFrame[]{
 if(!Number.isInteger(k)||k<1||k>points.length)throw Error('Invalid number of clusters');
 let centers=Array.from({length:k},(_,i)=>({...points[alternate?(i*4+6)%points.length:i]}));const frames:ClusterFrame[]=[];
 for(let iteration=0;iteration<iterations;iteration++){
  const distances=points.map(p=>centers.map(c=>Math.sqrt(distance2(p,c))));
  const labels=distances.map(ds=>ds.indexOf(Math.min(...ds)));
  const inertia=points.reduce((s,p,i)=>s+distance2(p,centers[labels[i]]),0);
  const empty:number[]=[];
  const next=centers.map((c,j)=>{const members=points.filter((_,i)=>labels[i]===j);if(!members.length){empty.push(j);return {...c}}return {x:members.reduce((s,p)=>s+p.x,0)/members.length,y:members.reduce((s,p)=>s+p.y,0)/members.length}});
  const stable=next.every((c,j)=>distance2(c,centers[j])<1e-12);
  for(const phase of ['assign','move'] as const){const shown=phase==='move'?next:centers;frames.push({iteration,phase,centers:shown.map(c=>({...c})),before:centers.map(c=>({...c})),labels:[...labels],distances:distances.map(r=>[...r]),inertia:phase==='move'?points.reduce((s,p,i)=>s+distance2(p,shown[labels[i]]),0):inertia,empty:[...empty],stable})}
  centers=next;if(stable)break;
 }
 return frames;
}
export function nearestNeighbors(points:Point[],query:Point,k:number,stretch=1,scaled=false){
 if(!Number.isInteger(k)||k<1||k>points.length)throw Error('Invalid k');
 const meanX=points.reduce((s,p)=>s+p.x,0)/points.length,meanY=points.reduce((s,p)=>s+p.y,0)/points.length;
 const sx=Math.sqrt(points.reduce((s,p)=>s+(p.x-meanX)**2,0)/points.length)||1,sy=Math.sqrt(points.reduce((s,p)=>s+(p.y-meanY)**2,0)/points.length)||1;
 const rows=points.map((p,i)=>({i,point:p,dx:(p.x-query.x)*stretch/(scaled?sx*stretch:1),dy:(p.y-query.y)/(scaled?sy:1),distance:Math.hypot((p.x-query.x)*stretch/(scaled?sx*stretch:1),(p.y-query.y)/(scaled?sy:1))})).sort((a,b)=>a.distance-b.distance||a.i-b.i);
 const votes=[0,0];for(const row of rows.slice(0,k))votes[row.point.label??0]++;
 return {rows,votes,prediction:votes[1]>votes[0]?1:0,tie:votes[0]===votes[1]};
}
