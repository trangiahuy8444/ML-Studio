export interface TreeSample {id:number;x:number[];y:0|1}
export interface SplitCandidate {feature:number;threshold:number;left:number[];right:number[];leftCounts:number[];rightCounts:number[];leftGini:number;rightGini:number;weighted:number;gain:number;valid:boolean}
export interface TreeNode {id:string;depth:number;indices:number[];counts:number[];prediction:0|1;status:'pending'|'branch'|'leaf';feature?:number;threshold?:number;reason?:string}
export interface TreeFrame {phase:'evaluate'|'choose'|'split'|'leaf';nodeId:string;nodes:TreeNode[];candidates:SplitCandidate[];candidate:number;best:number;reason:string}
export function treeSamples(flipped=false):TreeSample[]{
 const area=[40,55,65,70,80,90,100,110,120,130],age=[30,5,20,32,8,35,15,40,10,3];
 return area.map((v,i)=>({id:i+1,x:[v,age[i]],y:((v>=80&&age[i]<=20)!==(flipped&&i===9)?1:0) as 0|1}));
}
export function classCounts(rows:TreeSample[],indices:number[]){return [0,1].map(c=>indices.filter(i=>rows[i].y===c).length)}
export function gini(counts:number[]){const n=counts.reduce((s,v)=>s+v,0);return n?1-counts.reduce((s,v)=>s+(v/n)**2,0):0}
export function splitCandidates(rows:TreeSample[],indices:number[],features=[0,1],minLeaf=1):SplitCandidate[]{
 const parent=gini(classCounts(rows,indices)),n=indices.length,out:SplitCandidate[]=[];
 for(const feature of features){
  const values=[...new Set(indices.map(i=>rows[i].x[feature]))].sort((a,b)=>a-b);
  for(let j=0;j<values.length-1;j++){
   const threshold=values[j]+(values[j+1]-values[j])/2,left=indices.filter(i=>rows[i].x[feature]<=threshold),right=indices.filter(i=>rows[i].x[feature]>threshold);
   const leftCounts=classCounts(rows,left),rightCounts=classCounts(rows,right),leftGini=gini(leftCounts),rightGini=gini(rightCounts),weighted=(left.length*leftGini+right.length*rightGini)/n;
   out.push({feature,threshold,left,right,leftCounts,rightCounts,leftGini,rightGini,weighted,gain:parent-weighted,valid:left.length>=minLeaf&&right.length>=minLeaf});
  }
 }
 return out;
}
export function decisionTreeTrace(rows:TreeSample[],maxDepth=2,minLeaf=1,features=[0,1]){
 if(!rows.length||!Number.isInteger(minLeaf)||minLeaf<1||!Number.isInteger(maxDepth)||maxDepth<0||!features.length)throw Error('Invalid tree configuration');
 const nodes:TreeNode[]=[],frames:TreeFrame[]=[];
 const snapshot=(phase:TreeFrame['phase'],nodeId:string,candidates:SplitCandidate[],candidate=-1,best=-1,reason='')=>frames.push({phase,nodeId,nodes:nodes.map(n=>({...n,indices:[...n.indices],counts:[...n.counts]})),candidates:structuredClone(candidates),candidate,best,reason});
 function visit(id:string,indices:number[],depth:number){
  const counts=classCounts(rows,indices),existing=nodes.find(n=>n.id===id),node:TreeNode=existing??{id,depth,indices:[...indices],counts,prediction:counts[1]>counts[0]?1:0,status:'pending'};if(!existing)nodes.push(node);
  let reason=counts.some(c=>c===indices.length)?'Nút thuần nhất':depth>=maxDepth?'Đã đạt độ sâu tối đa':indices.length<2*minLeaf?'Không đủ mẫu cho hai lá':'';
  if(reason){node.status='leaf';node.reason=reason;snapshot('leaf',id,[],-1,-1,reason);return}
  const candidates=splitCandidates(rows,indices,features,minLeaf);let best=-1;
  for(let i=0;i<candidates.length;i++){
   const c=candidates[i];if(c.valid&&(best<0||c.gain>candidates[best].gain+1e-12))best=i;
   snapshot('evaluate',id,candidates,i,best,c.valid?'Tính Gini sau chia có trọng số':'Loại: số mẫu của ít nhất một lá dưới giới hạn');
  }
  if(best<0||candidates[best].gain<=1e-12){node.status='leaf';reason=best<0?'Không có phép chia hợp lệ':'Không có phép chia làm giảm Gini';node.reason=reason;snapshot('leaf',id,candidates,-1,best,reason);return}
  snapshot('choose',id,candidates,best,best,'Chọn phép chia hợp lệ có gain lớn nhất; khi hòa chọn ứng viên xuất hiện trước');
  const chosen=candidates[best];node.status='branch';node.feature=chosen.feature;node.threshold=chosen.threshold;
  // Children appear as pending until their own training frames are visited.
  for(const [suffix,subset]of [['L',chosen.left],['R',chosen.right]] as const){const cs=classCounts(rows,[...subset]);nodes.push({id:id+suffix,depth:depth+1,indices:[...subset],counts:cs,prediction:cs[1]>cs[0]?1:0,status:'pending'})}
  snapshot('split',id,candidates,best,best,'Tạo hai nút con; sau đó huấn luyện từng nút');
  visit(id+'L',chosen.left,depth+1);visit(id+'R',chosen.right,depth+1);
 }
 visit('r',rows.map((_,i)=>i),0);
 return {frames,nodes:nodes.map(n=>({...n}))};
}
export function treePredict(nodes:TreeNode[],x:number[]){
 const path:string[]=[];let id='r',node=nodes.find(n=>n.id===id);
 while(node){path.push(node.id);if(node.status!=='branch'||node.feature===undefined||node.threshold===undefined)break;id=node.id+(x[node.feature]<=node.threshold?'L':'R');const next=nodes.find(n=>n.id===id);if(!next)break;node=next}
 return {prediction:node?.prediction??0,path,provisional:node?.status==='pending'};
}

export interface Vec2 {x:number;y:number}
export const pcaSamples=(stretch=1,outlier=false):Vec2[]=>{
 const x=[-3,-2,-1,0,1,2,3,4],y=[-3.2,-2.8,-.9,.4,.9,2.7,3.1,4.8];
 return [...x.map((v,i)=>({x:v,y:y[i]*stretch})),...(outlier?[{x:4,y:-5*stretch}]:[])];
};
export function pca2D(points:Vec2[],scaled=false){
 if(points.length<2||points.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)))throw Error('PCA needs at least two finite samples');
 const n=points.length,mean={x:points.reduce((s,p)=>s+p.x,0)/n,y:points.reduce((s,p)=>s+p.y,0)/n};
 const scale=scaled?{x:Math.sqrt(points.reduce((s,p)=>s+(p.x-mean.x)**2,0)/n)||1,y:Math.sqrt(points.reduce((s,p)=>s+(p.y-mean.y)**2,0)/n)||1}:{x:1,y:1};
 const centered=points.map(p=>({x:(p.x-mean.x)/scale.x,y:(p.y-mean.y)/scale.y}));
 const xx=centered.reduce((s,p)=>s+p.x*p.x,0)/(n-1),xy=centered.reduce((s,p)=>s+p.x*p.y,0)/(n-1),yy=centered.reduce((s,p)=>s+p.y*p.y,0)/(n-1);
 const disc=Math.hypot(xx-yy,2*xy),values=[Math.max(0,(xx+yy+disc)/2),Math.max(0,(xx+yy-disc)/2)],angle=.5*Math.atan2(2*xy,xx-yy);
 const vectors=[{x:Math.cos(angle),y:Math.sin(angle)},{x:-Math.sin(angle),y:Math.cos(angle)}],total=xx+yy;
 return {mean,scale,centered,covariance:[[xx,xy],[xy,yy]],values,vectors,total,ratio:total?values.map(v=>v/total):[0,0]};
}
export function pcaProjection(points:Vec2[],direction:Vec2,components=1){
 const norm=Math.hypot(direction.x,direction.y);if(norm<1e-12||![1,2].includes(components))throw Error('Invalid projection');
 const u={x:direction.x/norm,y:direction.y/norm},v={x:-u.y,y:u.x};
 const scores=points.map(p=>[p.x*u.x+p.y*u.y,p.x*v.x+p.y*v.y]);
 const reconstructed=scores.map(s=>({x:s[0]*u.x+(components===2?s[1]*v.x:0),y:s[0]*u.y+(components===2?s[1]*v.y:0)}));
 const mse=points.reduce((s,p,i)=>s+(p.x-reconstructed[i].x)**2+(p.y-reconstructed[i].y)**2,0)/points.length;
 const mean=scores.reduce((s,p)=>s+p[0],0)/points.length,variance=scores.reduce((s,p)=>s+(p[0]-mean)**2,0)/(points.length-1);
 return {scores,reconstructed,mse,variance,u,v};
}
