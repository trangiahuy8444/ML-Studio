export type Matrix=number[][];
export function matrixShape(a:Matrix){if(!a.length||!a[0].length||a.some(r=>r.length!==a[0].length||r.some(v=>!Number.isFinite(v))))throw Error('Ma trận phải chữ nhật, hữu hạn và không rỗng');return [a.length,a[0].length] as const}
export function convolution(image:Matrix,kernel:Matrix,stride=1,padding=0,bias=0){
 const [h,w]=matrixShape(image),[kh,kw]=matrixShape(kernel);if(!Number.isInteger(stride)||stride<1||!Number.isInteger(padding)||padding<0||!Number.isFinite(bias))throw Error('Cấu hình không hợp lệ');
 const oh=Math.floor((h+2*padding-kh)/stride)+1,ow=Math.floor((w+2*padding-kw)/stride)+1;if(oh<1||ow<1)throw Error('Kernel vượt ảnh');
 const windows=Array.from({length:oh*ow},(_,n)=>{const row=Math.floor(n/ow),col=n%ow;const terms=kernel.flatMap((r,i)=>r.map((weight,j)=>{const y=row*stride+i-padding,x=col*stride+j-padding;const inside=y>=0&&y<h&&x>=0&&x<w;const pixel=inside?image[y][x]:0;return {y,x,inside,pixel,weight,product:pixel*weight}}));const sum=terms.reduce((s,t)=>s+t.product,0)+bias;return {row,col,terms,sum,relu:Math.max(0,sum)}});
 return {height:oh,width:ow,windows,output:Array.from({length:oh},(_,i)=>windows.slice(i*ow,(i+1)*ow).map(v=>v.sum))};
}
export function maxPool(image:Matrix){const [h,w]=matrixShape(image);if(h<2||w<2)throw Error('Pooling cần ít nhất 2×2');return Array.from({length:Math.floor(h/2)},(_,i)=>Array.from({length:Math.floor(w/2)},(_,j)=>Math.max(...[image[2*i][2*j],image[2*i+1][2*j],image[2*i][2*j+1],image[2*i+1][2*j+1]])))}
export function matmul(a:Matrix,b:Matrix):Matrix{const [h,d]=matrixShape(a),[d2,w]=matrixShape(b);if(d!==d2)throw Error('Kích thước nhân không khớp');return Array.from({length:h},(_,i)=>Array.from({length:w},(_,j)=>a[i].reduce((s,v,k)=>s+v*b[k][j],0)))}
export function softmax(values:number[]){if(!values.length||values.some(v=>Number.isNaN(v)||v===Infinity))throw Error('Score không hợp lệ');const m=Math.max(...values);if(m===-Infinity)throw Error('Không có key được phép');const e=values.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
export function attention(X:Matrix,Wq:Matrix,Wk:Matrix,Wv:Matrix,causal=false,temperature=1){if(!Number.isFinite(temperature)||temperature<=0)throw Error('Temperature phải dương');const Q=matmul(X,Wq),K=matmul(X,Wk),V=matmul(X,Wv);if(Q[0].length!==K[0].length)throw Error('Chiều Q K khác nhau');const scale=Math.sqrt(Q[0].length);const dots=Q.map(q=>K.map(k=>q.reduce((s,v,j)=>s+v*k[j],0)));const scores=dots.map((r,i)=>r.map((s,j)=>causal&&j>i?-Infinity:s/scale/temperature));const weights=scores.map(softmax);const output=matmul(weights,V);return {Q,K,V,dots,scores,weights,output,scale}}
export const toyTokens=['Em','học','AI','hôm nay'];
export const toyX:Matrix=[[1,0],[0,1],[1,1],[-1,.5]];
export const toyWq:Matrix=[[1,.5],[0,1]];
export const toyWk:Matrix=[[.5,1],[1,0]];
export const toyWv:Matrix=[[1,-1],[.5,1]];
export const toyImage:Matrix=[[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]];
export const toyKernels:Record<string,Matrix>={vertical:[[-1,0,1],[-1,0,1],[-1,0,1]],horizontal:[[-1,-1,-1],[0,0,0],[1,1,1]],blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]};
