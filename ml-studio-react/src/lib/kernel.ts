import type {KernelResult} from '../types';
import {housing} from '../data';
interface WorkerResponse {id?:number;type?:string;text?:string;ok?:boolean;result?:unknown;error?:string}
const REAL_ESTATE_CSV = ['transaction_date,age_years,mrt_distance_m,stores,latitude,longitude,price_10k_ntd_per_ping', ...housing.rows.map(r => r.slice(1).join(','))].join('\n') + '\n';
export class PythonKernel {
 private worker:Worker|null=null;private sequence=0;private pending=new Map<number,{resolve:(v:unknown)=>void;reject:(e:Error)=>void}>();private boot:Promise<void>|null=null;
 ready=false;
 constructor(private progress:(text:string)=>void){}
 async start(){if(this.ready)return;if(this.boot)return this.boot;this.boot=(async()=>{const url=URL.createObjectURL(new Blob([window.ML_KERNEL_WORKER_SOURCE],{type:'application/javascript'}));this.worker=new Worker(url);URL.revokeObjectURL(url);this.worker.onmessage=({data}:{data:WorkerResponse})=>{if(data.type==='progress'){this.progress(data.text||'Đang tải Python…');return}if(data.id===undefined)return;const p=this.pending.get(data.id);if(!p)return;this.pending.delete(data.id);data.ok?p.resolve(data.result):p.reject(Error(data.error||'Python không phản hồi.'))};this.worker.onerror=e=>{e.preventDefault();for(const p of this.pending.values())p.reject(Error(e.message||'Không tải được Python. Kiểm tra Internet rồi thử lại.'));this.pending.clear()};try{await this.request('init',{iris:window.IRIS_CSV,realEstate:REAL_ESTATE_CSV});this.ready=true;this.progress('Python sẵn sàng.')}catch(e){this.dispose();throw e}})();return this.boot}
 private request(type:string,payload:Record<string,unknown>={}):Promise<unknown>{return new Promise((resolve,reject)=>{if(!this.worker){reject(Error('Kernel chưa khởi động.'));return}const id=++this.sequence;this.pending.set(id,{resolve,reject});this.worker.postMessage({id,type,...payload})})}
 async run(code:string):Promise<KernelResult>{await this.start();return await this.request('run',{code}) as KernelResult}
 async reset(){if(!this.ready)return;await this.request('reset');this.progress('Đã khởi động lại biến Python.')}
 async file(name:string,bytes:ArrayBuffer){await this.start();return this.request('file',{name,bytes})}
 async dataset(X:number[][],y:number[],features:string[],target:string){await this.start();return this.request('dataset',{dataset:{X,y,features,target}})}
 dispose(){this.worker?.terminate();this.worker=null;this.ready=false;this.boot=null;for(const p of this.pending.values())p.reject(Error('Đã dừng kernel và xóa biến Python.'));this.pending.clear()}
}
