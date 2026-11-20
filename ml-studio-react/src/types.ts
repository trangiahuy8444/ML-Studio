import type {AssessmentState} from './lib/assessment';
export type CellType='code'|'markdown';
export interface Output {output_type:'stream'|'execute_result'|'display_data'|'error';text?:string|string[];data?:Record<string,string>;execution_count?:number;ename?:string;evalue?:string;traceback?:string[];name?:string}
export interface Cell {id:string;cell_type:CellType;source:string[];metadata:{section?:string;support?:boolean;web_dirty?:boolean;web_executed?:boolean;[key:string]:unknown};outputs?:Output[];execution_count?:number|null}
export interface NotebookData {nbformat:number;nbformat_minor:number;metadata:Record<string,unknown>;cells:Cell[]}
export interface Day {id:number;title:string;phase:number;checkpoint:boolean;bullets:{label:string;text:string}[];practice:{A:string;B:string;C:string;hint:string;solution:string};notebook:NotebookData}
export interface Phase {id:number;title:string;range:[number,number];goal:string;gate:string;color:string}
export interface Course {phases:Phase[];days:Day[];resources:{id:string;title:string;path:string;note:string}[];fullStudyPlan:string;deploymentPlan:string}
export interface DayProgress {a:boolean;b:boolean;c:boolean;review:boolean;note:string}
export interface Progress {version:1;days:Record<string,DayProgress>;assessment?:AssessmentState}
export type Route={kind:'project'}|{kind:'overview'}|{kind:'phase';id:number}|{kind:'day';id:number}|{kind:'exercises'}|{kind:'resources'}|{kind:'lab'}|{kind:'py';id?:number}|{kind:'gnn'}|{kind:'assessments'}|{kind:'guide';id?:number};
export interface PlotTrace {type:string;[key:string]:unknown}
export interface PlotApi {react:(el:HTMLElement,traces:PlotTrace[],layout:Record<string,unknown>,config:Record<string,unknown>)=>Promise<void>;purge:(el:HTMLElement)=>void;Plots:{resize:(el:HTMLElement)=>void}}
export interface KernelResult {ok:boolean;execution_count:number;outputs:Output[];has_plot_epoch:boolean}
declare global {interface Window {Plotly:PlotApi;IRIS_CSV:string;ML_KERNEL_WORKER_SOURCE:string;renderLessonSimulation:(day:number)=>void;cleanupLessonSimulations:()=>void;ML_LEGACY_HTML?:string;ML_HIGHLIGHT_PYTHON:(source:string)=>string}}
