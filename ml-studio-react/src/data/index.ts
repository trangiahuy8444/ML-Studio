import courseJson from './course.json';import housingJson from './housing.json';import type {Course,NotebookData} from '../types';
export const course=courseJson as unknown as Course;
export const housing=housingJson as unknown as {rows:number[][];headers:string[];notebook:NotebookData};
