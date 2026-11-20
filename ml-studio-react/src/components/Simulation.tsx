import {useEffect,useState} from 'react';
import {AlgorithmLab} from './AlgorithmLab';
const deepDays:Record<number,'linear'|'multiple'|'logistic'|'knn'|'kmeans'|'tree'|'pca'|'neural'|'cnn'|'attention'>={6:'linear',7:'linear',8:'multiple',9:'linear',12:'logistic',13:'knn',15:'tree',17:'kmeans',18:'pca',19:'neural',20:'neural',21:'neural',22:'cnn',25:'attention'};
function BasicSimulation({day}:{day:number}){const [error,setError]=useState('');useEffect(()=>{try{window.renderLessonSimulation(day)}catch(e){setError(e instanceof Error?e.message:'Không tải được mô phỏng.')}return()=>window.cleanupLessonSimulations()},[day]);return error?<p role="alert" className="error">{error}</p>:<div id="courseDemo"/>}
export function Simulation({day}:{day:number}){return deepDays[day]?<AlgorithmLab key={day} initial={deepDays[day]} day={day}/>:<section className="simulation-studio" aria-label="Mô phỏng tương tác"><BasicSimulation day={day}/></section>}
