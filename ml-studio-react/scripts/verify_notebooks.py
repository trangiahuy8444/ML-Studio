"""Run all learner notebooks in fresh real kernels, without editing learners' copies.
Usage: python scripts/verify_notebooks.py --output /tmp/ml-notebook-verification
Requires nbclient, nbformat, ipykernel, numpy, matplotlib. Output stays outside source.
"""
import argparse,ast,copy,json,os,platform,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
 output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
 for env,child in [('MPLCONFIGDIR','mpl'),('JUPYTER_DATA_DIR','data'),('IPYTHONDIR','ipython'),('XDG_CACHE_HOME','cache')]:(output/child).mkdir(parents=True,exist_ok=True);os.environ[env]=str(output/child)
 import nbformat
 from nbclient import NotebookClient
 from jupyter_client import KernelManager
 from jupyter_client.kernelspec import KernelSpecManager
 course_path=Path(__file__).resolve().parents[1]/'src/data/course.json';course=json.loads(course_path.read_text())
 spec=output/'kernels/python3';spec.mkdir(parents=True,exist_ok=True)
 (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python 3 verification','language':'python'}))
 def run(day,mode):
  start=time.monotonic();result={'day':day['id'],'title':day['title'],'mode':mode,'passed':False};book=nbformat.from_dict(copy.deepcopy(day['notebook']))
  try:
   nbformat.validate(book);matched=0
   for c in book.cells:
    if c.cell_type!='code':continue
    if isinstance(c.source,list):c.source=''.join(c.source)
    c.outputs=[];c.execution_count=None
    if mode!='learner' and 'answer = None' in c.source:
     assert c.source.count('answer = None')==1
     replacement=day['practice']['solution']
     if mode=='wrong-answer':replacement+='\n'+'''def _verification_wrong(a):
    if isinstance(a,str): return 'WRONG_ANSWER'
    if isinstance(a,list): return [_verification_wrong(a[0]),*a[1:]] if a else [999]
    if isinstance(a,tuple): return tuple(_verification_wrong(list(a)))
    return a + 123456
answer = _verification_wrong(answer)'''
     c.source=c.source.replace('answer = None',replacement);matched+=1
    ast.parse(c.source)
   if mode!='learner':assert matched==1, f'Expected one exercise placeholder, got {matched}'
   km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(output/'kernels')]))
   try:NotebookClient(book,km=km,timeout=90,allow_errors=mode=='wrong-answer',resources={'metadata':{'path':str(output)}}).execute()
   finally:
    if km.has_kernel:km.shutdown_kernel(now=True)
   errors=[{'name':o.ename,'value':o.evalue,'cell':i} for i,c in enumerate(book.cells) if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
   streams='\n'.join(o.text for c in book.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='stream')
   if mode=='wrong-answer':assert len(errors)==1 and errors[0]['name']=='AssertionError',f'Wrong answer must cause exactly one AssertionError: {errors}'
   else:assert not errors,errors
   if mode=='learner':assert 'CHƯA HOÀN THÀNH' in streams and 'ĐẠT: kết quả bài B khớp kiểm tra' not in streams
   if mode=='solution':assert 'ĐẠT: kết quả bài B khớp kiểm tra' in streams and 'CHƯA HOÀN THÀNH' not in streams
   result.update(passed=True,code_cells=sum(c.cell_type=='code' for c in book.cells),figures=sum('image/png' in o.get('data',{}) for c in book.cells if c.cell_type=='code' for o in c.outputs),expected_rejection=mode=='wrong-answer')
  except Exception as e:
   result['error']=str(e);(output/f"error-{day['id']:02}-{mode}.txt").write_text(traceback.format_exc())
  finally:
   result['seconds']=round(time.monotonic()-start,2);nbformat.write(book,output/f"day-{day['id']:02}-{mode}.ipynb")
  return result
 tasks=[(day,mode) for day in course['days'] for mode in ['learner','solution','wrong-answer']];results=[]
 with ThreadPoolExecutor(max_workers=max(1,min(args.workers,3))) as pool:
  future={pool.submit(run,day,mode):(day['id'],mode) for day,mode in tasks}
  for f in as_completed(future):
   r=f.result();results.append(r);print(f"{'PASS' if r['passed'] else 'FAIL'} day {r['day']:02} {r['mode']} ({r['seconds']}s)",flush=True)
   (output/'progress.json').write_text(json.dumps(sorted(results,key=lambda r:(r['day'],r['mode'])),ensure_ascii=False,indent=2))
 summary={'python':sys.version.split()[0],'platform':platform.platform(),'course':str(course_path),'results':sorted(results,key=lambda r:(r['day'],r['mode'])),'passed':sum(r['passed'] for r in results),'total':len(tasks),'note':'Fresh real Python kernel for every case. Learner copies are unchanged. Wrong-answer mode passes only when exercise rejects the answer with AssertionError. This does not evaluate free-form A/C work or UI appearance.'}
 import importlib.metadata
 summary['packages']={p:importlib.metadata.version(p) for p in ['numpy','matplotlib','nbclient','nbformat','ipykernel']}
 (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(f"RESULT {summary['passed']}/{summary['total']}",flush=True)
 return 0 if summary['passed']==summary['total'] else 1
if __name__=='__main__':sys.exit(main())
