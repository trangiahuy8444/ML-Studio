/* Browser worker kernel. Pin a documented release; Python executes off the UI thread. */
function mlPythonWorker() {
  'use strict';
  let py=null, execute=null, booting=false;
  const PYODIDE_URL='https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
  const setup=String.raw`
import ast as _ml_ast, base64 as _ml_b64, contextlib as _ml_context
import io as _ml_io, json as _ml_json, os as _ml_os, traceback as _ml_tb
_ml_os.environ['MPLBACKEND'] = 'Agg'
import numpy as _ml_np
import matplotlib as _ml_mpl
_ml_mpl.use('Agg')
import matplotlib.pyplot as _ml_plt
_ml_ns = {'__name__': '__main__', '__builtins__': __builtins__}
_ml_count = 0
_ml_outputs = []

def _ml_capture_show(*args, **kwargs):
    global _ml_outputs
    for _ml_n in _ml_plt.get_fignums():
        _ml_fig = _ml_plt.figure(_ml_n)
        _ml_buffer = _ml_io.BytesIO()
        _ml_fig.savefig(_ml_buffer, format='png', bbox_inches='tight', dpi=110)
        _ml_outputs.append({'output_type': 'display_data', 'data': {
            'image/png': _ml_b64.b64encode(_ml_buffer.getvalue()).decode('ascii')}, 'metadata': {}})
    _ml_plt.close('all')
_ml_plt.show = _ml_capture_show

def _ml_display(*objects):
    for obj in objects:
        if isinstance(obj, _ml_mpl.figure.Figure):
            _ml_capture_show()
        else:
            print(repr(obj))
_ml_ns['display'] = _ml_display

def _ml_execute(source):
    global _ml_count, _ml_outputs
    _ml_count += 1
    _ml_outputs = []
    _ml_stream = _ml_io.StringIO()
    _ml_error = None
    with _ml_context.redirect_stdout(_ml_stream), _ml_context.redirect_stderr(_ml_stream):
        try:
            tree = _ml_ast.parse(source, filename=f'<notebook-cell-{_ml_count}>', mode='exec')
            if tree.body and isinstance(tree.body[-1], _ml_ast.Expr):
                last = tree.body.pop()
                exec(compile(tree, '<notebook>', 'exec'), _ml_ns)
                value = eval(compile(_ml_ast.Expression(last.value), '<notebook>', 'eval'), _ml_ns)
                if value is not None:
                    _ml_outputs.append({'output_type': 'execute_result', 'execution_count': _ml_count,
                                        'data': {'text/plain': repr(value)}, 'metadata': {}})
            else:
                exec(compile(tree, '<notebook>', 'exec'), _ml_ns)
        except BaseException as exc:
            _ml_error = {'output_type': 'error', 'ename': type(exc).__name__,
                         'evalue': str(exc), 'traceback': _ml_tb.format_exc().splitlines()}
        try:
            if _ml_plt.get_fignums():
                _ml_capture_show()
        except BaseException as exc:
            if _ml_error is None:
                _ml_error = {'output_type': 'error', 'ename': type(exc).__name__,
                             'evalue': str(exc), 'traceback': _ml_tb.format_exc().splitlines()}
    outputs = []
    if _ml_stream.getvalue():
        outputs.append({'output_type': 'stream', 'name': 'stdout',
                        'text': _ml_stream.getvalue().splitlines(keepends=True)})
    outputs.extend(_ml_outputs)
    if _ml_error:
        outputs.append(_ml_error)
    return _ml_json.dumps({'ok': _ml_error is None, 'execution_count': _ml_count,
                          'outputs': outputs, 'has_plot_epoch': callable(_ml_ns.get('plot_epoch'))})

def _ml_set_web_dataset(payload):
    data = _ml_json.loads(payload)
    _ml_ns['X_web'] = _ml_np.array(data['X'], dtype=float)
    _ml_ns['y_web'] = _ml_np.array(data['y'], dtype=float)
    _ml_ns['feature_names_web'] = data['features']
    _ml_ns['target_name_web'] = data['target']
    return len(data['y'])
`;
  self.onmessage=async ({data})=>{
    const {id,type}=data;
    try{
      if(type==='init'){
        if(booting||py)throw Error('Kernel đang được khởi tạo.');booting=true;
        self.postMessage({type:'progress',text:'Đang tải Python WebAssembly…'});
        importScripts(PYODIDE_URL+'pyodide.js');
        py=await self.loadPyodide({indexURL:PYODIDE_URL,stdout:()=>{},stderr:()=>{},stdin:()=>{throw Error('input() chưa được hỗ trợ. Hãy đặt giá trị vào biến trong ô mã.')}});
        self.postMessage({type:'progress',text:'Đang tải NumPy, Matplotlib và thư viện phụ thuộc…'});
        await py.loadPackage(['numpy','matplotlib'],{messageCallback:text=>self.postMessage({type:'progress',text:'Tải thư viện: '+text}),errorCallback:text=>self.postMessage({type:'progress',text:'Thông báo tải: '+text})});
        py.runPython(setup);execute=py.globals.get('_ml_execute');
        py.FS.writeFile('/home/pyodide/iris.csv',data.iris||'');
        self.postMessage({id,ok:true,result:{ready:true,version:py.version}});return;
      }
      if(!py||!execute)throw Error('Python chưa sẵn sàng.');
      if(type==='run'){
        // Synchronous Python stays inside this worker, avoiding UI blocking / JSPI.
        const result=JSON.parse(execute(data.code));self.postMessage({id,ok:true,result});
      }else if(type==='dataset'){
        const setter=py.globals.get('_ml_set_web_dataset');const n=setter(JSON.stringify(data.dataset));setter.destroy();
        self.postMessage({id,ok:true,result:{rows:n}});
      }else if(type==='file'){
        const name=String(data.name).replace(/[^a-zA-Z0-9_.-]/g,'_');
        if(!name||name==='.'||name==='..')throw Error('Tên tệp không hợp lệ.');
        py.FS.writeFile('/home/pyodide/'+name,new Uint8Array(data.bytes));
        self.postMessage({id,ok:true,result:{name}});
      }else throw Error('Yêu cầu kernel không hợp lệ.');
    }catch(error){self.postMessage({id,ok:false,error:error?.message||String(error)});}
  };
}
window.ML_KERNEL_WORKER_SOURCE='('+mlPythonWorker.toString()+')();';
