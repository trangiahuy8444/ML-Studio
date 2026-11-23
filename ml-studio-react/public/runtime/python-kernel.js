/* Browser worker kernel. Pin a documented release; Python executes off the UI thread. */
function mlPythonWorker() {
  'use strict';
  let py=null, execute=null, booting=false, torchInstalled=false;
  const loadedPkgs=new Set(['numpy','matplotlib']);
  const PYODIDE_URL='https://cdn.jsdelivr.net/pyodide/v0.28.2/full/';
  const setup=String.raw`
import ast as _ml_ast, base64 as _ml_b64, contextlib as _ml_context
import io as _ml_io, json as _ml_json, linecache as _ml_linecache, os as _ml_os, traceback as _ml_tb
_ml_os.makedirs('/home/pyodide/data', exist_ok=True)
_ml_os.chdir('/home/pyodide')
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

def _ml_reset():
    global _ml_count
    _ml_plt.close('all')
    _ml_ns.clear()
    _ml_ns.update({'__name__': '__main__', '__builtins__': __builtins__, 'display': _ml_display})
    _ml_count = 0

def _ml_execute(source):
    global _ml_count, _ml_outputs
    _ml_count += 1
    _ml_outputs = []
    _ml_stream = _ml_io.StringIO()
    _ml_error = None
    _ml_filename = f'<notebook-cell-{_ml_count}>'
    _ml_linecache.cache[_ml_filename] = (len(source), None, source.splitlines(True), _ml_filename)
    with _ml_context.redirect_stdout(_ml_stream), _ml_context.redirect_stderr(_ml_stream):
        try:
            tree = _ml_ast.parse(source, filename=_ml_filename, mode='exec')
            if tree.body and isinstance(tree.body[-1], _ml_ast.Expr):
                last = tree.body.pop()
                exec(compile(tree, _ml_filename, 'exec'), _ml_ns)
                value = eval(compile(_ml_ast.Expression(last.value), _ml_filename, 'eval'), _ml_ns)
                if value is not None:
                    _ml_outputs.append({'output_type': 'execute_result', 'execution_count': _ml_count,
                                        'data': {'text/plain': repr(value)}, 'metadata': {}})
            else:
                exec(compile(tree, _ml_filename, 'exec'), _ml_ns)
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
  const torchShim=String.raw`
import math
import types
import sys
import numpy as np

if 'torch' not in sys.modules:
    _no_grad_depth = 0
    _rng_state = np.random.default_rng(0)

    class _NoGrad:
        def __enter__(self):
            global _no_grad_depth
            _no_grad_depth += 1
            return self
        def __exit__(self, *args):
            global _no_grad_depth
            _no_grad_depth -= 1

    def no_grad():
        return _NoGrad()

    def manual_seed(seed):
        global _rng_state
        _rng_state = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        return _rng_state

    class Generator:
        def __init__(self):
            self._rng = np.random.default_rng()
        def manual_seed(self, seed):
            self._rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
            return self

    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    bool = np.bool_

    def _to_np(x):
        return x.data if isinstance(x, Tensor) else np.asarray(x)

    def _unbroadcast(grad, shape):
        if grad.shape == shape:
            return grad
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    class Tensor:
        def __init__(self, data, dtype=None, requires_grad=False, _prev=(), _op=''):
            if isinstance(data, Tensor):
                arr = np.array(data.data, dtype=dtype if dtype is not None else data.data.dtype, copy=True)
            else:
                arr = np.asarray(data, dtype=dtype)
                if arr.dtype == np.float64 and dtype is None:
                    arr = arr.astype(np.float32)
            self.data = arr
            self.requires_grad = bool(requires_grad)
            self.grad = None
            self._prev = tuple(_prev)
            self._backward = lambda: None
            self._op = _op

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

        @property
        def ndim(self):
            return self.data.ndim

        @property
        def T(self):
            return self.transpose(-2, -1) if self.data.ndim >= 2 else self

        def __len__(self):
            return len(self.data)

        def __repr__(self):
            return f"tensor({self.data!r})"

        def numel(self):
            return int(self.data.size)

        def item(self):
            return self.data.item()

        def __float__(self):
            return float(self.data.item())

        def __int__(self):
            return int(self.data.item())

        def detach(self):
            return Tensor(self.data.copy(), dtype=self.data.dtype, requires_grad=False)

        def numpy(self):
            return self.data

        def tolist(self):
            return self.data.tolist()

        def float(self):
            return Tensor(self.data.astype(np.float32), dtype=np.float32, requires_grad=(_no_grad_depth == 0 and self.requires_grad))

        def abs(self):
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(np.abs(self.data), dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(out.grad.data * np.sign(self.data))
            return out

        def _acc_grad(self, g):
            if not self.requires_grad or g is None:
                return
            g = np.asarray(g, dtype=self.data.dtype)
            if self.grad is None:
                self.grad = Tensor(g.copy(), dtype=self.data.dtype)
            else:
                self.grad.data = self.grad.data + g

        def backward(self):
            order, seen = [], set()
            def visit(n):
                if id(n) not in seen:
                    seen.add(id(n))
                    for p in n._prev:
                        visit(p)
                    order.append(n)
            visit(self)
            self.grad = Tensor(np.ones_like(self.data), dtype=self.data.dtype)
            for node in reversed(order):
                if node.grad is not None:
                    node._backward()

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
            elif isinstance(idx, Tensor):
                idx = idx.data
            res = self.data[idx]
            if np.isscalar(res):
                res = np.asarray(res, dtype=self.data.dtype)
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(res, dtype=res.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                def _b():
                    g = np.zeros_like(self.data)
                    np.add.at(g, idx, out.grad.data)
                    self._acc_grad(g)
                out._backward = _b
            return out

        def __setitem__(self, idx, val):
            if isinstance(idx, tuple):
                idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
            elif isinstance(idx, Tensor):
                idx = idx.data
            self.data[idx] = _to_np(val)

        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(self.data.reshape(shape), dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                orig_shape = self.data.shape
                out._backward = lambda: self._acc_grad(out.grad.data.reshape(orig_shape))
            return out

        reshape = view

        def transpose(self, dim0, dim1):
            axes = list(range(self.data.ndim))
            d0 = dim0 if dim0 >= 0 else self.data.ndim + dim0
            d1 = dim1 if dim1 >= 0 else self.data.ndim + dim1
            axes[d0], axes[d1] = axes[d1], axes[d0]
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(np.transpose(self.data, axes), dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(np.transpose(out.grad.data, axes))
            return out

        def flatten(self, start_dim=0, end_dim=-1):
            nd = self.data.ndim
            sd = start_dim if start_dim >= 0 else nd + start_dim
            ed = end_dim if end_dim >= 0 else nd + end_dim
            new_shape = self.data.shape[:sd] + (-1,) + self.data.shape[ed+1:]
            return self.view(*new_shape)

        def chunk(self, chunks, dim=0):
            d = dim if dim >= 0 else self.data.ndim + dim
            n = self.data.shape[d]
            step = math.ceil(n / chunks)
            out = []
            for start in range(0, n, step):
                sl = [slice(None)] * self.data.ndim
                sl[d] = slice(start, min(n, start + step))
                out.append(self[tuple(sl)])
            return tuple(out)

        def sum(self, dim=None, keepdim=False):
            req = (_no_grad_depth == 0) and self.requires_grad
            res = self.data.sum(axis=dim, keepdims=keepdim)
            if np.isscalar(res):
                res = np.asarray(res, dtype=self.data.dtype)
            out = Tensor(res, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                def _b():
                    g = out.grad.data
                    if dim is not None and not keepdim:
                        axes = (dim,) if isinstance(dim, int) else tuple(dim)
                        for ax in sorted(a if a >= 0 else self.data.ndim + a for a in axes):
                            g = np.expand_dims(g, axis=ax)
                    self._acc_grad(np.broadcast_to(g, self.data.shape))
                out._backward = _b
            return out

        def mean(self, dim=None, keepdim=False):
            s = self.sum(dim=dim, keepdim=keepdim)
            denom = self.data.size if dim is None else int(np.prod([self.data.shape[a] for a in ((dim,) if isinstance(dim, int) else dim)]))
            return s / denom

        def amax(self, dim=None, keepdim=False):
            return Tensor(self.data.max(axis=dim, keepdims=keepdim), dtype=self.data.dtype)

        def argmax(self, dim=None):
            return Tensor(self.data.argmax(axis=dim), dtype=np.int64)

        def masked_fill(self, mask, value):
            m = _to_np(mask).astype(np.bool_)
            res = np.where(m, value, self.data).astype(self.data.dtype)
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(res, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(np.where(m, 0, out.grad.data))
            return out

        def softmax(self, dim=-1):
            x = self.data - np.max(self.data, axis=dim, keepdims=True)
            e = np.exp(x)
            p = (e / e.sum(axis=dim, keepdims=True)).astype(self.data.dtype)
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(p, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                def _b():
                    g = out.grad.data
                    dot = (g * p).sum(axis=dim, keepdims=True)
                    self._acc_grad(p * (g - dot))
                out._backward = _b
            return out

        def triu(self, diagonal=0):
            return Tensor(np.triu(self.data, k=diagonal), dtype=self.data.dtype)

        def tril(self, diagonal=0):
            return Tensor(np.tril(self.data, k=diagonal), dtype=self.data.dtype)

        def __add__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            req = (_no_grad_depth == 0) and (self.requires_grad or o.requires_grad)
            res = self.data + o.data
            out = Tensor(res, dtype=res.dtype, requires_grad=req, _prev=(self, o) if req else ())
            if req:
                def _b():
                    if self.requires_grad: self._acc_grad(_unbroadcast(out.grad.data, self.data.shape))
                    if o.requires_grad: o._acc_grad(_unbroadcast(out.grad.data, o.data.shape))
                out._backward = _b
            return out

        __radd__ = __add__

        def __sub__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            return self + (-o)

        def __rsub__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            return o + (-self)

        def __neg__(self):
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(-self.data, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(-out.grad.data)
            return out

        def __mul__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            req = (_no_grad_depth == 0) and (self.requires_grad or o.requires_grad)
            res = self.data * o.data
            out = Tensor(res, dtype=res.dtype, requires_grad=req, _prev=(self, o) if req else ())
            if req:
                def _b():
                    if self.requires_grad: self._acc_grad(_unbroadcast(out.grad.data * o.data, self.data.shape))
                    if o.requires_grad: o._acc_grad(_unbroadcast(out.grad.data * self.data, o.data.shape))
                out._backward = _b
            return out

        __rmul__ = __mul__

        def __truediv__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            req = (_no_grad_depth == 0) and (self.requires_grad or o.requires_grad)
            res = self.data / o.data
            out = Tensor(res, dtype=res.dtype, requires_grad=req, _prev=(self, o) if req else ())
            if req:
                def _b():
                    if self.requires_grad: self._acc_grad(_unbroadcast(out.grad.data / o.data, self.data.shape))
                    if o.requires_grad: o._acc_grad(_unbroadcast(-out.grad.data * self.data / (o.data ** 2), o.data.shape))
                out._backward = _b
            return out

        def __rtruediv__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            return o.__truediv__(self)

        def __pow__(self, p):
            req = (_no_grad_depth == 0) and self.requires_grad
            pv = _to_np(p)
            res = self.data ** pv
            out = Tensor(res, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(out.grad.data * pv * (self.data ** (pv - 1)))
            return out

        def __rpow__(self, base):
            bv = _to_np(base)
            res = bv ** self.data
            req = (_no_grad_depth == 0) and self.requires_grad
            out = Tensor(res, dtype=self.data.dtype, requires_grad=req, _prev=(self,) if req else ())
            if req:
                out._backward = lambda: self._acc_grad(out.grad.data * res * np.log(bv))
            return out

        def __matmul__(self, other):
            o = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
            req = (_no_grad_depth == 0) and (self.requires_grad or o.requires_grad)
            res = self.data @ o.data
            out = Tensor(res, dtype=res.dtype, requires_grad=req, _prev=(self, o) if req else ())
            if req:
                def _b():
                    g = out.grad.data
                    if self.requires_grad:
                        o_t = np.swapaxes(o.data, -1, -2) if o.data.ndim >= 2 else o.data
                        self._acc_grad(_unbroadcast(g @ o_t, self.data.shape))
                    if o.requires_grad:
                        s_t = np.swapaxes(self.data, -1, -2) if self.data.ndim >= 2 else self.data
                        o._acc_grad(_unbroadcast(s_t @ g, o.data.shape))
                out._backward = _b
            return out

        def __invert__(self):
            return Tensor(~self.data.astype(np.bool_), dtype=np.bool_)

        def __eq__(self, other):
            return Tensor(self.data == _to_np(other), dtype=np.bool_)

        def __ne__(self, other):
            return Tensor(self.data != _to_np(other), dtype=np.bool_)

        def __lt__(self, other):
            return Tensor(self.data < _to_np(other), dtype=np.bool_)

        def __le__(self, other):
            return Tensor(self.data <= _to_np(other), dtype=np.bool_)

        def __gt__(self, other):
            return Tensor(self.data > _to_np(other), dtype=np.bool_)

        def __ge__(self, other):
            return Tensor(self.data >= _to_np(other), dtype=np.bool_)

    def tensor(data, dtype=None, requires_grad=False):
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)

    def _shape_arg(size):
        return size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size

    def zeros(*size, dtype=float32):
        return Tensor(np.zeros(_shape_arg(size), dtype=dtype), dtype=dtype)

    def ones(*size, dtype=float32):
        return Tensor(np.ones(_shape_arg(size), dtype=dtype), dtype=dtype)

    def randn(*size, dtype=float32):
        return Tensor(_rng_state.normal(size=_shape_arg(size)).astype(dtype), dtype=dtype)

    def rand(*size, dtype=float32):
        return Tensor(_rng_state.random(size=_shape_arg(size)).astype(dtype), dtype=dtype)

    def arange(start, end=None, step=1, dtype=None):
        arr = np.arange(start, end, step) if end is not None else np.arange(0, start, step)
        if dtype is None:
            dtype = float32 if isinstance(start, float) or isinstance(end, float) or isinstance(step, float) else int64
        return Tensor(arr.astype(dtype), dtype=dtype)

    def tril(x):
        return Tensor(np.tril(_to_np(x)), dtype=x.dtype)

    def triu(x, diagonal=0):
        return Tensor(np.triu(_to_np(x), k=diagonal), dtype=x.dtype)

    def sin(x):
        return Tensor(np.sin(_to_np(x)), dtype=float32)

    def cos(x):
        return Tensor(np.cos(_to_np(x)), dtype=float32)

    def cat(tensors, dim=0):
        return Tensor(np.concatenate([_to_np(t) for t in tensors], axis=dim), dtype=tensors[0].dtype)

    def equal(a, b):
        return bool(np.array_equal(_to_np(a), _to_np(b)))

    def allclose(a, b, rtol=1e-5, atol=1e-8):
        return bool(np.allclose(_to_np(a), _to_np(b), rtol=rtol, atol=atol))

    def einsum(sub, *operands):
        return Tensor(np.einsum(sub, *[_to_np(o) for o in operands]), dtype=float32)

    class Parameter(Tensor):
        def __init__(self, data, requires_grad=True):
            super().__init__(data, dtype=float32, requires_grad=requires_grad)

    class Module:
        def __init__(self):
            self._modules = {}
            self._parameters = {}
            self.training = True

        def __setattr__(self, name, value):
            if isinstance(value, Parameter):
                self._parameters[name] = value
            elif isinstance(value, Module):
                self._modules[name] = value
            object.__setattr__(self, name, value)

        def parameters(self):
            seen = set()
            out = []
            for p in self._parameters.values():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
            for m in self._modules.values():
                for p in m.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
            return out

        def modules(self):
            yield self
            for m in self._modules.values():
                yield from m.modules()

        def train(self, mode=True):
            self.training = mode
            for m in self._modules.values():
                m.train(mode)
            return self

        def eval(self):
            return self.train(False)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            for idx, layer in enumerate(layers):
                setattr(self, str(idx), layer)
            self._layers = layers

        def forward(self, x):
            for layer in self._layers:
                x = layer(x)
            return x

    def _linear_fn(x, weight, bias):
        req = (_no_grad_depth == 0) and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
        res = x.data @ weight.data.T
        if bias is not None:
            res = res + bias.data
        prev = [t for t in (x, weight, bias) if t is not None]
        out = Tensor(res, dtype=np.float32, requires_grad=req, _prev=prev if req else ())
        if req:
            def _b():
                g = out.grad.data
                if x.requires_grad:
                    x._acc_grad(g @ weight.data)
                if weight.requires_grad:
                    g2 = g.reshape(-1, g.shape[-1])
                    x2 = x.data.reshape(-1, x.data.shape[-1])
                    weight._acc_grad(g2.T @ x2)
                if bias is not None and bias.requires_grad:
                    axes = tuple(range(g.ndim - 1))
                    bias._acc_grad(g.sum(axis=axes))
            out._backward = _b
        return out

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            bound = 1.0 / math.sqrt(in_features)
            self.weight = Parameter(_rng_state.uniform(-bound, bound, (out_features, in_features)).astype(np.float32))
            self.bias = Parameter(_rng_state.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None

        def forward(self, x):
            return _linear_fn(x, self.weight, self.bias)

    class ReLU(Module):
        def forward(self, x):
            req = (_no_grad_depth == 0) and x.requires_grad
            mask = x.data > 0
            out = Tensor(np.where(mask, x.data, 0.0).astype(x.data.dtype), dtype=x.data.dtype, requires_grad=req, _prev=(x,) if req else ())
            if req:
                out._backward = lambda: x._acc_grad(np.where(mask, out.grad.data, 0.0))
            return out

    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
        def forward(self, x):
            if not self.training or self.p == 0:
                return x
            mask = (_rng_state.random(x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
            req = (_no_grad_depth == 0) and x.requires_grad
            out = Tensor(x.data * mask, dtype=x.data.dtype, requires_grad=req, _prev=(x,) if req else ())
            if req:
                out._backward = lambda: x._acc_grad(out.grad.data * mask)
            return out

    class Flatten(Module):
        def __init__(self, start_dim=1, end_dim=-1):
            super().__init__()
            self.start_dim = start_dim
            self.end_dim = end_dim
        def forward(self, x):
            return x.flatten(self.start_dim, self.end_dim)

    def _pad_np(arr, pad, mode='constant', value=0):
        np_pad = [(0, 0)] * arr.ndim
        for i in range(0, len(pad), 2):
            dim = arr.ndim - 1 - (i // 2)
            np_pad[dim] = (pad[i], pad[i + 1])
        return np.pad(arr, np_pad, mode=mode, constant_values=value)

    def _conv2d_fn(x, weight, bias=None, stride=1, padding=0):
        s = (stride, stride) if isinstance(stride, int) else tuple(stride)
        p = (padding, padding) if isinstance(padding, int) else tuple(padding)
        xp = np.pad(x.data, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1]))) if (p[0] or p[1]) else x.data
        N, C_in, H, W = xp.shape
        C_out, _, kH, kW = weight.data.shape
        Ho = (H - kH) // s[0] + 1
        Wo = (W - kW) // s[1] + 1
        win = np.lib.stride_tricks.sliding_window_view(xp, (kH, kW), axis=(2, 3))[:, :, ::s[0], ::s[1], :, :]
        col = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C_in * kH * kW)
        w_mat = weight.data.reshape(C_out, C_in * kH * kW)
        out_flat = col @ w_mat.T
        if bias is not None:
            out_flat = out_flat + bias.data
        res = out_flat.reshape(N, Ho, Wo, C_out).transpose(0, 3, 1, 2).astype(np.float32)
        req = (_no_grad_depth == 0) and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
        prev = [t for t in (x, weight, bias) if t is not None]
        out = Tensor(res, dtype=np.float32, requires_grad=req, _prev=prev if req else ())
        if req:
            def _b():
                g = out.grad.data
                g_flat = g.transpose(0, 2, 3, 1).reshape(N * Ho * Wo, C_out)
                if bias is not None and bias.requires_grad:
                    bias._acc_grad(g_flat.sum(axis=0))
                if weight.requires_grad:
                    weight._acc_grad((g_flat.T @ col).reshape(weight.data.shape))
                if x.requires_grad:
                    dx_col = (g_flat @ w_mat).reshape(N, Ho, Wo, C_in, kH, kW)
                    dxp = np.zeros_like(xp)
                    for i in range(kH):
                        for j in range(kW):
                            dxp[:, :, i:i + Ho * s[0]:s[0], j:j + Wo * s[1]:s[1]] += dx_col[:, :, :, :, i, j].transpose(0, 3, 1, 2)
                    x._acc_grad(dxp[:, :, p[0]:H - p[0], p[1]:W - p[1]] if (p[0] or p[1]) else dxp)
            out._backward = _b
        return out

    class Conv2d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
            super().__init__()
            k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
            self.stride = stride
            self.padding = padding
            bound = 1.0 / math.sqrt(in_channels * k[0] * k[1])
            self.weight = Parameter(_rng_state.uniform(-bound, bound, (out_channels, in_channels, k[0], k[1])).astype(np.float32))
            self.bias = Parameter(_rng_state.uniform(-bound, bound, (out_channels,)).astype(np.float32)) if bias else None

        def forward(self, x):
            return _conv2d_fn(x, self.weight, self.bias, self.stride, self.padding)

    def _max_pool2d_fn(x, kernel_size, stride=None, padding=0):
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s = k if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        N, C, H, W = x.data.shape
        Ho = (H - k[0]) // s[0] + 1
        Wo = (W - k[1]) // s[1] + 1
        win = np.lib.stride_tricks.sliding_window_view(x.data, k, axis=(2, 3))[:, :, ::s[0], ::s[1], :, :]
        flat = win.reshape(N, C, Ho, Wo, k[0] * k[1])
        idx = flat.argmax(axis=-1)
        res = flat.max(axis=-1).astype(np.float32)
        req = (_no_grad_depth == 0) and x.requires_grad
        out = Tensor(res, dtype=np.float32, requires_grad=req, _prev=(x,) if req else ())
        if req:
            def _b():
                g = out.grad.data
                dx_flat = np.zeros_like(flat)
                n_i, c_i, ho_i, wo_i = np.meshgrid(np.arange(N), np.arange(C), np.arange(Ho), np.arange(Wo), indexing='ij')
                dx_flat[n_i, c_i, ho_i, wo_i, idx] = g
                dx = np.zeros_like(x.data)
                dx_win = dx_flat.reshape(N, C, Ho, Wo, k[0], k[1])
                for i in range(k[0]):
                    for j in range(k[1]):
                        dx[:, :, i:i + Ho * s[0]:s[0], j:j + Wo * s[1]:s[1]] += dx_win[:, :, :, :, i, j]
                x._acc_grad(dx)
            out._backward = _b
        return out

    class MaxPool2d(Module):
        def __init__(self, kernel_size, stride=None, padding=0):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
        def forward(self, x):
            return _max_pool2d_fn(x, self.kernel_size, self.stride, self.padding)

    class CrossEntropyLoss(Module):
        def forward(self, logits, targets):
            x = logits.data
            y = _to_np(targets).astype(np.int64)
            x_shift = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x_shift)
            probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            n = len(y)
            loss_val = -np.mean(np.log(np.clip(probs[np.arange(n), y], 1e-12, None)))
            req = (_no_grad_depth == 0) and logits.requires_grad
            out = Tensor(np.asarray(loss_val, dtype=np.float32), dtype=np.float32, requires_grad=req, _prev=(logits,) if req else ())
            if req:
                def _b():
                    grad = probs.copy()
                    grad[np.arange(n), y] -= 1.0
                    grad *= (float(out.grad.data) / n)
                    logits._acc_grad(grad)
                out._backward = _b
            return out

    class MultiheadAttention(Module):
        def __init__(self, embed_dim, num_heads, batch_first=False):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.batch_first = batch_first
            bound = math.sqrt(6.0 / (embed_dim + 3 * embed_dim))
            self.in_proj_weight = Parameter(_rng_state.uniform(-bound, bound, (3 * embed_dim, embed_dim)).astype(np.float32))
            self.in_proj_bias = Parameter(np.zeros(3 * embed_dim, dtype=np.float32))
            self.out_proj = Linear(embed_dim, embed_dim)

        def forward(self, query, key, value, need_weights=True):
            q_in = query if self.batch_first else query.transpose(0, 1)
            k_in = key if self.batch_first else key.transpose(0, 1)
            v_in = value if self.batch_first else value.transpose(0, 1)
            B, T, E = q_in.shape
            h = self.num_heads
            Wq, Wk, Wv = self.in_proj_weight.chunk(3)
            bq, bk, bv = self.in_proj_bias.chunk(3)
            q = (q_in @ Tensor(Wq.data.T) + bq).view(B, T, h, E // h).transpose(1, 2)
            k = (k_in @ Tensor(Wk.data.T) + bk).view(B, k_in.shape[1], h, E // h).transpose(1, 2)
            v = (v_in @ Tensor(Wv.data.T) + bv).view(B, v_in.shape[1], h, E // h).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(E // h)
            weights = scores.softmax(dim=-1)
            ctx = (weights @ v).transpose(1, 2).reshape(B, T, E)
            out = self.out_proj(ctx)
            if not self.batch_first:
                out = out.transpose(0, 1)
            return out, (weights.mean(dim=1) if need_weights else None)

    F_mod = types.ModuleType('torch.nn.functional')
    F_mod.pad = lambda x, pad, mode='constant', value=0: Tensor(_pad_np(_to_np(x), pad, mode, value), dtype=x.dtype)
    F_mod.conv2d = _conv2d_fn
    F_mod.max_pool2d = _max_pool2d_fn
    def _avg_pool2d(x, kernel_size, stride=None):
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s = k if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        win = np.lib.stride_tricks.sliding_window_view(x.data, k, axis=(2, 3))[:, :, ::s[0], ::s[1], :, :]
        return Tensor(win.mean(axis=(-2, -1)), dtype=np.float32)
    F_mod.avg_pool2d = _avg_pool2d
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if is_causal:
            T, S = query.shape[-2], key.shape[-2]
            cm = np.tril(np.ones((T, S), dtype=np.bool_))
            scores = scores.masked_fill(Tensor(cm, dtype=np.bool_).__invert__(), float('-inf'))
        if attn_mask is not None:
            m = _to_np(attn_mask)
            if m.dtype == np.bool_:
                scores = scores.masked_fill(~m, float('-inf'))
            else:
                scores = scores + Tensor(m)
        return scores.softmax(dim=-1) @ value
    F_mod.scaled_dot_product_attention = _sdpa

    nn_mod = types.ModuleType('torch.nn')
    nn_mod.Parameter = Parameter
    nn_mod.Module = Module
    nn_mod.Sequential = Sequential
    nn_mod.Linear = Linear
    nn_mod.ReLU = ReLU
    nn_mod.Dropout = Dropout
    nn_mod.Flatten = Flatten
    nn_mod.Conv2d = Conv2d
    nn_mod.MaxPool2d = MaxPool2d
    nn_mod.CrossEntropyLoss = CrossEntropyLoss
    nn_mod.MultiheadAttention = MultiheadAttention
    nn_mod.functional = F_mod

    class Optimizer:
        def __init__(self, params):
            self.params = list(params)
        def zero_grad(self):
            for p in self.params:
                p.grad = None

    class SGD(Optimizer):
        def __init__(self, params, lr=0.01, momentum=0.0):
            super().__init__(params)
            self.lr = lr
            self.momentum = momentum
            self.state = {}
        def step(self):
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad.data
                if self.momentum != 0:
                    v = self.state.get(id(p))
                    v = g.copy() if v is None else self.momentum * v + g
                    self.state[id(p)] = v
                    g = v
                p.data = p.data - self.lr * g

    class Adam(Optimizer):
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
            super().__init__(params)
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.state = {}
        def step(self):
            b1, b2 = self.betas
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad.data
                st = self.state.setdefault(id(p), {'t': 0, 'm': np.zeros_like(p.data), 'v': np.zeros_like(p.data)})
                st['t'] += 1
                st['m'] = b1 * st['m'] + (1.0 - b1) * g
                st['v'] = b2 * st['v'] + (1.0 - b2) * (g * g)
                m_hat = st['m'] / (1.0 - b1 ** st['t'])
                v_hat = st['v'] / (1.0 - b2 ** st['t'])
                p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    optim_mod = types.ModuleType('torch.optim')
    optim_mod.SGD = SGD
    optim_mod.Adam = Adam

    class TensorDataset:
        def __init__(self, *tensors):
            self.tensors = tensors
        def __len__(self):
            return len(self.tensors[0])
        def __getitem__(self, idx):
            return tuple(t[idx] for t in self.tensors)

    class DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False, generator=None):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.generator = generator
        def __iter__(self):
            n = len(self.dataset)
            idx = np.arange(n)
            if self.shuffle:
                rng = self.generator._rng if self.generator is not None else _rng_state
                idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                yield tuple(Tensor(t.data[b], dtype=t.dtype) for t in self.dataset.tensors)
        def __len__(self):
            return math.ceil(len(self.dataset) / self.batch_size)

    data_mod = types.ModuleType('torch.utils.data')
    data_mod.TensorDataset = TensorDataset
    data_mod.DataLoader = DataLoader

    utils_mod = types.ModuleType('torch.utils')
    utils_mod.data = data_mod

    torch_mod = types.ModuleType('torch')
    torch_mod.Tensor = Tensor
    torch_mod.tensor = tensor
    torch_mod.zeros = zeros
    torch_mod.ones = ones
    torch_mod.randn = randn
    torch_mod.rand = rand
    torch_mod.arange = arange
    torch_mod.tril = tril
    torch_mod.triu = triu
    torch_mod.sin = sin
    torch_mod.cos = cos
    torch_mod.cat = cat
    torch_mod.equal = equal
    torch_mod.allclose = allclose
    torch_mod.einsum = einsum
    torch_mod.manual_seed = manual_seed
    torch_mod.Generator = Generator
    torch_mod.no_grad = no_grad
    torch_mod.float32 = float32
    torch_mod.float64 = float64
    torch_mod.int64 = int64
    torch_mod.bool = bool
    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.utils = utils_mod

    sys.modules['torch'] = torch_mod
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.nn.functional'] = F_mod
    sys.modules['torch.optim'] = optim_mod
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod
`;
  async function ensurePackages(code) {
    const needed=[];
    if(/\bpandas\b/.test(code)&&!loadedPkgs.has('pandas'))needed.push('pandas');
    if(/\b(?:sklearn|scikit_learn)\b/.test(code)&&!loadedPkgs.has('scikit-learn'))needed.push('scikit-learn');
    if(/\bscipy\b/.test(code)&&!loadedPkgs.has('scipy'))needed.push('scipy');
    if(needed.length){
      self.postMessage({type:'progress',text:'Đang tải thư viện học máy ('+needed.join(', ')+')…'});
      await py.loadPackage(needed,{messageCallback:text=>self.postMessage({type:'progress',text:'Tải thư viện: '+text})});
      for(const pkg of needed)loadedPkgs.add(pkg);
      if(needed.includes('scikit-learn'))loadedPkgs.add('scipy');
    }
    if(!torchInstalled&&/\btorch\b/.test(code)){
      py.runPython(torchShim);
      torchInstalled=true;
    }
  }
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
        if(data.realEstate)py.FS.writeFile('/home/pyodide/data/real_estate.csv',data.realEstate);
        self.postMessage({id,ok:true,result:{ready:true,version:py.version}});return;
      }
      if(!py||!execute)throw Error('Python chưa sẵn sàng.');
      if(type==='run'){
        await ensurePackages(String(data.code||''));
        // Synchronous Python stays inside this worker, avoiding UI blocking / JSPI.
        const result=JSON.parse(execute(data.code));self.postMessage({id,ok:true,result});
      }else if(type==='reset'){
        py.globals.get('_ml_reset')();
        self.postMessage({id,ok:true,result:{reset:true}});
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
