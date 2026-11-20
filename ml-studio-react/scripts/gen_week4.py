"""Sinh src/track-py/week4.json (Ngày 22–28): mạng nơ-ron và PyTorch.

    python scripts/gen_week4.py
Sau đó: python ../linear-regression-lab/py30/build_notebooks.py và verify_py30.py --days 22-28.
Ngày 22–23 chỉ dùng NumPy; Ngày 24–28 cần PyTorch (requirements-py30.txt). Mọi bài chạy được trên CPU, không cần GPU.
Các số trong phần "Python · Ví dụ tính từng dòng" đã được tính lại bằng code khi soạn."""
import json, sys
from pathlib import Path

BASE = '''import json, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd

DATA = next((p for p in (Path('data'), Path('../data')) if p.is_dir()), Path('data'))

def _check(name, fn):
    try:
        ok, detail = fn()
    except NotImplementedError:
        ok, detail = False, 'Chưa cài đặt (NotImplementedError)'
    except Exception as error:
        ok, detail = False, f'{type(error).__name__}: {error}'
    return {'name': name, 'ok': bool(ok), 'detail': str(detail)}

def _report(ex_id, *checks):
    print('__PY30__' + json.dumps({'id': ex_id, 'checks': list(checks)}, ensure_ascii=False))
'''

TORCH = '''try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as error:
    raise ImportError('Kernel này chưa có PyTorch. Cài bằng Python của Jupyter: python -m pip install -r requirements-py30.txt') from error
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# load_digits (UCI Optical Recognition of Handwritten Digits): 1797 ảnh 8×8, chia 80/20 phân tầng, seed 0.
_Xd, _yd = load_digits(return_X_y=True)
_itr, _iva = train_test_split(np.arange(len(_yd)), test_size=0.2, random_state=0, stratify=_yd)
X_tr = torch.tensor(_Xd[_itr] / 16, dtype=torch.float32); y_tr = torch.tensor(_yd[_itr])
X_va = torch.tensor(_Xd[_iva] / 16, dtype=torch.float32); y_va = torch.tensor(_yd[_iva])
def _loader(X, y, batch_size, seed):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed))
def _train_epoch(model, loader, loss_fn, optimizer):
    model.train(); total, n = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n
def _accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())
'''

def lesson(**k):
    return k

L = []

# ───────────────────────── Ngày 22 ─────────────────────────
L.append(lesson(
 id=22, title='Perceptron và MLP', projectTitle='Fashion-MNIST',
 goals=['Cài quy tắc cập nhật perceptron và thấy nó hội tụ khi dữ liệu tách được tuyến tính', 'Thấy perceptron thất bại trên XOR', 'Viết lan truyền xuôi của MLP hai tầng với ReLU và softmax ổn định số'],
 reads=['Rosenblatt (1958), The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain, Psychological Review 65(6), 386–408', 'ISLR (bản 2), §10.1 Single Layer Neural Networks và §10.2 Multilayer Neural Networks', 'Goodfellow, Bengio & Courville (2016), Deep Learning, Chương 6 Deep Feedforward Networks (§6.1 Example: Learning XOR)'],
 project='Mở notebook cloud Ngay_25_26_28_CNN (Kaggle hoặc Colab) và chạy phần tải Fashion-MNIST. Huấn luyện LogisticRegression trên ảnh làm phẳng để có mốc so sánh cho cả tuần; ghi accuracy và ma trận nhầm lẫn, đánh dấu hai lớp hay bị nhầm nhất.',
 questions=['Vì sao perceptron không học được XOR?', 'Bỏ hàm kích hoạt ReLU giữa hai tầng thì MLP còn mạnh hơn mô hình tuyến tính không?', 'Vì sao trừ max(z) trước khi lấy exp trong softmax?'],
 answers=['XOR không tách được bằng một đường thẳng; perceptron chỉ vẽ được một siêu phẳng nên luôn còn ít nhất một điểm sai.', 'Không: W₂(W₁x + b₁) + b₂ vẫn là một hàm tuyến tính của x.', 'softmax không đổi khi cộng cùng một hằng số vào mọi z; trừ max làm số mũ ≤ 0 nên exp không tràn.'],
 steps=[{'line': 'X = [[2, 1], [−1, −1], [1, −2]]; y = [1, −1, −1]; w, b = 0, 0', 'state': 'bắt đầu từ w = [0, 0], b = 0'},
        {'line': 'mẫu 1: y·(w·x + b) = 0 ≤ 0 → w += y·x; b += y', 'state': 'w = [2, 1], b = 1'},
        {'line': 'mẫu 2: y·(w·x + b) = −1·(−3 + 1) = 2 > 0', 'state': 'đúng, không cập nhật'},
        {'line': 'mẫu 3: y·(w·x + b) = −1·(0 + 1) = −1 ≤ 0 → cập nhật', 'state': 'w = [1, 3], b = 0'},
        {'line': 'vòng 2: cả ba mẫu đều có y·(w·x + b) > 0', 'state': 'không còn lỗi → dừng'},
        {'line': 'Iris setosa / không setosa, petal length và width', 'state': 'hội tụ sau 3 vòng: w = [−0,5, −0,8], b = 2 (khớp sklearn.linear_model.Perceptron)'}],
 setup=BASE + '''from scipy.special import softmax as _sp_softmax
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
_iris = load_iris()
X_ir = _iris.data[:, 2:4]; y_ir = np.where(_iris.target == 0, 1, -1)       # setosa: tách được tuyến tính
X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]); y_xor = np.array([-1, 1, 1, -1])
_sk22 = Perceptron(shuffle=False, eta0=1.0, alpha=0, penalty=None, max_iter=100, tol=None).fit(X_ir, y_ir)
_r22 = np.random.default_rng(22)
_p22 = {'W1': _r22.normal(size=(4, 5)), 'b1': _r22.normal(size=5), 'W2': _r22.normal(size=(5, 3)), 'b2': _r22.normal(size=3)}
_X22 = _r22.normal(size=(7, 4))
''',
 exercises=[dict(
  id='d22-perceptron', title='Quy tắc học của perceptron',
  hint='Duyệt mẫu theo thứ tự. Nếu y·(w·x + b) ≤ 0 thì w ← w + y·x, b ← b + y. Đếm số lần cập nhật trong mỗi vòng; vòng nào bằng 0 thì dừng.',
  starter='''def perceptron_train(X, y, epochs=100):
    # y ∈ {−1, +1}. Trả (w, b, mistakes) với mistakes là list số lần cập nhật ở từng vòng.
    raise NotImplementedError

def perceptron_predict(w, b, X):
    # Trả mảng ±1 (score = 0 tính là +1).
    raise NotImplementedError
''',
  solution='''def perceptron_train(X, y, epochs=100):
    w, b, mistakes = np.zeros(X.shape[1]), 0.0, []
    for _ in range(epochs):
        count = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w, b, count = w + yi * xi, b + yi, count + 1
        mistakes.append(count)
        if count == 0:
            break
    return w, b, mistakes

def perceptron_predict(w, b, X):
    return np.where(X @ w + b >= 0, 1, -1)
''',
  test='''_report('d22-perceptron',
    _check('Ví dụ tính tay: w = [1, 3], b = 0 sau 2 vòng', lambda: ((lambda r: np.allclose(r[0], [1, 3]) and r[1] == 0 and list(r[2]) == [2, 0])(perceptron_train(np.array([[2., 1.], [-1., -1.], [1., -2.]]), np.array([1, -1, -1]))), 'Cập nhật khi y·(w·x + b) ≤ 0.')),
    _check('Iris setosa: hội tụ, khớp sklearn Perceptron', lambda: ((lambda r: r[2][-1] == 0 and np.allclose(r[0], _sk22.coef_[0]) and np.isclose(r[1], _sk22.intercept_[0]) and np.all(perceptron_predict(r[0], r[1], X_ir) == y_ir))(perceptron_train(X_ir, y_ir)), 'shuffle=False, eta0=1.')),
    _check('XOR: không bao giờ hết lỗi', lambda: (min(perceptron_train(X_xor, y_xor, 50)[2]) > 0, 'Không có đường thẳng nào tách được XOR.')),
)
'''), dict(
  id='d22-mlp', title='Lan truyền xuôi của MLP hai tầng',
  hint='softmax: trừ max theo hàng trước khi exp. Lan truyền xuôi: Z1 = X @ W1 + b1, H = ReLU(Z1), Z2 = H @ W2 + b2, P = softmax(Z2). Cross-entropy: −mean(log P[i, y_i]) có chặn p ≥ eps.',
  starter='''def relu(z):
    raise NotImplementedError

def softmax(Z):
    # Theo từng hàng, không tràn số khi Z rất lớn.
    raise NotImplementedError

def mlp_forward(X, params):
    # params: W1 (d, h), b1 (h,), W2 (h, C), b2 (C,). Trả (P, cache) với cache = {'X', 'Z1', 'H'}.
    raise NotImplementedError

def cross_entropy(P, y, eps=1e-12):
    raise NotImplementedError
''',
  solution='''def relu(z):
    return np.maximum(0, z)

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def mlp_forward(X, params):
    Z1 = X @ params['W1'] + params['b1']
    H = relu(Z1)
    P = softmax(H @ params['W2'] + params['b2'])
    return P, {'X': X, 'Z1': Z1, 'H': H}

def cross_entropy(P, y, eps=1e-12):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1))))
''',
  test='''_xor_params = {'W1': np.array([[1., 1.], [1., 1.]]), 'b1': np.array([0., -1.]), 'W2': np.array([[0., 10.], [0., -20.]]), 'b2': np.array([0., -5.])}
_report('d22-mlp',
    _check('softmax ổn định và mỗi hàng cộng bằng 1', lambda: ((lambda P: np.all(np.isfinite(P)) and np.allclose(P, _sp_softmax(np.array([[1000., 1001., 999.], [1., 2., 3.]]), axis=1)))(softmax(np.array([[1000., 1001., 999.], [1., 2., 3.]]))), 'Z = 1000 không được sinh NaN.')),
    _check('Lan truyền xuôi khớp cách tính tham chiếu', lambda: ((lambda P: np.allclose(P, _sp_softmax(np.maximum(0, _X22 @ _p22['W1'] + _p22['b1']) @ _p22['W2'] + _p22['b2'], axis=1)))(mlp_forward(_X22, _p22)[0]), '4 → 5 → 3, ReLU.')),
    _check('Với trọng số cho sẵn, MLP giải được XOR', lambda: (np.array_equal(mlp_forward(X_xor, _xor_params)[0].argmax(1), [0, 1, 1, 0]), 'h₁ = ReLU(x₁ + x₂), h₂ = ReLU(x₁ + x₂ − 1).')),
    _check('Cross-entropy đúng và hữu hạn khi p = 0', lambda: (np.isclose(cross_entropy(np.array([[.7, .3], [.2, .8]]), np.array([0, 1])), -np.mean(np.log([.7, .8]))) and np.isfinite(cross_entropy(np.array([[1., 0.]]), np.array([1]))), 'Chặn xác suất trước khi lấy log.')),
)
''')]))

# ───────────────────────── Ngày 23 ─────────────────────────
L.append(lesson(
 id=23, title='Backpropagation và autograd', projectTitle='Fashion-MNIST',
 goals=['Suy ra gradient của MLP hai tầng bằng quy tắc dây chuyền và kiểm tra bằng sai phân', 'Huấn luyện MLP bằng gradient descent trên dữ liệu không tách được tuyến tính', 'Tự viết một autograd vô hướng nhỏ để hiểu loss.backward() làm gì'],
 reads=['Rumelhart, Hinton & Williams (1986), Learning Representations by Back-Propagating Errors, Nature 323, 533–536', 'Goodfellow, Bengio & Courville (2016), Deep Learning, §6.5 Back-Propagation and Other Differentiation Algorithms', 'Ng & Ma, CS229 Lecture Notes (2023), Chương 7 Deep learning (phần backpropagation)', 'Karpathy (2020), micrograd, kho mã GitHub karpathy/micrograd'],
 project='Trong notebook cloud, thay LogisticRegression bằng MLP NumPy của bạn (784 → 128 → 10) trên 10.000 ảnh Fashion-MNIST đầu tiên. So accuracy với mốc ngày 22 và vẽ loss theo bước. Ghi lại learning rate bạn thử và chuyện gì xảy ra khi nó quá lớn.',
 questions=['Vì sao gradient của softmax + cross-entropy theo logit lại gọn là P − onehot(y)?', 'Đạo hàm của ReLU tại 0 lấy bằng bao nhiêu và có quan trọng không?', 'Vì sao một nút dùng hai lần trong đồ thị phải cộng dồn gradient?'],
 answers=['∂(−log softmax_y)/∂z_j = softmax_j − 1[j = y]: phần log và phần chuẩn hóa triệt tiêu nhau.', 'Quy ước 0 (như PyTorch); xác suất rơi đúng vào 0 rất nhỏ nên hầu như không ảnh hưởng.', 'Theo quy tắc dây chuyền nhiều biến, đạo hàm toàn phần là tổng đóng góp qua mọi đường đi từ nút đó tới loss.'],
 steps=[{'line': 'x = [1, 2]; Z1 = x @ W1', 'state': 'W1 = [[0,5, −1], [0,25, 0,5]] → Z1 = [1, 0]'},
        {'line': 'H = relu(Z1); Z2 = H @ W2', 'state': 'H = [1, 0]; W2 = [[1, −1], [0,5, 1]] → Z2 = [1, −1]'},
        {'line': 'P = softmax(Z2); loss = −log P[y], y = 0', 'state': 'P = [0,881, 0,119], loss = 0,127'},
        {'line': 'D2 = P − onehot(y); dW2 = H.T @ D2', 'state': 'D2 = [−0,119, 0,119]; dW2 = [[−0,119, 0,119], [0, 0]]'},
        {'line': 'DH = D2 @ W2.T; D1 = DH * (Z1 > 0)', 'state': 'DH = [−0,238, 0,060] → D1 = [−0,238, 0] (Z1₂ = 0 nên ReLU chặn)'},
        {'line': 'dW1 = x.T @ D1', 'state': 'dW1 = [[−0,238, 0], [−0,477, 0]]'}],
 setup=BASE + '''from sklearn.datasets import make_moons
# make_moons: dữ liệu tổng hợp hai nửa vầng trăng (giả lập), không tách được bằng đường thẳng.
X_moon, y_moon = make_moons(300, noise=0.2, random_state=0)
def _relu(z): return np.maximum(0, z)
def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
def _forward(X, p):
    Z1 = X @ p['W1'] + p['b1']; H = _relu(Z1); return _softmax(H @ p['W2'] + p['b2']), {'X': X, 'Z1': Z1, 'H': H}
def _loss(X, y, p):
    P, _ = _forward(X, p); return float(-np.mean(np.log(P[np.arange(len(y)), y])))
_r23 = np.random.default_rng(23)
_p23 = {'W1': _r23.normal(size=(3, 6)), 'b1': _r23.normal(size=6), 'W2': _r23.normal(size=(6, 4)), 'b2': _r23.normal(size=4)}
_X23 = _r23.normal(size=(9, 3)); _y23 = _r23.integers(0, 4, 9)
def _num_grads(X, y, p, eps=1e-6):
    out = {}
    for k in p:
        g = np.zeros_like(p[k])
        for idx in np.ndindex(p[k].shape):
            a = {q: v.copy() for q, v in p.items()}; b = {q: v.copy() for q, v in p.items()}
            a[k][idx] += eps; b[k][idx] -= eps
            g[idx] = (_loss(X, y, a) - _loss(X, y, b)) / (2 * eps)
        out[k] = g
    return out
''',
 exercises=[dict(
  id='d23-backprop', title='Lan truyền ngược và huấn luyện MLP',
  hint='D2 = (P − onehot(y)) / n. dW2 = Hᵀ D2, db2 = tổng D2 theo hàng. DH = D2 W2ᵀ, D1 = DH ⊙ [Z1 > 0]. dW1 = Xᵀ D1, db1 = tổng D1. train_mlp: khởi tạo W ~ N(0, √(2/fan_in)) bằng np.random.default_rng(seed), b = 0, rồi lặp gradient descent toàn bộ dữ liệu.',
  starter='''def mlp_backward(y, params, P, cache):
    # cache = {'X', 'Z1', 'H'} từ lan truyền xuôi; P là xác suất softmax. Trả dict gradient cùng khóa với params.
    raise NotImplementedError

def train_mlp(X, y, hidden, lr, steps, seed):
    # Trả (params, loss_history). Có thể dùng _forward(X, params) đã có sẵn.
    raise NotImplementedError
''',
  solution='''def mlp_backward(y, params, P, cache):
    n = len(y)
    D2 = P.copy()
    D2[np.arange(n), y] -= 1
    D2 /= n
    DH = D2 @ params['W2'].T
    D1 = DH * (cache['Z1'] > 0)
    return {'W1': cache['X'].T @ D1, 'b1': D1.sum(axis=0), 'W2': cache['H'].T @ D2, 'b2': D2.sum(axis=0)}

def train_mlp(X, y, hidden, lr, steps, seed):
    rng = np.random.default_rng(seed)
    d, C = X.shape[1], int(y.max()) + 1
    params = {'W1': rng.normal(0, np.sqrt(2 / d), (d, hidden)), 'b1': np.zeros(hidden),
              'W2': rng.normal(0, np.sqrt(2 / hidden), (hidden, C)), 'b2': np.zeros(C)}
    history = []
    for _ in range(steps):
        P, cache = _forward(X, params)
        history.append(float(-np.mean(np.log(P[np.arange(len(y)), y]))))
        grads = mlp_backward(y, params, P, cache)
        params = {k: params[k] - lr * grads[k] for k in params}
    return params, history
''',
  test='''_report('d23-backprop',
    _check('Gradient khớp sai phân trung tâm (mọi tham số)', lambda: ((lambda g, n: all(np.allclose(g[k], n[k], atol=1e-5) for k in n))(mlp_backward(_y23, _p23, *_forward(_X23, _p23)), _num_grads(_X23, _y23, _p23)), 'MLP 3 → 6 → 4 với tham số ngẫu nhiên.')),
    _check('Loss giảm và accuracy trên make_moons ≥ 0,95', lambda: ((lambda r: r[1][-1] < .5 * r[1][0] and (_forward(X_moon, r[0])[0].argmax(1) == y_moon).mean() >= .95)(train_mlp(X_moon, y_moon, 16, 0.5, 1500, 0)), 'Hồi quy logistic chỉ đạt 0,84 trên tập này.')),
)
'''), dict(
  id='d23-autograd', title='Autograd vô hướng tự viết',
  hint='Mỗi Value giữ data, grad, các nút cha và một hàm _backward. backward(): sắp xếp tô-pô các nút, đặt grad của nút cuối = 1, rồi gọi _backward theo thứ tự ngược. Luôn CỘNG DỒN vào grad (+=), không gán.',
  starter='''class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    __radd__ = lambda self, other: self + other
    __rmul__ = lambda self, other: self * other
''',
  solution='''class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(node):
            if id(node) not in seen:
                seen.add(id(node))
                for parent in node._parents:
                    visit(parent)
                order.append(node)
        visit(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    __radd__ = lambda self, other: self + other
    __rmul__ = lambda self, other: self * other
''',
  test='''def _ag1():
    a, b, c = Value(0.5), Value(-1.5), Value(0.3)
    f = (a * b + c).tanh(); f.backward()
    s = 1 - math.tanh(0.5 * -1.5 + 0.3) ** 2
    return np.allclose([a.grad, b.grad, c.grad], [s * -1.5, s * 0.5, s])
def _ag2():
    a = Value(3.0); f = a * a + a; f.backward()
    return np.isclose(f.data, 12) and np.isclose(a.grad, 7)
def _ag3():
    x = Value(2.0); f = 3 * x + 1; f.backward()
    return np.isclose(f.data, 7) and np.isclose(x.grad, 3)
_report('d23-autograd',
    _check('tanh(a·b + c): gradient khớp công thức', lambda: (_ag1(), 'd tanh(u)/du = 1 − tanh²(u).')),
    _check('Nút dùng hai lần cộng dồn gradient', lambda: (_ag2(), 'f = a² + a tại a = 3: df/da = 7.')),
    _check('Phép tính với số thường ở bên trái', lambda: (_ag3(), 'f = 3·x + 1.')),
)
''')]))

# ───────────────────────── Ngày 24 ─────────────────────────
L.append(lesson(
 id=24, title='Tensor và vòng huấn luyện PyTorch', projectTitle='Fashion-MNIST',
 goals=['Định nghĩa mô hình bằng nn.Module', 'Viết vòng huấn luyện chuẩn: zero_grad → forward → loss → backward → step', 'Đánh giá đúng cách với model.eval() và torch.no_grad()'],
 reads=['Paszke et al. (2019), PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 32', 'Tài liệu PyTorch, Learn the Basics (Tensors, Datasets & DataLoaders, Build the Neural Network, Automatic Differentiation, Optimization Loop)', 'ISLR (bản 2), §10.7 Fitting a Neural Network'],
 project='Trong notebook cloud, chuyển MLP ngày 23 sang PyTorch (784 → 256 → 10), huấn luyện 5 epoch trên toàn bộ Fashion-MNIST với batch 128 trên GPU nếu có. Ghi thời gian mỗi epoch trên CPU và GPU, loss train và accuracy test.',
 questions=['Quên optimizer.zero_grad() thì chuyện gì xảy ra?', 'model.eval() và torch.no_grad() khác nhau thế nào?', 'Vì sao trộn dữ liệu mỗi epoch?'],
 answers=['PyTorch cộng dồn gradient vào .grad, nên mỗi bước dùng tổng gradient của mọi batch trước đó: bước cập nhật sai và lớn dần.', 'eval() đổi hành vi của lớp như Dropout, BatchNorm; no_grad() tắt việc ghi đồ thị để tiết kiệm bộ nhớ. Khi đánh giá thường cần cả hai.', 'Để các batch không lặp cùng một thứ tự, gradient ngẫu nhiên ít tương quan giữa các epoch.'],
 steps=[{'line': 'X_tr, y_tr = tensor(digits[train] / 16), tensor(labels)', 'state': 'X_tr.shape = (1437, 64), X_va.shape = (360, 64)'},
        {'line': 'model = MLP(64, 64, 10)', 'state': '64·64 + 64 + 64·10 + 10 = 4810 tham số'},
        {'line': 'loader = DataLoader(..., batch_size=64, shuffle=True)', 'state': '23 batch mỗi epoch (batch cuối 29 mẫu)'},
        {'line': 'optimizer.zero_grad(); loss = loss_fn(model(xb), yb)', 'state': 'xóa gradient cũ rồi tính loss của batch'},
        {'line': 'loss.backward(); optimizer.step()', 'state': 'autograd điền .grad cho mọi tham số, Adam cập nhật'},
        {'line': 'evaluate(model, X_va, y_va)', 'state': 'seed 0, Adam lr = 0,01, 15 epoch: accuracy validation 0,969'}],
 setup=BASE + TORCH + '''import copy
class _Drop(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 10))
    def forward(self, x): return self.net(x)
''',
 exercises=[dict(
  id='d24-loop', title='nn.Module, DataLoader và vòng huấn luyện',
  hint='MLP: nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out)). make_loader: TensorDataset + DataLoader(shuffle=True, generator=torch.Generator().manual_seed(seed)). evaluate: model.eval() rồi with torch.no_grad().',
  starter='''class MLP(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def make_loader(X, y, batch_size, seed):
    raise NotImplementedError

def train_epoch(model, loader, loss_fn, optimizer):
    # Một epoch; trả loss trung bình theo mẫu.
    raise NotImplementedError

def evaluate(model, X, y):
    # Accuracy (float).
    raise NotImplementedError
''',
  solution='''class MLP(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)

def make_loader(X, y, batch_size, seed):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed))

def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        count += len(yb)
    return total / count

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(dim=1) == y).float().mean())
''',
  test='''def _loader_ok():
    L = make_loader(X_tr, y_tr, 100, 3)
    a = torch.cat([yb for _, yb in L]); b = torch.cat([yb for _, yb in L])
    c = torch.cat([yb for _, yb in make_loader(X_tr, y_tr, 100, 3)])
    sizes = [len(yb) for _, yb in make_loader(X_tr, y_tr, 100, 3)]
    return (sorted(a.tolist()) == sorted(y_tr.tolist()) and not torch.equal(a, b) and torch.equal(a, c)
            and sizes == [100] * 14 + [37])
def _same_step():
    torch.manual_seed(1); m1 = MLP(64, 16, 10); m2 = copy.deepcopy(m1)
    o1, o2 = torch.optim.SGD(m1.parameters(), 0.1), torch.optim.SGD(m2.parameters(), 0.1)
    l1 = train_epoch(m1, make_loader(X_tr, y_tr, 64, 5), nn.CrossEntropyLoss(), o1)
    l2 = _train_epoch(m2, _loader(X_tr, y_tr, 64, 5), nn.CrossEntropyLoss(), o2)
    return np.isclose(l1, l2) and all(torch.allclose(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
def _trained():
    torch.manual_seed(0); m = MLP(64, 64, 10); opt = torch.optim.Adam(m.parameters(), 1e-2); L = make_loader(X_tr, y_tr, 64, 0)
    for _ in range(15): train_epoch(m, L, nn.CrossEntropyLoss(), opt)
    return evaluate(m, X_va, y_va)
def _eval_mode():
    torch.manual_seed(0); m = _Drop()
    a, b = evaluate(m, X_va, y_va), evaluate(m, X_va, y_va)
    return a == b and np.isclose(a, _accuracy(m, X_va, y_va))
_report('d24-loop',
    _check('MLP có đúng số tham số và đầu ra (n, 10)', lambda: ((lambda m: sum(p.numel() for p in m.parameters()) == 4810 and tuple(m(X_va[:5]).shape) == (5, 10))(MLP(64, 64, 10)), '64·64 + 64 + 64·10 + 10 = 4810.')),
    _check('DataLoader trộn theo seed, đủ mẫu, đúng cỡ batch', lambda: (_loader_ok(), '1437 mẫu, batch 100 → 14 batch đủ + 1 batch 37.')),
    _check('Một epoch khớp vòng huấn luyện tham chiếu', lambda: (_same_step(), 'Cùng khởi tạo, cùng thứ tự batch, SGD lr = 0,1.')),
    _check('evaluate dùng model.eval() (Dropout tắt)', lambda: (_eval_mode(), 'Gọi hai lần phải ra cùng kết quả.')),
    _check('Sau 15 epoch Adam: accuracy validation ≥ 0,95', lambda: ((lambda a: (a >= .95, f'Đạt {a:.3f}.'))(_trained())[0], 'Seed 0, lr = 0,01, batch 64.')),
)
''')]))

# ───────────────────────── Ngày 25 ─────────────────────────
L.append(lesson(
 id=25, title='Tối ưu và regularization', projectTitle='Fashion-MNIST',
 goals=['Cài SGD có momentum và Adam đúng như PyTorch', 'Viết dropout đảo (inverted dropout) và hiểu vì sao tắt nó khi đánh giá', 'Dừng sớm theo validation với tham số patience'],
 reads=['Goodfellow, Bengio & Courville (2016), Deep Learning, §8.3.2 Momentum và §8.5.3 Adam; §7.8 Early Stopping; §7.12 Dropout', 'Kingma & Ba (2015), Adam: A Method for Stochastic Optimization, ICLR', 'Srivastava et al. (2014), Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR 15, 1929–1958'],
 project='Trong notebook cloud, huấn luyện cùng một MLP trên Fashion-MNIST với SGD, SGD + momentum 0,9 và Adam (cùng số epoch). Vẽ loss validation của ba cách trên một hình. Thêm Dropout(0,3) và dừng sớm với patience = 3, ghi epoch dừng và accuracy test.',
 questions=['Momentum giúp gì trên mặt loss hẹp và dài?', 'Vì sao Adam cần hiệu chỉnh độ lệch (bias correction) ở các bước đầu?', 'Vì sao dropout chia cho (1 − p) khi huấn luyện?'],
 answers=['Nó cộng dồn các bước cùng hướng và triệt tiêu dao động qua lại giữa hai vách, nên tiến nhanh dọc thung lũng.', 'm và v khởi đầu bằng 0 nên các bước đầu bị kéo về 0; chia cho 1 − βᵗ khử độ lệch đó.', 'Để kỳ vọng của mỗi nơ-ron khi huấn luyện bằng giá trị khi đánh giá (lúc không bỏ nơ-ron nào), nên không phải sửa gì lúc suy luận.'],
 steps=[{'line': 'f(w) = ½(10·w₁² + w₂²); g = [10·w₁, w₂]; w = [1, 1]', 'state': 'thung lũng hẹp theo w₁, thoải theo w₂'},
        {'line': 'SGD lr = 0,05: w = w − lr·g (3 bước)', 'state': 'w = [0,5; 0,95] → [0,25; 0,903] → [0,125; 0,857]'},
        {'line': 'v = 0,9·v + g; w = w − lr·v', 'state': 'momentum: w = [0,5; 0,95] → [−0,2; 0,858] → [−0,73; 0,731]'},
        {'line': 'm = β₁m + (1−β₁)g; v = β₂v + (1−β₂)g²', 'state': 'Adam bước 1: m̂ = [10, 1], v̂ = [100, 1]'},
        {'line': 'w = w − lr·m̂ / (√v̂ + ε)', 'state': 'lr = 0,1: w = [0,9; 0,9] → [0,80; 0,80] → [0,70; 0,70] (cùng bước cho mọi tọa độ)'},
        {'line': 'early_stopping(val_losses, patience=3)', 'state': 'dừng khi 3 epoch liên tiếp không cải thiện, giữ trọng số của epoch tốt nhất'}],
 setup=BASE + TORCH + '''_A25 = np.array([10., 1., 3.])
def _torch_run(make_opt, steps):
    w = torch.tensor([1., 1., -2.], requires_grad=True); opt = make_opt([w]); out = []
    for _ in range(steps):
        opt.zero_grad(); (0.5 * (torch.tensor(_A25, dtype=torch.float32) * w ** 2).sum()).backward(); opt.step(); out.append(w.detach().numpy().copy())
    return np.array(out)
''',
 exercises=[dict(
  id='d25-optim', title='SGD có momentum và Adam',
  hint='Quy ước PyTorch: v ← momentum·v + g; w ← w − lr·v. Adam: m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g², m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ), w ← w − lr·m̂/(√v̂ + ε). t bắt đầu từ 1.',
  starter='''def sgd_momentum_step(w, g, v, lr, momentum):
    # Trả (w_mới, v_mới). Không sửa mảng đầu vào tại chỗ.
    raise NotImplementedError

def adam_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    # Trả (w_mới, m_mới, v_mới).
    raise NotImplementedError
''',
  solution='''def sgd_momentum_step(w, g, v, lr, momentum):
    v = momentum * v + g
    return w - lr * v, v

def adam_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), m, v
''',
  test='''def _mine(kind, steps):
    w = np.array([1., 1., -2.]); s1 = np.zeros(3); s2 = np.zeros(3); out = []
    for t in range(1, steps + 1):
        g = _A25 * w
        if kind == 'sgd':
            w, s1 = sgd_momentum_step(w, g, s1, 0.05, 0.9)
        else:
            w, s1, s2 = adam_step(w, g, s1, s2, t, 0.1)
        out.append(w.copy())
    return np.array(out)
def _no_inplace():
    w, g, v = np.ones(3), np.ones(3), np.zeros(3); sgd_momentum_step(w, g, v, .1, .9)
    return np.all(w == 1) and np.all(v == 0)
_report('d25-optim',
    _check('SGD momentum khớp torch.optim.SGD qua 20 bước', lambda: (np.allclose(_mine('sgd', 20), _torch_run(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9), 20), atol=1e-5), 'lr = 0,05, momentum = 0,9.')),
    _check('Adam khớp torch.optim.Adam qua 20 bước', lambda: (np.allclose(_mine('adam', 20), _torch_run(lambda p: torch.optim.Adam(p, lr=0.1), 20), atol=1e-5), 'lr = 0,1, β = (0,9; 0,999).')),
    _check('Không sửa mảng đầu vào tại chỗ', lambda: (_no_inplace(), 'Trả mảng mới.')),
)
'''), dict(
  id='d25-reg', title='Dropout đảo và dừng sớm',
  hint='dropout: khi training, mask = rng.random(x.shape) >= p, trả x·mask/(1 − p); khi đánh giá trả x. early_stopping: theo dõi loss nhỏ nhất; nếu không giảm trong patience epoch liên tiếp thì dừng ở epoch đó.',
  starter='''def dropout(x, p, training, rng):
    raise NotImplementedError

def early_stopping(val_losses, patience):
    # Trả (best_epoch, stop_epoch), đếm từ 0. Chạy hết danh sách mà không dừng thì stop_epoch = len − 1.
    raise NotImplementedError
''',
  solution='''def dropout(x, p, training, rng):
    if not training or p == 0:
        return x
    mask = rng.random(x.shape) >= p
    return x * mask / (1 - p)

def early_stopping(val_losses, patience):
    best, best_epoch, wait = float('inf'), 0, 0
    for epoch, loss in enumerate(val_losses):
        if loss < best:
            best, best_epoch, wait = loss, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                return best_epoch, epoch
    return best_epoch, len(val_losses) - 1
''',
  test='''_losses25 = [1.0, .8, .7, .72, .69, .71, .73, .74, .6]
_report('d25-reg',
    _check('Giữ nguyên kỳ vọng và tỉ lệ bỏ đúng p', lambda: ((lambda z: abs(z.mean() - 1) < .02 and abs((z == 0).mean() - .3) < .01)(dropout(np.ones(200000), .3, True, np.random.default_rng(0))), 'p = 0,3 trên 200.000 phần tử.')),
    _check('Khi đánh giá trả nguyên x', lambda: (np.array_equal(dropout(np.arange(5.), .5, False, np.random.default_rng(0)), np.arange(5.)), 'training = False.')),
    _check('Dừng sớm đúng epoch', lambda: (early_stopping(_losses25, 3) == (4, 7) and early_stopping(_losses25, 10) == (8, 8), 'patience = 3: tốt nhất epoch 4, dừng ở epoch 7 (bỏ lỡ 0,6 ở epoch 8).')),
)
''')]))

# ───────────────────────── Ngày 26 ─────────────────────────
L.append(lesson(
 id=26, title='Tích chập và CNN', projectTitle='Fashion-MNIST',
 goals=['Tính kích thước đầu ra của tích chập theo kernel, stride, padding', 'Cài conv2d và max pooling bằng vòng lặp rồi so với PyTorch', 'Dựng và huấn luyện một CNN nhỏ trên ảnh chữ số'],
 reads=['Goodfellow, Bengio & Courville (2016), Deep Learning, Chương 9 Convolutional Networks (§9.1–9.3)', 'ISLR (bản 2), §10.3 Convolutional Neural Networks', 'LeCun, Bottou, Bengio & Haffner (1998), Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE 86(11), 2278–2324', 'Dumoulin & Visin (2016), A Guide to Convolution Arithmetic for Deep Learning, arXiv:1603.07285'],
 project='Chạy notebook cloud Ngay_25_26_28_CNN trên GPU: huấn luyện CNN trên Fashion-MNIST, so accuracy với MLP ngày 24 và logistic ngày 22. In số tham số của hai mô hình. Sau đó chạy phần Digit Recognizer để xuất submission và so với kết quả PCA + KNN của ngày 21.',
 questions=['Một lớp Conv2d(1, 8, 3) có bao nhiêu tham số và vì sao không phụ thuộc kích thước ảnh?', 'padding = 1 với kernel 3×3 giữ nguyên kích thước ảnh: vì sao?', 'Max pooling cho mô hình tính chất gì?'],
 answers=['8·(1·3·3) + 8 = 80; cùng một bộ lọc được dùng lại ở mọi vị trí (chia sẻ trọng số).', '⌊(n + 2·1 − 3)/1⌋ + 1 = n.', 'Bất biến (gần đúng) với dịch chuyển nhỏ trong vùng pooling và giảm kích thước không gian.'],
 steps=[{'line': 'x.shape = (N, 1, 8, 8)', 'state': 'ảnh load_digits'},
        {'line': 'Conv2d(1, 8, 3, padding=1) → ReLU → MaxPool2d(2)', 'state': '(N, 8, 8, 8) → (N, 8, 4, 4); 80 tham số'},
        {'line': 'Conv2d(8, 16, 3, padding=1) → ReLU → MaxPool2d(2)', 'state': '(N, 16, 4, 4) → (N, 16, 2, 2); 1168 tham số'},
        {'line': 'Flatten → Linear(64, 10)', 'state': '650 tham số; tổng 1898'},
        {'line': 'out = ⌊(n + 2p − k)/s⌋ + 1', 'state': 'n = 8, k = 3, p = 1, s = 1 → 8; s = 2 → 4'},
        {'line': 'huấn luyện 15 epoch, Adam lr = 0,01, seed 0', 'state': 'accuracy validation 0,972 (MLP 4810 tham số: 0,969)'}],
 setup=BASE + TORCH,
 exercises=[dict(
  id='d26-conv', title='conv2d, max pooling và kích thước đầu ra',
  hint='Đệm bằng F.pad(x, (p, p, p, p)). Với mỗi vị trí đầu ra (i, j): lấy vùng x[:, :, i·s : i·s + k, j·s : j·s + k], nhân với trọng số rồi cộng theo (C_in, k, k). Có thể dùng torch.einsum("nchw,ochw->no", vùng, w).',
  starter='''def conv_output_size(n, k, stride=1, padding=0):
    raise NotImplementedError

def conv2d_naive(x, w, b, stride=1, padding=0):
    # x: (N, C_in, H, W); w: (C_out, C_in, k, k); b: (C_out,). Dùng vòng lặp trên vị trí đầu ra.
    raise NotImplementedError

def max_pool2d_naive(x, k):
    # Cửa sổ k×k, bước k, không đệm.
    raise NotImplementedError
''',
  solution='''def conv_output_size(n, k, stride=1, padding=0):
    return (n + 2 * padding - k) // stride + 1

def conv2d_naive(x, w, b, stride=1, padding=0):
    x = F.pad(x, (padding, padding, padding, padding))
    N, _, H, W = x.shape
    C_out, _, k, _ = w.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    out = torch.zeros(N, C_out, Ho, Wo)
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, :, i, j] = torch.einsum('nchw,ochw->no', patch, w) + b
    return out

def max_pool2d_naive(x, k):
    N, C, H, W = x.shape
    out = torch.zeros(N, C, H // k, W // k)
    for i in range(H // k):
        for j in range(W // k):
            out[:, :, i, j] = x[:, :, i * k:(i + 1) * k, j * k:(j + 1) * k].amax(dim=(2, 3))
    return out
''',
  test='''torch.manual_seed(26)
_x26, _w26, _b26 = torch.randn(2, 3, 9, 7), torch.randn(4, 3, 3, 3), torch.randn(4)
_report('d26-conv',
    _check('Công thức kích thước đầu ra', lambda: (conv_output_size(8, 3, 1, 1) == 8 and conv_output_size(28, 5, 1, 0) == 24 and conv_output_size(9, 3, 2, 1) == 5, '⌊(n + 2p − k)/s⌋ + 1.')),
    _check('conv2d khớp F.conv2d (stride 1, padding 0)', lambda: (torch.allclose(conv2d_naive(_x26, _w26, _b26), F.conv2d(_x26, _w26, _b26), atol=1e-5), 'Ảnh 9×7, 3 kênh vào, 4 kênh ra.')),
    _check('conv2d khớp F.conv2d (stride 2, padding 1)', lambda: (torch.allclose(conv2d_naive(_x26, _w26, _b26, 2, 1), F.conv2d(_x26, _w26, _b26, stride=2, padding=1), atol=1e-5), 'Ảnh không vuông.')),
    _check('max pooling khớp F.max_pool2d', lambda: (torch.allclose(max_pool2d_naive(_x26[:, :, :8, :6], 2), F.max_pool2d(_x26[:, :, :8, :6], 2)), 'Cửa sổ 2×2.')),
)
'''), dict(
  id='d26-cnn', title='CNN nhỏ cho ảnh chữ số 8×8',
  hint='Conv2d(1, 8, 3, padding=1) → ReLU → MaxPool2d(2) → Conv2d(8, 16, 3, padding=1) → ReLU → MaxPool2d(2) → Flatten → Linear(16·2·2, 10). Ảnh đầu vào: X.view(-1, 1, 8, 8). Dùng _train_epoch và _loader có sẵn.',
  starter='''class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        # x: (N, 1, 8, 8) → logits (N, 10)
        raise NotImplementedError

def fit_cnn(seed, epochs=15):
    # torch.manual_seed(seed), Adam lr = 0.01, batch 64 với _loader(..., seed). Trả (model, accuracy validation).
    raise NotImplementedError
''',
  solution='''class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))

def fit_cnn(seed, epochs=15):
    torch.manual_seed(seed)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = _loader(X_tr.view(-1, 1, 8, 8), y_tr, 64, seed)
    for _ in range(epochs):
        _train_epoch(model, loader, nn.CrossEntropyLoss(), optimizer)
    return model, _accuracy(model, X_va.view(-1, 1, 8, 8), y_va)
''',
  test='''_c26 = {}
def _fit26():
    if 'r' not in _c26:
        _c26['r'] = fit_cnn(0)
    return _c26['r']
_report('d26-cnn',
    _check('Đầu ra (N, 10) và đúng 1898 tham số', lambda: ((lambda m: tuple(m(torch.zeros(3, 1, 8, 8)).shape) == (3, 10) and sum(p.numel() for p in m.parameters()) == 1898)(SmallCNN()), '80 + 1168 + 650.')),
    _check('Có tích chập và pooling', lambda: ((lambda m: sum(isinstance(x, nn.Conv2d) for x in m.modules()) == 2 and sum(isinstance(x, nn.MaxPool2d) for x in m.modules()) == 2)(SmallCNN()), 'Hai khối Conv → ReLU → MaxPool.')),
    _check('Accuracy validation ≥ 0,95 sau 15 epoch', lambda: ((lambda r: (r[1] >= .95, f'Đạt {r[1]:.3f}.'))(_fit26())[0], 'Seed 0.')),
)
''')]))

# ───────────────────────── Ngày 27 ─────────────────────────
L.append(lesson(
 id=27, title='Embedding, attention và Transformer', projectTitle='IMDB',
 goals=['Cài scaled dot-product attention và mặt nạ nhân quả (causal mask)', 'Tạo positional encoding hình sin', 'Ghép nhiều đầu (multi-head) từ trọng số của nn.MultiheadAttention'],
 reads=['Vaswani et al. (2017), Attention Is All You Need, NeurIPS 30', 'Zhang, Lipton, Li & Smola, Dive into Deep Learning, chương Attention Mechanisms and Transformers', 'Tài liệu PyTorch, torch.nn.MultiheadAttention và torch.nn.functional.scaled_dot_product_attention'],
 project='Mở notebook cloud Ngay_27_Transformer (Kaggle hoặc Colab) với bộ IMDB 50k review. Huấn luyện mô hình embedding trung bình (bag of embeddings) rồi Transformer encoder nhỏ, so accuracy và thời gian một epoch trên GPU. Ghi độ dài chuỗi tối đa bạn chọn và ảnh hưởng của nó.',
 questions=['Vì sao chia điểm số cho √d?', 'Thiếu positional encoding thì self-attention nhìn câu như thế nào?', 'Mặt nạ nhân quả dùng khi nào?'],
 answers=['Với q, k có các thành phần độc lập phương sai 1, q·k có phương sai d; chia √d giữ softmax không bị bão hòa khi d lớn.', 'Như một túi từ: hoán vị các token thì đầu ra hoán vị theo, thứ tự từ không được dùng.', 'Khi sinh chuỗi từ trái sang phải (decoder, mô hình ngôn ngữ): token thứ t không được nhìn token sau nó.'],
 steps=[{'line': 'Q = K = [[1, 0], [0, 1], [1, 1]]; V = [[1, 0], [0, 1], [0,5, 0,5]]', 'state': '3 token, d = 2'},
        {'line': 'S = Q @ K.T / √2', 'state': 'S = [[0,71, 0, 0,71], [0, 0,71, 0,71], [0,71, 0,71, 1,41]]'},
        {'line': 'A = softmax(S, dim=−1)', 'state': 'hàng 1: [0,401, 0,198, 0,401]; hàng 3: [0,248, 0,248, 0,503]'},
        {'line': 'out = A @ V', 'state': 'out = [[0,602, 0,398], [0,398, 0,602], [0,5, 0,5]]'},
        {'line': 'S[~causal] = −∞; A = softmax(S)', 'state': 'hàng 1: [1, 0, 0]; hàng 2: [0,330, 0,670, 0]'},
        {'line': 'PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)', 'state': 'PE[0] = [0, 1, 0, 1, ...]'}],
 setup=BASE + TORCH + '''torch.manual_seed(27)
_q27, _k27, _v27 = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
_mha27 = nn.MultiheadAttention(embed_dim=12, num_heads=3, batch_first=True)
_x27 = torch.randn(2, 6, 12)
''',
 exercises=[dict(
  id='d27-attention', title='Scaled dot-product attention, mặt nạ và positional encoding',
  hint='scores = q @ k.transpose(−2, −1) / √d; nếu có mask (bool, True = được nhìn) thì scores.masked_fill(~mask, −inf); weights = scores.softmax(−1); out = weights @ v. causal_mask: torch.tril(torch.ones(T, T, dtype=torch.bool)).',
  starter='''def attention(q, k, v, mask=None):
    # Trả (out, weights).
    raise NotImplementedError

def causal_mask(T):
    raise NotImplementedError

def positional_encoding(T, d):
    # Tensor (T, d); d chẵn.
    raise NotImplementedError
''',
  solution='''def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

def causal_mask(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))

def positional_encoding(T, d):
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    angle = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return pe
''',
  test='''_report('d27-attention',
    _check('Khớp F.scaled_dot_product_attention', lambda: (torch.allclose(attention(_q27, _k27, _v27)[0], F.scaled_dot_product_attention(_q27, _k27, _v27), atol=1e-5), 'Không mặt nạ.')),
    _check('Mặt nạ nhân quả: khớp is_causal=True và không nhìn tương lai', lambda: ((lambda r: torch.allclose(r[0], F.scaled_dot_product_attention(_q27, _k27, _v27, is_causal=True), atol=1e-5) and float(r[1].triu(1).abs().sum()) == 0)(attention(_q27, _k27, _v27, causal_mask(5))), 'Trọng số phía trên đường chéo bằng 0.')),
    _check('Positional encoding đúng công thức Vaswani et al.', lambda: ((lambda pe: tuple(pe.shape) == (50, 16) and torch.allclose(pe[0], torch.tensor([0., 1.] * 8)) and np.isclose(float(pe[7, 4]), math.sin(7 / 10000 ** (4 / 16)), atol=1e-6) and np.isclose(float(pe[7, 5]), math.cos(7 / 10000 ** (4 / 16)), atol=1e-6))(positional_encoding(50, 16)), 'Kiểm tra PE[0] và PE[7, 4:6].')),
)
'''), dict(
  id='d27-mha', title='Multi-head attention từ trọng số có sẵn',
  hint='W_q, W_k, W_v = mha.in_proj_weight.chunk(3) (bias tương tự). Chiếu x, tách thành h đầu: (B, T, E) → (B, h, T, E/h). Dùng attention của bài trên cho từng đầu, ghép lại (B, T, E), rồi qua mha.out_proj.',
  starter='''def multi_head_attention(x, mha):
    # x: (B, T, E); mha: nn.MultiheadAttention(batch_first=True). Trả tensor (B, T, E) bằng phép tính thủ công.
    raise NotImplementedError
''',
  solution='''def multi_head_attention(x, mha):
    B, T, E = x.shape
    h = mha.num_heads
    Wq, Wk, Wv = mha.in_proj_weight.chunk(3)
    bq, bk, bv = mha.in_proj_bias.chunk(3)
    split = lambda t: t.view(B, T, h, E // h).transpose(1, 2)
    q, k, v = split(x @ Wq.T + bq), split(x @ Wk.T + bk), split(x @ Wv.T + bv)
    out, _ = attention(q, k, v)
    return mha.out_proj(out.transpose(1, 2).reshape(B, T, E))
''',
  test='''def _ref_attention(q, k, v, mask=None):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return s.softmax(-1) @ v, s.softmax(-1)
attention = globals().get('attention') or _ref_attention
_report('d27-mha',
    _check('Khớp nn.MultiheadAttention (3 đầu, E = 12)', lambda: (torch.allclose(multi_head_attention(_x27, _mha27), _mha27(_x27, _x27, _x27, need_weights=False)[0], atol=1e-5), 'Self-attention, batch_first=True.')),
    _check('Đúng kích thước đầu ra', lambda: (tuple(multi_head_attention(_x27, _mha27).shape) == (2, 6, 12), '(B, T, E).')),
)
''')]))

# ───────────────────────── Ngày 28 ─────────────────────────
L.append(lesson(
 id=28, title='Dự án: CNN với nhiều seed', projectTitle='Fashion-MNIST',
 goals=['Cố định mọi nguồn ngẫu nhiên để một lần chạy lặp lại được', 'Báo cáo kết quả bằng trung bình ± độ lệch chuẩn và khoảng tin cậy qua nhiều seed', 'So hai cấu hình bằng hiệu cặp (paired) trên cùng các seed'],
 reads=['Bouthillier et al. (2021), Accounting for Variance in Machine Learning Benchmarks, Proceedings of MLSys 3', 'Tài liệu PyTorch, Reproducibility (torch.manual_seed, torch.use_deterministic_algorithms)', 'Henderson et al. (2018), Deep Reinforcement Learning that Matters, AAAI (phần về seed)'],
 project='Trong notebook cloud trên GPU, huấn luyện CNN Fashion-MNIST với 5 seed cho hai cấu hình (ví dụ có và không có Dropout). Báo cáo bảng accuracy test từng seed, trung bình ± độ lệch chuẩn, khoảng tin cậy 95% của hiệu cặp, và kết luận cấu hình nào tốt hơn có chắc chắn không. Ghi rõ GPU có thể không tất định tuyệt đối.',
 questions=['Vì sao không chọn "seed tốt nhất" để báo cáo?', 'Vì sao dùng hiệu cặp thay vì so hai trung bình độc lập?', 'Cùng seed trên CPU và GPU có cho cùng kết quả không?'],
 answers=['Đó là chọn giá trị lớn nhất trong các lần chạy có nhiễu, giống chọn mô hình theo tập test: kết quả thiên lạc quan và không lặp lại được.', 'Cùng seed thì cùng khởi tạo và thứ tự batch, nên phần nhiễu chung bị trừ đi; phương sai của hiệu nhỏ hơn nhiều.', 'Không đảm bảo: nhiều phép tính trên GPU (ví dụ cộng dồn song song) không tất định và thư viện khác nhau cho sai khác số học nhỏ.'],
 steps=[{'line': 'seed_everything(s): random, numpy, torch', 'state': 'cùng seed → cùng khởi tạo và cùng thứ tự batch'},
        {'line': 'scores = [fit_cnn(s)[1] for s in range(5)]', 'state': 'load_digits: [0,972, 0,983, 0,981, 0,969, 0,969]'},
        {'line': 'mean, std = scores.mean(), scores.std(ddof=1)', 'state': '0,975 ± 0,0065'},
        {'line': 'half = t₀,₉₇₅(n − 1) · std / √n', 'state': 't₀,₉₇₅(4) = 2,776 → khoảng tin cậy 95%: 0,975 ± 0,008'},
        {'line': 'max(scores)', 'state': '0,983 nếu chỉ báo cáo seed tốt nhất: cao hơn trung bình 0,008'},
        {'line': 'diff = scores_A − scores_B (cùng seed)', 'state': 'kết luận chỉ chắc chắn khi khoảng tin cậy của hiệu không chứa 0'}],
 setup=BASE + TORCH + '''from scipy import stats
class _CNN28(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 10))
    def forward(self, x): return self.f(x)
def train_digits_cnn(epochs=5):
    """Huấn luyện CNN nhỏ trên load_digits KHÔNG tự đặt seed: kết quả phụ thuộc trạng thái ngẫu nhiên hiện tại."""
    m = _CNN28(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    L = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr.view(-1, 1, 8, 8), y_tr), batch_size=64, shuffle=True)
    for _ in range(epochs): _train_epoch(m, L, nn.CrossEntropyLoss(), opt)
    return _accuracy(m, X_va.view(-1, 1, 8, 8), y_va)
''',
 exercises=[dict(
  id='d28-seeds', title='Seed, tổng hợp nhiều lần chạy và so sánh cặp',
  hint='seed_everything: random.seed, np.random.seed, torch.manual_seed. summarize: mean, std với ddof = 1, half = stats.t.ppf(0.975, n − 1)·std/√n. paired_compare: d = a − b rồi summarize(d).',
  starter='''def seed_everything(seed):
    raise NotImplementedError

def run_seeds(train_once, seeds):
    # Gọi seed_everything(s) rồi train_once() cho từng seed; trả np.array điểm.
    raise NotImplementedError

def summarize(scores):
    # Trả dict: mean, std (ddof=1), ci_low, ci_high (khoảng tin cậy 95% theo phân phối t).
    raise NotImplementedError

def paired_compare(a, b):
    # a, b: điểm của hai cấu hình trên cùng danh sách seed. Trả summarize(a − b).
    raise NotImplementedError
''',
  solution='''def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_seeds(train_once, seeds):
    out = []
    for s in seeds:
        seed_everything(s)
        out.append(train_once())
    return np.array(out)

def summarize(scores):
    scores = np.asarray(scores, dtype=float)
    n, mean, std = len(scores), scores.mean(), scores.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
    return {'mean': float(mean), 'std': float(std), 'ci_low': float(mean - half), 'ci_high': float(mean + half)}

def paired_compare(a, b):
    return summarize(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
''',
  test='''_s28 = np.array([.972, .983, .981, .969, .969])
def _det():
    a = run_seeds(lambda: (random.random(), float(np.random.rand()), float(torch.rand(1))), [3, 3, 4])
    return np.allclose(a[0], a[1]) and not np.allclose(a[0], a[2])
def _cnn_det():
    r = run_seeds(lambda: train_digits_cnn(2), [7, 7])
    return r[0] == r[1]
_report('d28-seeds',
    _check('Cùng seed cho cùng số ngẫu nhiên ở random, NumPy và PyTorch', lambda: (_det(), 'Seed 3 hai lần giống nhau, seed 4 khác.')),
    _check('Huấn luyện CNN lặp lại được với cùng seed', lambda: (_cnn_det(), 'train_digits_cnn không tự đặt seed; run_seeds phải làm.')),
    _check('summarize đúng trung bình, độ lệch chuẩn mẫu và khoảng tin cậy t', lambda: ((lambda r: np.isclose(r['mean'], .9748) and np.isclose(r['std'], _s28.std(ddof=1)) and np.isclose(r['ci_high'] - r['mean'], stats.t.ppf(.975, 4) * _s28.std(ddof=1) / math.sqrt(5)))(summarize(_s28)), 'Năm điểm accuracy của CNN.')),
    _check('paired_compare dùng hiệu từng cặp', lambda: ((lambda r: np.isclose(r['mean'], .01) and np.isclose(r['std'], np.std([.01, .012, .008, .011, .009], ddof=1)))(paired_compare(_s28, _s28 - np.array([.01, .012, .008, .011, .009]))), 'Hiệu cặp ổn định thì khoảng tin cậy hẹp.')),
)
''')]))

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'src/track-py/week4.json'
out.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding='utf-8')
print('ghi', out, len(L), 'bài,', sum(len(l['exercises']) for l in L), 'bài tập')
