"""Sinh src/track-py/week3.json (Ngày 15–21): cây, tổ hợp và học không nhãn.

    python scripts/gen_week3.py
Sau đó: python ../linear-regression-lab/py30/build_notebooks.py và verify_py30.py --days 15-21.
Mọi số trong phần "Python · Ví dụ tính từng dòng" đã được tính lại bằng code (xem /home/claude/w3/p15–p21 khi soạn)."""
import json, sys
from pathlib import Path

BASE = '''import json, warnings
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

BREAST = '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Breast Cancer Wisconsin (Diagnostic), UCI, có sẵn trong scikit-learn. Cùng phép chia với tuần 2.
_bc = load_breast_cancer()
X_bc, y_bc = _bc.data, 1 - _bc.target   # 1 = ác tính
idx_tr, idx_va = train_test_split(np.arange(len(y_bc)), test_size=0.3, random_state=0, stratify=y_bc)
_scaler = StandardScaler().fit(X_bc[idx_tr])
X_tr, X_va = _scaler.transform(X_bc[idx_tr]), _scaler.transform(X_bc[idx_va])
y_tr, y_va = y_bc[idx_tr], y_bc[idx_va]
'''

DIGITS = '''from sklearn.datasets import load_digits
# Optical Recognition of Handwritten Digits (UCI), có sẵn trong scikit-learn: 1797 ảnh 8×8, giá trị 0–16.
X_dg, y_dg = load_digits(return_X_y=True)
'''

def lesson(**k):
    return k

L = []

# ───────────────────────── Ngày 15 ─────────────────────────
L.append(lesson(
 id=15, title='Cây quyết định CART', projectTitle='Titanic',
 goals=['Tính độ bẩn Gini và mức giảm Gini của một phép tách', 'Tìm phép tách tốt nhất bằng cách quét mọi ngưỡng giữa hai giá trị liền kề', 'Dựng cây đệ quy có max_depth và thấy cây sâu quá khớp'],
 reads=['ISLR (bản 2), §8.1 The Basics of Decision Trees (8.1.1–8.1.2)', 'Hastie, Tibshirani & Friedman, ESL, §9.2 Tree-Based Methods', 'Breiman, Friedman, Olshen & Stone (1984), Classification and Regression Trees', 'scikit-learn User Guide, §1.10 Decision Trees (mục Mathematical formulation)'],
 project='Trên Titanic, thay LogisticRegression trong pipeline ngày 14 bằng DecisionTreeClassifier(max_depth=d, random_state=0) với d = 1…10. Vẽ accuracy train và accuracy stratified 5-fold theo d, chọn d và giải thích chỗ hai đường tách xa nhau. In ba tầng đầu của cây bằng sklearn.tree.export_text.',
 questions=['Vì sao chỉ cần thử ngưỡng ở giữa hai giá trị liền kề đã sắp xếp?', 'Cây sâu tối đa có accuracy train bao nhiêu nếu không có hai mẫu trùng đặc trưng mà khác nhãn?', 'Vì sao cây không cần chuẩn hóa đặc trưng?'],
 answers=['Mọi ngưỡng nằm giữa cùng hai giá trị liền kề cho cùng một cách chia dữ liệu, nên chỉ có n − 1 cách chia khác nhau trên mỗi cột.', '100%: cây cứ tách đến khi mỗi lá thuần, nên nó thuộc lòng tập train; đó là quá khớp.', 'Phép tách chỉ so sánh xⱼ ≤ t; một phép biến đổi đơn điệu như chuẩn hóa giữ nguyên thứ tự nên giữ nguyên cách chia.'],
 steps=[{'line': 'x = [1, 2, 3, 4, 5, 6]; y = [0, 0, 1, 0, 1, 1]', 'state': 'gini(y) = 1 − (0,5² + 0,5²) = 0,5'},
        {'line': 't = 1,5 → trái [0], phải [0, 1, 0, 1, 1]', 'state': 'gini 0 và 0,48 → giảm 0,5 − 5/6·0,48 = 0,10'},
        {'line': 't = 2,5 → trái [0, 0], phải [1, 0, 1, 1]', 'state': 'gini 0 và 0,375 → giảm 0,25'},
        {'line': 't = 3,5 → trái [0, 0, 1], phải [0, 1, 1]', 'state': 'gini 0,444 và 0,444 → giảm 0,056'},
        {'line': 't = 4,5 → trái [0, 0, 1, 0], phải [1, 1]', 'state': 'gini 0,375 và 0 → giảm 0,25 (hòa với 2,5)'},
        {'line': 'best = ngưỡng đầu tiên có mức giảm lớn nhất', 'state': 'best_split → (cột 0, t = 2,5, giảm 0,25)'}],
 setup=BASE + BREAST + '''from sklearn.tree import DecisionTreeClassifier
_x15 = np.array([[1.], [2.], [3.], [4.], [5.], [6.]]); _y15 = np.array([0, 0, 1, 0, 1, 1])
_sk15 = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
def _best_split_ref(X, y):
    n, d = X.shape; C = int(y.max()) + 1; best = (None, None, 0.0)
    parent = 1 - np.sum((np.bincount(y, minlength=C) / n) ** 2)
    for j in range(d):
        o = np.argsort(X[:, j], kind='stable'); xs, ys = X[o, j], y[o]
        left = np.cumsum(np.eye(C)[ys], axis=0)[:-1]; nl = np.arange(1, n)[:, None]
        right = left[-1] + np.eye(C)[ys[-1]] - left; nr = n - nl
        gain = parent - (nl[:, 0] * (1 - ((left / nl) ** 2).sum(1)) + nr[:, 0] * (1 - ((right / nr) ** 2).sum(1))) / n
        gain = np.where(xs[1:] > xs[:-1], gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best[2] + 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best
def _same_as_sklearn(build, predict, depth):
    ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr).predict(X_va)
    return np.array_equal(np.asarray(predict(build(X_tr, y_tr, depth), X_va)), ref)
''',
 exercises=[dict(
  id='d15-split', title='Gini và phép tách tốt nhất',
  hint='Với mỗi cột: sắp xếp, dùng np.cumsum trên one-hot của nhãn để có số đếm bên trái ở mọi vị trí cắt. Chỉ giữ vị trí mà giá trị sau lớn hơn giá trị trước, ngưỡng là trung điểm của hai giá trị đó.',
  starter='''def gini(y):
    # y: mảng nhãn nguyên 0..C-1. Trả 1 - Σ p_c².
    raise NotImplementedError

def best_split(X, y):
    # Trả (cột, ngưỡng, mức giảm Gini). Không có phép tách nào làm giảm Gini thì trả (None, None, 0.0).
    # Hòa nhau: giữ cột nhỏ hơn, rồi ngưỡng nhỏ hơn.
    raise NotImplementedError
''',
  solution='''def gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return float(1 - np.sum(p ** 2))

def best_split(X, y):
    n, d = X.shape
    C = int(y.max()) + 1
    parent = gini(y)
    best = (None, None, 0.0)
    for j in range(d):
        order = np.argsort(X[:, j], kind='stable')
        xs, ys = X[order, j], y[order]
        left = np.cumsum(np.eye(C)[ys], axis=0)[:-1]          # số đếm từng lớp bên trái
        right = left[-1] + np.eye(C)[ys[-1]] - left
        nl = np.arange(1, n)[:, None]; nr = n - nl
        g_left = 1 - ((left / nl) ** 2).sum(1)
        g_right = 1 - ((right / nr) ** 2).sum(1)
        gain = parent - (nl[:, 0] * g_left + nr[:, 0] * g_right) / n
        gain = np.where(xs[1:] > xs[:-1], gain, -np.inf)      # không cắt giữa hai giá trị bằng nhau
        i = int(np.argmax(gain))
        if gain[i] > best[2] + 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, float(gain[i]))
    return best
''',
  test='''_report('d15-split',
    _check('Gini đúng cho 2 và 3 lớp', lambda: (np.isclose(gini(np.array([0, 0, 1, 1])), .5) and np.isclose(gini(np.array([1, 1, 1])), 0) and np.isclose(gini(np.array([0, 1, 2])), 2 / 3), 'gini([0,0,1,1]) = 0,5; gini([0,1,2]) = 2/3.')),
    _check('Ví dụ tính tay: (cột 0, t = 2,5, giảm 0,25)', lambda: ((lambda b: b[0] == 0 and np.isclose(b[1], 2.5) and np.isclose(b[2], .25))(best_split(_x15, _y15)), 'Hòa với t = 4,5 thì giữ ngưỡng nhỏ hơn.')),
    _check('Không cắt giữa hai giá trị bằng nhau', lambda: ((lambda b: b[0] == 0 and np.isclose(b[1], 1.5))(best_split(np.array([[1.], [1.], [1.], [2.]]), np.array([0, 0, 1, 1]))), 'x = [1, 1, 1, 2]: chỉ có ngưỡng 1,5.')),
    _check('Gốc cây khớp DecisionTreeClassifier trên Breast Cancer', lambda: ((lambda b: b[0] == _sk15.tree_.feature[0] and np.isclose(b[1], _sk15.tree_.threshold[0], atol=1e-5))(best_split(X_tr, y_tr)), f'scikit-learn chọn cột {_sk15.tree_.feature[0]} ({_bc.feature_names[_sk15.tree_.feature[0]]}).')),
)
'''), dict(
  id='d15-tree', title='Dựng cây đệ quy và dự đoán',
  hint='Nút là dict. Dừng khi depth = max_depth, khi số mẫu < min_samples_split, khi nút đã thuần, hoặc khi best_split trả None. Lá dự đoán lớp đa số (hòa thì lớp nhỏ hơn: np.argmax trên np.bincount).',
  starter='''def build_tree(X, y, max_depth, min_samples_split=2, depth=0):
    # Lá: {'leaf': True, 'value': lớp}. Nút trong: {'leaf': False, 'feature', 'threshold', 'left', 'right'}.
    # Mẫu có X[:, feature] <= threshold đi sang trái. Dùng best_split của bài trên.
    raise NotImplementedError

def predict_tree(tree, X):
    raise NotImplementedError
''',
  solution='''def build_tree(X, y, max_depth, min_samples_split=2, depth=0):
    leaf = {'leaf': True, 'value': int(np.argmax(np.bincount(y)))}
    if depth >= max_depth or len(y) < min_samples_split or np.all(y == y[0]):
        return leaf
    j, t, _ = best_split(X, y)
    if j is None:
        return leaf
    m = X[:, j] <= t
    return {'leaf': False, 'feature': j, 'threshold': t,
            'left': build_tree(X[m], y[m], max_depth, min_samples_split, depth + 1),
            'right': build_tree(X[~m], y[~m], max_depth, min_samples_split, depth + 1)}

def predict_tree(tree, X):
    out = []
    for x in X:
        node = tree
        while not node['leaf']:
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        out.append(node['value'])
    return np.array(out)
''',
  test='''best_split = globals().get('best_split') or _best_split_ref
_report('d15-tree',
    _check('Khớp scikit-learn với max_depth = 1, 2, 3', lambda: (all(_same_as_sklearn(build_tree, predict_tree, d) for d in (1, 2, 3)), 'So dự đoán trên 171 mẫu validation.')),
    _check('Cây không giới hạn thuộc lòng tập train', lambda: (np.mean(predict_tree(build_tree(X_tr, y_tr, 100), X_tr) == y_tr) == 1.0, 'Accuracy train = 1 là dấu hiệu quá khớp, không phải mục tiêu.')),
    _check('max_depth = 0 cho một lá dự đoán lớp đa số', lambda: (np.all(predict_tree(build_tree(X_tr, y_tr, 0), X_va) == 0), 'Lớp đa số của train là 0 (lành tính).')),
)
''')]))

# ───────────────────────── Ngày 16 ─────────────────────────
L.append(lesson(
 id=16, title='Bagging, Random Forest và lỗi out-of-bag', projectTitle='House Prices',
 goals=['Lấy mẫu bootstrap và xác định các mẫu out-of-bag (OOB)', 'Ghép nhiều cây trên mẫu bootstrap, mỗi lần tách chỉ xét √d cột', 'Ước lượng lỗi bằng OOB mà không cần tập validation riêng'],
 reads=['ISLR (bản 2), §8.2.1 Bagging (đoạn Out-of-Bag Error Estimation) và §8.2.2 Random Forests', 'Breiman (2001), Random Forests, Machine Learning 45(1), 5–32', 'Hastie, Tibshirani & Friedman, ESL, §8.7 Bagging và Chương 15 Random Forests'],
 project='Tải House Prices (python py30/download_data.py house). Dùng các cột số (điền thiếu bằng median), dự đoán log(SalePrice) bằng RandomForestRegressor(n_estimators=300, oob_score=True, random_state=0). So RMSE OOB (tính từ oob_prediction_) với RMSE 5-fold KFold(shuffle=True, random_state=0). Liệt kê 10 cột quan trọng nhất theo feature_importances_ và theo permutation_importance trên fold validation; ghi chỗ hai cách xếp hạng khác nhau.',
 questions=['Vì sao mỗi mẫu bootstrap chỉ chứa khoảng 63% số điểm khác nhau?', 'Random Forest khác bagging ở điểm nào và vì sao điều đó giúp?', 'Lỗi OOB có thay thế được tập test cuối cùng không?'],
 answers=['Xác suất một điểm không bao giờ được chọn trong n lần là (1 − 1/n)ⁿ ≈ e⁻¹ ≈ 0,368, nên khoảng 63,2% điểm có mặt.', 'Mỗi lần tách chỉ xét một tập con ngẫu nhiên các cột, nên các cây ít giống nhau hơn; trung bình các cây ít tương quan giảm phương sai nhiều hơn.', 'Nó là ước lượng gần như không chệch cho mô hình đang có, nhưng nếu bạn dùng nó để chọn siêu tham số thì nó lại trở nên lạc quan; vẫn cần giữ một tập test khóa.'],
 steps=[{'line': 'rng = np.random.default_rng(0); idx = rng.integers(0, 6, 6)', 'state': 'idx = [5, 3, 3, 1, 1, 0]'},
        {'line': 'oob = np.setdiff1d(np.arange(6), idx)', 'state': 'oob = [2, 4] (2/6 mẫu, gần (5/6)⁶ ≈ 0,335)'},
        {'line': 'tree = DecisionTreeClassifier(max_features="sqrt").fit(X[idx], y[idx])', 'state': 'Breast Cancer: √30 ≈ 5 cột được xét ở mỗi lần tách'},
        {'line': 'votes = mean([t.predict(X_va) for t in trees])', 'state': '50 cây, seed 0: accuracy trung bình từng cây 0,917'},
        {'line': 'pred = (votes >= 0.5)', 'state': 'accuracy của cả rừng trên validation 0,942'},
        {'line': 'oob_accuracy(forest, X_tr, y_tr)', 'state': 'OOB 0,952 (mỗi mẫu train chỉ được bỏ phiếu bởi cây không thấy nó)'}],
 setup=BASE + BREAST + '''from sklearn.tree import DecisionTreeClassifier
def _bootstrap_ref(n, rng):
    idx = rng.integers(0, n, n)
    return idx, np.setdiff1d(np.arange(n), idx)
def _ref_oob(forest, X, y):
    votes, count = np.zeros(len(y)), np.zeros(len(y))
    for tree, oob in forest:
        votes[oob] += tree.predict(X[oob]); count[oob] += 1
    m = count > 0
    return float(np.mean((votes[m] / count[m] >= .5).astype(int) == y[m]))
def _forest_shape_ok(forest, n, n_trees):
    sizes = [len(o) for _, o in forest]
    return (len(forest) == n_trees and all(.25 * n < s < .5 * n for s in sizes)
            and len({tuple(o) for _, o in forest}) > 1)
''',
 exercises=[dict(
  id='d16-bootstrap', title='Mẫu bootstrap và chỉ số out-of-bag',
  hint='rng.integers(0, n, n) lấy n chỉ số có hoàn lại. OOB là các chỉ số trong 0..n−1 không xuất hiện: np.setdiff1d.',
  starter='''def bootstrap_indices(n, rng):
    # rng là np.random.Generator. Trả (idx, oob): idx dài n (có hoàn lại), oob là các chỉ số không được chọn, đã sắp xếp.
    raise NotImplementedError
''',
  solution='''def bootstrap_indices(n, rng):
    idx = rng.integers(0, n, n)
    return idx, np.setdiff1d(np.arange(n), idx)
''',
  test='''_report('d16-bootstrap',
    _check('Có hoàn lại: khoảng 63,2% chỉ số khác nhau', lambda: ((lambda i: len(i) == 10000 and i.min() >= 0 and i.max() < 10000 and .62 < len(np.unique(i)) / 10000 < .65)(np.asarray(bootstrap_indices(10000, np.random.default_rng(1))[0])), 'n = 10000, kỳ vọng 1 − e⁻¹ ≈ 0,632.')),
    _check('OOB đúng bằng phần bù của idx', lambda: ((lambda r: np.array_equal(np.asarray(r[1]), np.setdiff1d(np.arange(50), r[0])))(bootstrap_indices(50, np.random.default_rng(2))), 'n = 50.')),
    _check('Dùng đúng rng được truyền vào', lambda: ((lambda a, b: np.array_equal(a[0], b[0]))(bootstrap_indices(30, np.random.default_rng(7)), bootstrap_indices(30, np.random.default_rng(7))) and not np.array_equal(*(lambda r: (bootstrap_indices(30, r)[0], bootstrap_indices(30, r)[0]))(np.random.default_rng(7))), 'Cùng seed thì giống nhau; hai lần gọi trên cùng rng thì khác nhau.')),
)
'''), dict(
  id='d16-forest', title='Random Forest tự ghép và accuracy OOB',
  hint='Mỗi cây i: lấy bootstrap bằng một rng = np.random.default_rng(seed) tạo một lần ở đầu, rồi DecisionTreeClassifier(max_features=max_features, random_state=seed + i). Lưu cặp (cây, oob). OOB: mỗi mẫu chỉ nhận phiếu từ cây mà mẫu đó nằm trong oob.',
  starter='''from sklearn.tree import DecisionTreeClassifier

def fit_forest(X, y, n_trees, max_features, seed):
    # Trả list n_trees cặp (cây đã fit, mảng chỉ số oob). Dùng bootstrap_indices của bài trên.
    raise NotImplementedError

def predict_forest(forest, X):
    # Bỏ phiếu đa số; tỉ lệ phiếu lớp 1 >= 0,5 thì dự đoán 1.
    raise NotImplementedError

def oob_accuracy(forest, X, y):
    # Accuracy trên các mẫu có ít nhất một cây không thấy chúng.
    raise NotImplementedError
''',
  solution='''from sklearn.tree import DecisionTreeClassifier

def fit_forest(X, y, n_trees, max_features, seed):
    rng = np.random.default_rng(seed)
    forest = []
    for i in range(n_trees):
        idx, oob = bootstrap_indices(len(y), rng)
        tree = DecisionTreeClassifier(max_features=max_features, random_state=seed + i).fit(X[idx], y[idx])
        forest.append((tree, oob))
    return forest

def predict_forest(forest, X):
    votes = np.mean([tree.predict(X) for tree, _ in forest], axis=0)
    return (votes >= 0.5).astype(int)

def oob_accuracy(forest, X, y):
    votes, count = np.zeros(len(y)), np.zeros(len(y))
    for tree, oob in forest:
        votes[oob] += tree.predict(X[oob])
        count[oob] += 1
    seen = count > 0
    return float(np.mean((votes[seen] / count[seen] >= 0.5).astype(int) == y[seen]))
''',
  test='''bootstrap_indices = globals().get('bootstrap_indices') or _bootstrap_ref
_c16 = {}
def _f16():
    if 'f' not in _c16:
        _c16['f'] = fit_forest(X_tr, y_tr, 50, 'sqrt', 0)
    return _c16['f']
_report('d16-forest',
    _check('50 cây, mỗi cây có tập OOB riêng cỡ khoảng 37%', lambda: (_forest_shape_ok(_f16(), len(y_tr), 50), 'Kiểm tra cấu trúc (cây, oob).')),
    _check('Bỏ phiếu đa số của mọi cây (tỉ lệ ≥ 0,5 → 1)', lambda: (np.array_equal(predict_forest(_f16(), X_va), (np.mean([t.predict(X_va) for t, _ in _f16()], axis=0) >= .5).astype(int)), 'So với phiếu của 50 cây trên validation.')),
    _check('Cả rừng tốt hơn trung bình từng cây ít nhất 1 điểm %', lambda: ((lambda f, ind: f >= ind + .01 and f >= .93)(np.mean(predict_forest(_f16(), X_va) == y_va), np.mean([np.mean(t.predict(X_va) == y_va) for t, _ in _f16()])), 'Trên validation, max_features = "sqrt".')),
    _check('Accuracy OOB khớp cách tính tham chiếu', lambda: (np.isclose(oob_accuracy(_f16(), X_tr, y_tr), _ref_oob(_f16(), X_tr, y_tr)), 'Chỉ cây không thấy mẫu mới được bỏ phiếu cho mẫu đó.')),
)
''')]))

# ───────────────────────── Ngày 17 ─────────────────────────
L.append(lesson(
 id=17, title='Gradient boosting cho hồi quy', projectTitle='House Prices',
 goals=['Viết vòng boosting: mỗi cây nhỏ khớp phần dư y − F', 'Thấy vai trò của learning rate và số cây', 'Chọn số cây bằng sai số validation theo từng giai đoạn (early stopping)'],
 reads=['ISLR (bản 2), §8.2.3 Boosting (Algorithm 8.2)', 'Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics 29(5), 1189–1232', 'Hastie, Tibshirani & Friedman, ESL, §10.10 Numerical Optimization via Gradient Boosting'],
 project='Trên House Prices với log(SalePrice), so RandomForestRegressor với HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.1, random_state=0) theo RMSE 5-fold. Thử learning_rate ∈ {0,3; 0,1; 0,03} và ghi n_iter_ mà early stopping chọn. Xuất submission và tự nộp trên Kaggle (nhớ đổi ngược bằng np.expm1 nếu bạn dùng log1p).',
 questions=['Với hàm mất mát bình phương, vì sao "khớp phần dư" là đi theo gradient âm?', 'Giảm learning rate thì số cây tốt nhất thay đổi thế nào?', 'Vì sao boosting dùng cây nông còn Random Forest dùng cây sâu?'],
 answers=['L = ½(y − F)² nên −∂L/∂F = y − F, đúng bằng phần dư; mỗi cây xấp xỉ gradient âm này.', 'Tăng lên: mỗi bước nhỏ hơn nên cần nhiều cây hơn, đổi lại thường đạt lỗi validation thấp hơn một chút.', 'Boosting giảm bias bằng cách cộng dần nhiều mô hình yếu, mỗi cây nông có phương sai thấp; Random Forest giảm phương sai bằng trung bình nên dùng cây sâu có bias thấp.'],
 steps=[{'line': 'x = [1, 2, 3, 4]; y = [1, 3, 2, 6]; F = mean(y)', 'state': 'F = [3, 3, 3, 3], MSE = 3,5'},
        {'line': 'r = y - F', 'state': 'r = [−2, 0, −1, 3]'},
        {'line': 'tree = DecisionTreeRegressor(max_depth=1).fit(x, r)', 'state': 'tách tại x ≤ 3,5: lá trái −1, lá phải 3'},
        {'line': 'F = F + lr * tree.predict(x)', 'state': 'lr = 0,5 → F = [2,5, 2,5, 2,5, 4,5]'},
        {'line': 'mse = np.mean((y - F) ** 2)', 'state': 'MSE = 1,25; phần dư mới [−1,5, 0,5, −0,5, 1,5]'},
        {'line': 'best_n = argmin(staged_mse(model, X_va, y_va)) + 1', 'state': 'Diabetes, depth 2, lr 0,1: tốt nhất ở cây thứ 15 (MSE 3216), cây thứ 200 cho 3734'}],
 setup=BASE + '''from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# Diabetes (Efron et al., 2004), có sẵn trong scikit-learn: 442 bệnh nhân, 10 đặc trưng, y là chỉ số tiến triển sau 1 năm.
X_db, y_db = load_diabetes(return_X_y=True)
Xd_tr, Xd_va, yd_tr, yd_va = train_test_split(X_db, y_db, test_size=0.3, random_state=0)
_sk17 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0).fit(Xd_tr, yd_tr)
def _ref_staged(model, X, y):
    F, out = np.full(len(X), model['f0']), []
    for tree in model['trees']:
        F = F + model['lr'] * tree.predict(X)
        out.append(float(np.mean((y - F) ** 2)))
    return np.array(out)
''',
 exercises=[dict(
  id='d17-boost', title='Boosting từ cây hồi quy và early stopping',
  hint='F khởi đầu bằng mean(y). Mỗi vòng: r = y − F, fit DecisionTreeRegressor(max_depth=max_depth, random_state=0) lên (X, r), rồi F += lr · tree.predict(X). Lưu f0, lr và danh sách cây vào dict.',
  starter='''from sklearn.tree import DecisionTreeRegressor

def fit_gbm(X, y, n_estimators, lr, max_depth):
    # Trả {'f0': float, 'lr': lr, 'trees': [cây 1, ..., cây n]}.
    raise NotImplementedError

def predict_gbm(model, X):
    raise NotImplementedError

def staged_mse(model, X, y):
    # Mảng dài n_estimators: MSE sau khi cộng cây thứ 1, 2, ...
    raise NotImplementedError

def best_n_estimators(model, X_val, y_val):
    # Số cây (đếm từ 1) cho MSE validation nhỏ nhất.
    raise NotImplementedError
''',
  solution='''from sklearn.tree import DecisionTreeRegressor

def fit_gbm(X, y, n_estimators, lr, max_depth):
    f0 = float(np.mean(y))
    F = np.full(len(y), f0)
    trees = []
    for _ in range(n_estimators):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, y - F)
        F = F + lr * tree.predict(X)
        trees.append(tree)
    return {'f0': f0, 'lr': lr, 'trees': trees}

def predict_gbm(model, X):
    F = np.full(len(X), model['f0'])
    for tree in model['trees']:
        F = F + model['lr'] * tree.predict(X)
    return F

def staged_mse(model, X, y):
    F, out = np.full(len(X), model['f0']), []
    for tree in model['trees']:
        F = F + model['lr'] * tree.predict(X)
        out.append(float(np.mean((y - F) ** 2)))
    return np.array(out)

def best_n_estimators(model, X_val, y_val):
    return int(np.argmin(staged_mse(model, X_val, y_val)) + 1)
''',
  test='''_c17 = {}
def _m17():
    if 'm' not in _c17:
        _c17['m'] = fit_gbm(Xd_tr, yd_tr, 100, 0.1, 2)
    return _c17['m']
_report('d17-boost',
    _check('Dự đoán trên train khớp GradientBoostingRegressor', lambda: (np.allclose(predict_gbm(_m17(), Xd_tr), _sk17.predict(Xd_tr), atol=1e-6), '100 cây, lr = 0,1, max_depth = 2, loss bình phương.')),
    _check('staged_mse trên train khớp staged_predict và không tăng', lambda: ((lambda s: np.allclose(s, [np.mean((yd_tr - p) ** 2) for p in _sk17.staged_predict(Xd_tr)]) and np.all(np.diff(s) <= 1e-9))(staged_mse(_m17(), Xd_tr, yd_tr)), 'Lỗi train của boosting giảm đơn điệu.')),
    _check('best_n_estimators chọn đúng số cây theo validation', lambda: (best_n_estimators(_m17(), Xd_va, yd_va) == int(np.argmin(_ref_staged(_m17(), Xd_va, yd_va)) + 1), 'Lỗi validation giảm rồi tăng: đó là chỗ dừng sớm.')),
)
''')]))

# ───────────────────────── Ngày 18 ─────────────────────────
L.append(lesson(
 id=18, title='K-means và Gaussian Mixture (EM)', projectTitle='Digit Recognizer',
 goals=['Cài vòng Lloyd: gán về tâm gần nhất rồi dời tâm về trung bình', 'Thấy hàm mục tiêu (inertia) không bao giờ tăng nhưng có thể kẹt ở cực tiểu địa phương', 'Viết bước E và bước M cho hỗn hợp Gauss với ma trận hiệp phương sai đầy đủ'],
 reads=['ISLR (bản 2), §12.4.1 K-Means Clustering', 'Ng & Ma, CS229 Lecture Notes (2023), Chương 10 Clustering and the k-means algorithm; §11.1 EM for mixture of Gaussians', 'Arthur & Vassilvitskii (2007), k-means++: The Advantages of Careful Seeding, SODA', 'Dempster, Laird & Rubin (1977), Maximum Likelihood from Incomplete Data via the EM Algorithm, JRSS-B 39(1), 1–38'],
 project='Tải Digit Recognizer (python py30/download_data.py digits) hoặc dùng load_digits khi chưa có Kaggle. Chạy KMeans(n_clusters=10, n_init=10, random_state=0) trên ảnh, vẽ 10 tâm cụm thành ảnh, rồi tính adjusted_rand_score với nhãn thật. Ghi cặp chữ số nào hay bị gộp chung một cụm.',
 questions=['Vì sao inertia của vòng Lloyd không bao giờ tăng?', 'K-means là trường hợp riêng nào của GMM?', 'Vì sao phải cộng một lượng nhỏ (reg_covar) vào đường chéo của ma trận hiệp phương sai?'],
 answers=['Bước gán chọn tâm gần nhất nên không tăng tổng; bước cập nhật lấy trung bình, là điểm làm nhỏ nhất tổng bình phương trong cụm.', 'GMM với mọi hiệp phương sai bằng σ²I, trọng số bằng nhau và σ → 0: trách nhiệm mềm thành gán cứng về tâm gần nhất.', 'Một thành phần có thể co về vài điểm làm ma trận suy biến, log-likelihood tiến tới vô cùng; cộng εI giữ ma trận khả nghịch.'],
 steps=[{'line': 'X = [(0,0), (1,0), (0,1), (5,5), (6,5), (5,6)]; C = [(0,0), (1,0)]', 'state': 'khởi tạo tệ: cả hai tâm trong cụm dưới'},
        {'line': 'labels = argmin ‖x − C_k‖²', 'state': 'labels = [0, 1, 0, 1, 1, 1], inertia = 144'},
        {'line': 'C[k] = X[labels == k].mean(0)', 'state': 'C = [(0, 0,5), (4,25, 4)]'},
        {'line': 'labels = argmin ‖x − C_k‖² (vòng 2)', 'state': 'labels = [0, 0, 0, 1, 1, 1], inertia = 11,94'},
        {'line': 'C[k] = X[labels == k].mean(0)', 'state': 'C = [(0,33, 0,33), (5,33, 5,33)], inertia = 2,67'},
        {'line': 'nhãn không đổi → dừng', 'state': 'Iris (4 cột chuẩn hóa), khởi tạo 3 bông setosa: dừng sau 12 vòng, inertia 140,03 (tốt nhất trong 10 lần k-means++: 139,82)'}],
 setup=BASE + '''from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
# Iris (Fisher, 1936), có sẵn trong scikit-learn: 150 bông, 4 đặc trưng, chuẩn hóa từng cột.
_iris = load_iris()
X_ir = (_iris.data - _iris.data.mean(0)) / _iris.data.std(0)
_init18 = X_ir[[0, 1, 2]]                   # ba bông setosa: khởi tạo cố tình tệ
_sk18 = KMeans(3, init=_init18, n_init=1, algorithm='lloyd', max_iter=300, tol=0).fit(X_ir)
_w0, _mu0, _cov0 = np.array([.5, .3, .2]), _sk18.cluster_centers_.copy(), np.array([np.eye(4)] * 3)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')        # max_iter = 1 cố ý: chỉ một vòng EM để so
    _gm18 = GaussianMixture(3, covariance_type='full', weights_init=_w0, means_init=_mu0,
                            precisions_init=np.array([np.linalg.inv(c) for c in _cov0]), max_iter=1, tol=0, reg_covar=1e-6).fit(X_ir)
''',
 exercises=[dict(
  id='d18-kmeans', title='Vòng Lloyd của k-means',
  hint='assign: ma trận khoảng cách bình phương (n × K) bằng broadcasting, argmin theo trục 1. update: cụm rỗng giữ tâm cũ. kmeans: ghi inertia ngay sau mỗi bước gán, dừng khi tâm không đổi (np.allclose).',
  starter='''def assign(X, C):
    raise NotImplementedError

def update(X, labels, C):
    # Trả mảng tâm mới; cụm không có điểm nào thì giữ nguyên tâm cũ.
    raise NotImplementedError

def kmeans(X, init, iters=100):
    # Trả (C, labels, history) với history là list inertia sau mỗi lần gán.
    raise NotImplementedError
''',
  solution='''def assign(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

def update(X, labels, C):
    C = np.array(C, dtype=float, copy=True)
    for k in range(len(C)):
        if np.any(labels == k):
            C[k] = X[labels == k].mean(axis=0)
    return C

def kmeans(X, init, iters=100):
    C = np.array(init, dtype=float)
    history = []
    for _ in range(iters):
        labels = assign(X, C)
        history.append(float(((X - C[labels]) ** 2).sum()))
        new = update(X, labels, C)
        if np.allclose(new, C):
            break
        C = new
    return C, assign(X, C), history
''',
  test='''_report('d18-kmeans',
    _check('assign chọn tâm gần nhất', lambda: (np.array_equal(assign(X_ir, _init18 + .3), ((X_ir[:, None] - (_init18 + .3)[None]) ** 2).sum(2).argmin(1)), 'So với cách tính trực tiếp.')),
    _check('Cụm rỗng giữ nguyên tâm cũ', lambda: (np.allclose(update(np.array([[0., 0.], [2., 0.]]), np.array([0, 0]), np.array([[9., 9.], [5., 5.]])), [[1., 0.], [5., 5.]]), 'Không được sinh NaN.')),
    _check('Kết quả khớp KMeans(algorithm="lloyd") cùng khởi tạo', lambda: ((lambda r: np.allclose(r[0], _sk18.cluster_centers_, atol=1e-8) and np.array_equal(r[1], _sk18.labels_))(kmeans(X_ir, _init18)), f'Inertia tham chiếu {_sk18.inertia_:.2f}.')),
    _check('Inertia không bao giờ tăng', lambda: (np.all(np.diff(kmeans(X_ir, _init18)[2]) <= 1e-9), 'Mỗi bước gán và cập nhật chỉ làm hàm mục tiêu giảm hoặc giữ nguyên.')),
)
'''), dict(
  id='d18-gmm', title='Bước E và bước M của hỗn hợp Gauss',
  hint='Bước E: log p(x, k) = log w_k + multivariate_normal(mu_k, cov_k).logpdf(x); chuẩn hóa bằng logsumexp theo k. Bước M: N_k = Σ_i r_ik; w = N_k/n; mu_k = Σ r_ik x_i / N_k; cov_k = Σ r_ik (x_i − mu_k)(x_i − mu_k)ᵀ / N_k + reg·I.',
  starter='''def e_step(X, w, mu, cov):
    # Trả (resp shape (n, K), log-likelihood trung bình mỗi mẫu).
    raise NotImplementedError

def m_step(X, resp, reg=1e-6):
    # Trả (w, mu, cov).
    raise NotImplementedError
''',
  solution='''def e_step(X, w, mu, cov):
    log_joint = np.column_stack([np.log(w[k]) + multivariate_normal(mu[k], cov[k]).logpdf(X) for k in range(len(w))])
    log_like = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - log_like[:, None]), float(log_like.mean())

def m_step(X, resp, reg=1e-6):
    Nk = resp.sum(axis=0)
    w = Nk / len(X)
    mu = (resp.T @ X) / Nk[:, None]
    cov = np.array([((resp[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + reg * np.eye(X.shape[1])
                    for k in range(len(w))])
    return w, mu, cov
''',
  test='''def _em18(n):
    w, mu, c, lls = _w0, _mu0, _cov0, []
    for _ in range(n):
        R, ll = e_step(X_ir, w, mu, c); lls.append(ll); w, mu, c = m_step(X_ir, R)
    return w, mu, c, lls
_report('d18-gmm',
    _check('Trách nhiệm mỗi hàng cộng lại bằng 1', lambda: (np.allclose(e_step(X_ir, _w0, _mu0, _cov0)[0].sum(1), 1), 'Bước E.')),
    _check('Một vòng EM khớp GaussianMixture(max_iter=1)', lambda: ((lambda r: np.allclose(r[0], _gm18.weights_) and np.allclose(r[1], _gm18.means_) and np.allclose(r[2], _gm18.covariances_))(_em18(1)), 'reg_covar = 1e-6.')),
    _check('Log-likelihood không giảm qua 40 vòng', lambda: (np.all(np.diff(_em18(40)[3]) >= -1e-9), 'Tính chất đơn điệu của EM.')),
)
''')]))

# ───────────────────────── Ngày 19 ─────────────────────────
L.append(lesson(
 id=19, title='PCA bằng SVD', projectTitle='Digit Recognizer',
 goals=['Tính PCA từ SVD của dữ liệu đã trừ trung bình', 'Chiếu xuống k chiều, dựng lại và đo sai số dựng lại', 'Chọn k theo tỉ lệ phương sai giữ lại'],
 reads=['ISLR (bản 2), §12.2 Principal Components Analysis', 'Ng & Ma, CS229 Lecture Notes (2023), Chương 12 Principal components analysis', 'Hastie, Tibshirani & Friedman, ESL, §14.5.1 Principal Components'],
 project='Trên ảnh Digit Recognizer 28×28 (hoặc load_digits 8×8), vẽ 16 thành phần chính đầu tiên thành ảnh ("eigen-digit"), vẽ đường tỉ lệ phương sai tích lũy, và dựng lại 5 ảnh với k = 10, 50, 150. Ghi k nhỏ nhất giữ 95% phương sai.',
 questions=['Vì sao phải trừ trung bình trước khi làm SVD?', 'Thành phần chính có dấu cố định không?', 'Có nên chuẩn hóa từng cột trước PCA không?'],
 answers=['Không trừ trung bình thì thành phần đầu tiên chỉ hướng về trung bình của dữ liệu chứ không theo hướng phương sai lớn nhất.', 'Không: u và −u cùng là vector riêng; thư viện tự chọn một quy ước dấu, nên so sánh phải bỏ qua dấu.', 'Tùy: nếu các cột khác đơn vị đo thì nên, nếu không cột có thang đo lớn chiếm hết phương sai; với pixel cùng thang đo thì thường không cần.'],
 steps=[{'line': 'X = [[1, 1], [2, 2], [3, 3,5], [4, 3,5]]; Xc = X − mean', 'state': 'mean = [2,5, 2,5]'},
        {'line': 'U, S, Vt = np.linalg.svd(Xc, full_matrices=False)', 'state': 'Vt[0] = ±[0,726, 0,687]'},
        {'line': 'var = S**2 / (n − 1)', 'state': 'var = [3,086, 0,081]'},
        {'line': 'ratio = var / var.sum()', 'state': 'ratio = [0,974, 0,026]'},
        {'line': 'Z = Xc @ Vt[:1].T', 'state': 'Z = ±[−2,121, −0,707, 1,050, 1,777]'},
        {'line': 'n_components_for(X_digits, 0.95)', 'state': 'load_digits: 10 thành phần giữ 73,8%, 29 thành phần giữ 95%'}],
 setup=BASE + DIGITS + '''from sklearn.decomposition import PCA
_sk19 = PCA(10).fit(X_dg)
def _fit_pca_ref(X, k):
    mean = X.mean(0); _, S, Vt = np.linalg.svd(X - mean, full_matrices=False); ev = S ** 2 / (len(X) - 1)
    return {'mean': mean, 'components': Vt[:k], 'explained_variance': ev[:k], 'explained_variance_ratio': ev[:k] / ev.sum()}
''',
 exercises=[dict(
  id='d19-pca', title='fit, transform và dựng lại',
  hint='np.linalg.svd(X − mean, full_matrices=False). Hàng của Vt là các thành phần. Phương sai giải thích = S²/(n − 1). Dựng lại: Z @ components + mean.',
  starter='''def fit_pca(X, k):
    # Trả dict: mean, components (k, d), explained_variance (k,), explained_variance_ratio (k,).
    raise NotImplementedError

def transform(model, X):
    raise NotImplementedError

def inverse_transform(model, Z):
    raise NotImplementedError
''',
  solution='''def fit_pca(X, k):
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return {'mean': mean, 'components': Vt[:k], 'explained_variance': var[:k],
            'explained_variance_ratio': var[:k] / var.sum()}

def transform(model, X):
    return (X - model['mean']) @ model['components'].T

def inverse_transform(model, Z):
    return Z @ model['components'] + model['mean']
''',
  test='''_c19 = {}
def _m19():
    if 'm' not in _c19:
        _c19['m'] = fit_pca(X_dg, 10)
    return _c19['m']
_report('d19-pca',
    _check('Thành phần khớp scikit-learn (không xét dấu)', lambda: (np.allclose(np.abs((_m19()['components'] * _sk19.components_).sum(1)), 1, atol=1e-8), '10 thành phần trên load_digits.')),
    _check('Phương sai và tỉ lệ phương sai khớp', lambda: (np.allclose(_m19()['explained_variance'], _sk19.explained_variance_) and np.allclose(_m19()['explained_variance_ratio'], _sk19.explained_variance_ratio_), 'Chia cho n − 1.')),
    _check('Sai số dựng lại bằng tổng phương sai bị bỏ', lambda: ((lambda full: np.isclose(((X_dg - inverse_transform(_m19(), transform(_m19(), X_dg))) ** 2).sum(1).mean(), full['explained_variance'][10:].sum() * (len(X_dg) - 1) / len(X_dg)))(_fit_pca_ref(X_dg, 64)), 'Trung bình ‖x − x̂‖² = Σ phương sai các thành phần bị bỏ × (n − 1)/n.')),
)
'''), dict(
  id='d19-k', title='Chọn số thành phần theo phương sai giữ lại',
  hint='Lấy explained_variance_ratio của mọi thành phần, np.cumsum, rồi tìm vị trí đầu tiên ≥ ratio (np.searchsorted) và cộng 1 vì đếm từ 1.',
  starter='''def n_components_for(X, ratio):
    # Số thành phần nhỏ nhất có tổng tỉ lệ phương sai >= ratio. Dùng fit_pca của bài trên.
    raise NotImplementedError
''',
  solution='''def n_components_for(X, ratio):
    r = fit_pca(X, X.shape[1])['explained_variance_ratio']
    return int(np.searchsorted(np.cumsum(r), ratio) + 1)
''',
  test='''fit_pca = globals().get('fit_pca') or _fit_pca_ref
_report('d19-k',
    _check('Khớp PCA(n_components=ratio) của scikit-learn', lambda: (all(n_components_for(X_dg, r) == PCA(r, svd_solver='full').fit(X_dg).n_components_ for r in (.5, .8, .95)), 'ratio = 0,5; 0,8; 0,95.')),
    _check('Giữ 95% phương sai của load_digits cần 29 thành phần', lambda: (n_components_for(X_dg, .95) == 29, 'Trên 64 pixel.')),
)
''')]))

# ───────────────────────── Ngày 20 ─────────────────────────
L.append(lesson(
 id=20, title='Kỹ thuật đặc trưng và target encoding', projectTitle='House Prices',
 goals=['Mã hóa cột phân loại nhiều giá trị bằng trung bình mục tiêu có làm mượt', 'Thấy rò rỉ khi mã hóa trên chính các mẫu dùng để huấn luyện', 'Mã hóa out-of-fold để mỗi mẫu không thấy nhãn của chính nó'],
 reads=['Micci-Barreca (2001), A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, SIGKDD Explorations 3(1), 27–32', 'Kuhn & Johnson (2019), Feature Engineering and Selection, Chương 5 Encoding Categorical Predictors', 'ISLR (bản 2), §3.3.1 Qualitative Predictors (biến giả)'],
 project='Trên House Prices, so ba cách cho cột Neighborhood (25 giá trị) khi dự đoán log(SalePrice) bằng Ridge: one-hot, target encoding trong-mẫu (sai) và target encoding out-of-fold (đúng), đánh giá bằng 5-fold. Thêm log1p cho các cột lệch (GrLivArea, LotArea) và ghi thay đổi RMSE của từng bước.',
 questions=['Vì sao target encoding trong-mẫu rò rỉ nhãn?', 'Hệ số làm mượt m làm gì?', 'Giá trị phân loại mới ở test được mã hóa thế nào?'],
 answers=['Với nhóm ít mẫu, trung bình nhóm chứa chính nhãn của mẫu đang mã hóa, nên đặc trưng "nhìn thấy" đáp án; mô hình đạt điểm train cao giả tạo.', 'Kéo trung bình của nhóm nhỏ về trung bình toàn bộ: (Σy + m·ȳ)/(n_c + m); nhóm lớn gần như giữ trung bình riêng.', 'Bằng trung bình toàn bộ của phần dữ liệu đã dùng để fit (giá trị tiên nghiệm).'],
 steps=[{'line': 'cat = [a, a, b, b, b, c]; y = [1, 0, 1, 1, 0, 1]', 'state': 'ȳ = 4/6 = 0,667'},
        {'line': 'g = y.groupby(cat).agg(["sum", "count"])', 'state': 'a: (1, 2); b: (2, 3); c: (1, 1)'},
        {'line': 'enc = (sum + m*prior) / (count + m)', 'state': 'm = 2 → a = 0,583; b = 0,667; c = 0,778'},
        {'line': 'enc_test = map(cat_test).fillna(prior)', 'state': 'giá trị chưa gặp "d" → 0,667'},
        {'line': 'for tr, va in kfold_indices(n, 5, seed): out[va] = fit(tr).apply(va)', 'state': 'mỗi mẫu được mã hóa bằng các fold khác'},
        {'line': 'roc_auc_score(y, enc)', 'state': '300 mã định danh, nhãn ngẫu nhiên: trong-mẫu AUC 0,82; out-of-fold AUC 0,56'}],
 setup=BASE + '''from sklearn.metrics import roc_auc_score
def kfold_indices(n, k, seed):
    parts = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate(parts[:i] + parts[i + 1:]), parts[i]) for i in range(k)]
_r20 = np.random.default_rng(20)
_ids = np.array([f'id{i}' for i in _r20.integers(0, 300, 1000)]); _ynoise = _r20.integers(0, 2, 1000)   # giả lập: nhãn không liên quan
_cats = _r20.choice(list('ABCDEFGHIJ'), 1000)
_ysig = (_r20.random(1000) < 1 / (1 + np.exp(-np.vectorize(dict(zip('ABCDEFGHIJ', np.linspace(-1.5, 1.5, 10))).get)(_cats)))).astype(int)
def _ref_fit(cat, y, m):
    s = pd.Series(np.asarray(y, float)); g = s.groupby(pd.Series(cat)).agg(['sum', 'count']); prior = float(s.mean())
    return {'map': ((g['sum'] + m * prior) / (g['count'] + m)).to_dict(), 'prior': prior}
def _ref_oof(cat, y, k, seed, m):
    cat, y, out = np.asarray(cat, dtype=object), np.asarray(y, float), np.empty(len(y))
    for tr, va in kfold_indices(len(y), k, seed):
        e = _ref_fit(cat[tr], y[tr], m); out[va] = pd.Series(cat[va]).map(e['map']).fillna(e['prior']).to_numpy(float)
    return out
''',
 exercises=[dict(
  id='d20-target', title='Target encoding có làm mượt và out-of-fold',
  hint='Dùng pandas: pd.Series(y).groupby(pd.Series(cat)).agg(["sum", "count"]). Khi apply, .map(...) rồi .fillna(prior). Out-of-fold: lặp kfold_indices (đã có sẵn), fit trên train của fold, apply lên phần validation của fold.',
  starter='''def target_encode_fit(cat, y, m):
    # Trả {'map': {giá trị: mã}, 'prior': trung bình y}. mã = (Σy + m·prior) / (số mẫu + m).
    raise NotImplementedError

def target_encode_apply(enc, cat):
    # Trả mảng float; giá trị chưa gặp nhận prior.
    raise NotImplementedError

def oof_target_encode(cat, y, k, seed, m):
    # Mỗi mẫu được mã hóa bằng thống kê từ k − 1 fold còn lại (kfold_indices(len(y), k, seed)).
    raise NotImplementedError
''',
  solution='''def target_encode_fit(cat, y, m):
    y = pd.Series(np.asarray(y, dtype=float))
    stats = y.groupby(pd.Series(cat)).agg(['sum', 'count'])
    prior = float(y.mean())
    return {'map': ((stats['sum'] + m * prior) / (stats['count'] + m)).to_dict(), 'prior': prior}

def target_encode_apply(enc, cat):
    return pd.Series(cat).map(enc['map']).fillna(enc['prior']).to_numpy(dtype=float)

def oof_target_encode(cat, y, k, seed, m):
    cat, y = np.asarray(cat, dtype=object), np.asarray(y, dtype=float)
    out = np.empty(len(y))
    for tr, va in kfold_indices(len(y), k, seed):
        out[va] = target_encode_apply(target_encode_fit(cat[tr], y[tr], m), cat[va])
    return out
''',
  test='''_report('d20-target',
    _check('Ví dụ tính tay với m = 2', lambda: ((lambda e: np.isclose(e['prior'], 2 / 3) and np.allclose([e['map'][c] for c in 'abc'], [7 / 12, 2 / 3, 7 / 9]))(target_encode_fit(np.array(list('aabbbc')), np.array([1, 0, 1, 1, 0, 1]), 2)), 'a = 0,583; b = 0,667; c = 0,778.')),
    _check('Giá trị chưa gặp nhận prior', lambda: (np.allclose(target_encode_apply(target_encode_fit(np.array(list('aab')), np.array([1, 0, 1]), 1), np.array(['a', 'z'])), [(1 + 2 / 3) / 3, 2 / 3]), '"z" không có trong train.')),
    _check('Out-of-fold khớp cách tính tham chiếu', lambda: (np.allclose(oof_target_encode(_cats, _ysig, 5, 0, 10), _ref_oof(_cats, _ysig, 5, 0, 10)), '5 fold, seed 0, m = 10.')),
    _check('Không rò rỉ trên nhãn ngẫu nhiên, vẫn học được khi có tín hiệu', lambda: ((lambda a, b: a < .62 and b > .65)(roc_auc_score(_ynoise, oof_target_encode(_ids, _ynoise, 5, 0, 0)), roc_auc_score(_ysig, oof_target_encode(_cats, _ysig, 5, 0, 10))), 'Nhãn ngẫu nhiên: AUC phải gần 0,5 (bản trong-mẫu cho 0,82).')),
)
''')]))

# ───────────────────────── Ngày 21 ─────────────────────────
L.append(lesson(
 id=21, title='Dự án P3: Digit Recognizer', projectTitle='Digit Recognizer',
 goals=['Ghép PCA và KNN vào một Pipeline để PCA chỉ fit trên fold train', 'Chọn số thành phần và k bằng stratified 5-fold có seed', 'Xuất submission đúng định dạng ImageId, Label của Kaggle'],
 reads=['LeCun, Bottou, Bengio & Haffner (1998), Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE 86(11), 2278–2324 (bộ MNIST)', 'Alpaydin & Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository (bộ load_digits)', 'ISLR (bản 2), §12.2 (ôn) và §5.1.3'],
 project='Tải Digit Recognizer. Chạy select_config trên lưới n_components ∈ {20, 40, 80} × k ∈ {1, 3, 5} với 5000 ảnh train đầu (để chạy nhanh), rồi export_digits(make_digits_pipeline, nc, k) để ghi digits_submission.csv và tự nộp trên Kaggle. Báo cáo: bảng cấu hình → accuracy CV, điểm Kaggle, ma trận nhầm lẫn và 8 ảnh bị dự đoán sai. Tuần 4 sẽ so kết quả này với CNN bằng PyTorch.',
 questions=['Nếu fit PCA trên toàn bộ dữ liệu rồi mới cross-validation thì sai ở đâu?', 'Vì sao PCA giúp KNN?', 'Ảnh Kaggle 28×28 giá trị 0–255 còn load_digits 8×8 giá trị 0–16: pipeline có cần sửa không?'],
 answers=['Các thành phần chính đã "thấy" ảnh của fold validation; với PCA không dùng nhãn nên rò rỉ nhỏ, nhưng nguyên tắc vẫn là mọi bước fit phải nằm trong fold.', 'Bỏ các hướng có phương sai nhỏ (thường là nhiễu), làm khoảng cách có nghĩa hơn và tính nhanh hơn nhiều so với 784 chiều.', 'Không cần đổi cấu trúc; chỉ chọn lại n_components bằng CV vì số chiều và thang đo khác nhau.'],
 steps=[{'line': 'X, y = load_digits(return_X_y=True)', 'state': 'X.shape = (1797, 64)'},
        {'line': 'pipe = Pipeline([("pca", PCA(30, random_state=0)), ("knn", KNeighborsClassifier(3))])', 'state': 'PCA 30 thành phần giữ 95,9% phương sai'},
        {'line': 'cv = StratifiedKFold(5, shuffle=True, random_state=0)', 'state': 'mỗi fold ≈ 360 ảnh'},
        {'line': 'cross_val_score(pipe, X, y, cv=cv).mean()', 'state': 'accuracy 0,9855'},
        {'line': 'select_config(make, X, y, grid, seed=0)', 'state': 'lưới (5,1) 0,899 · (10,3) 0,976 · (20,1) 0,987 · (30,1) 0,989'},
        {'line': 'pd.DataFrame({"ImageId": 1..n, "Label": pred}).to_csv(...)', 'state': 'định dạng nộp của Digit Recognizer'}],
 setup=BASE + DIGITS + '''from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted
_grid21 = [(5, 1), (10, 3), (20, 1), (30, 1)]
def _unfitted(pipe):
    try:
        check_is_fitted(pipe)
        return False
    except NotFittedError:
        return True
def _ref_cv21(make, nc, k, seed):
    return float(cross_val_score(make(nc, k), X_dg, y_dg, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean())
def export_digits(make, n_components, k, limit=None):
    folder = DATA / 'kaggle/digit-recognizer'
    train, test = pd.read_csv(folder / 'train.csv', nrows=limit), pd.read_csv(folder / 'test.csv')
    pipe = make(n_components, k).fit(train.drop(columns='label').to_numpy(float), train.label.to_numpy())
    out = pd.DataFrame({'ImageId': np.arange(1, len(test) + 1), 'Label': pipe.predict(test.to_numpy(float))})
    out.to_csv('digits_submission.csv', index=False)
    return out
''',
 exercises=[dict(
  id='d21-digits', title='Pipeline PCA + KNN và chọn cấu hình',
  hint='Pipeline([("pca", PCA(n_components, random_state=0)), ("knn", KNeighborsClassifier(n_neighbors=k))]). cv_digits dùng StratifiedKFold(5, shuffle=True, random_state=seed). select_config trả cặp có điểm cao nhất; hòa thì cặp đứng trước trong lưới.',
  starter='''from sklearn.pipeline import Pipeline

def make_digits_pipeline(n_components, k):
    # Trả Pipeline CHƯA fit.
    raise NotImplementedError

def cv_digits(make, X, y, n_components, k, seed):
    raise NotImplementedError

def select_config(make, X, y, grid, seed):
    # grid: list các cặp (n_components, k). Trả cặp tốt nhất.
    raise NotImplementedError
''',
  solution='''from sklearn.pipeline import Pipeline

def make_digits_pipeline(n_components, k):
    return Pipeline([('pca', PCA(n_components=n_components, random_state=0)),
                     ('knn', KNeighborsClassifier(n_neighbors=k))])

def cv_digits(make, X, y, n_components, k, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(make(n_components, k), X, y, cv=cv).mean())

def select_config(make, X, y, grid, seed):
    scores = [cv_digits(make, X, y, nc, k, seed) for nc, k in grid]
    return grid[int(np.argmax(scores))]
''',
  test='''_report('d21-digits',
    _check('Pipeline chưa fit, có PCA rồi KNN đúng tham số', lambda: ((lambda p: _unfitted(p) and isinstance(p.steps[0][1], PCA) and p.steps[0][1].n_components == 12 and isinstance(p.steps[-1][1], KNeighborsClassifier) and p.steps[-1][1].n_neighbors == 4)(make_digits_pipeline(12, 4)), 'PCA phải nằm trong pipeline để fit lại trong từng fold.')),
    _check('cv_digits khớp stratified 5-fold tham chiếu', lambda: (np.isclose(cv_digits(make_digits_pipeline, X_dg, y_dg, 20, 3, 1), _ref_cv21(make_digits_pipeline, 20, 3, 1)), 'n_components = 20, k = 3, seed = 1.')),
    _check('select_config chọn đúng cặp tốt nhất', lambda: (select_config(make_digits_pipeline, X_dg, y_dg, _grid21, 0) == _grid21[int(np.argmax([_ref_cv21(make_digits_pipeline, nc, k, 0) for nc, k in _grid21]))], 'Lưới (5,1), (10,3), (20,1), (30,1).')),
)
''')]))

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'src/track-py/week3.json'
out.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding='utf-8')
print('ghi', out, len(L), 'bài,', sum(len(l['exercises']) for l in L), 'bài tập')
