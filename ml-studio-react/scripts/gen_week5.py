"""Sinh src/track-py/week5.json (Ngày 29–30): dự án tổng kết.

    python scripts/gen_week5.py
Sau đó: python ../linear-regression-lab/py30/build_notebooks.py và verify_py30.py --days 29-30.
Bài tập dùng Real estate valuation (UCI, py30/data/real_estate.csv) để chạy không cần mạng; dự án thật làm trên Kaggle
(mặc định House Prices, notebook py30/cloud/Ngay_29_30_Capstone_*). Các số trong "Ví dụ tính từng dòng" đã được tính lại bằng code."""
import json, sys
from pathlib import Path

BASE = '''import hashlib, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = next((p for p in (Path('data'), Path('../data')) if (p / 'real_estate.csv').is_file()), None)
if DATA is None:
    raise FileNotFoundError('Mở notebook từ thư mục py30 hoặc py30/bai_lam để thấy data/real_estate.csv.')
# Real estate valuation (Yeh, 2018; UCI, CC BY 4.0): 414 giao dịch, giá theo 10.000 TWD/ping.
real_estate = pd.read_csv(DATA / 'real_estate.csv')
TARGET = 'price_10k_ntd_per_ping'

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

def _cv_rmse_ref(make_model, X, y, seed):
    out = []
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_model().fit(X.iloc[tr], y[tr]); out.append(np.sqrt(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2)))
    return float(np.mean(out))
'''

DEV = '''# Tập phát triển cố định (80%) — 20% còn lại là holdout đã khóa ở Ngày 29.
_order = np.random.default_rng(0).permutation(len(real_estate))
_hold = np.sort(_order[:83]); _dev = np.sort(_order[83:])
dev = real_estate.iloc[_dev].reset_index(drop=True)
X_dev, y_dev = dev.drop(columns=TARGET), dev[TARGET].to_numpy()
'''

def lesson(**k):
    return k

L = []

# ───────────────────────── Ngày 29 ─────────────────────────
L.append(lesson(
 id=29, title='Dự án tổng kết: khóa holdout và baseline', projectTitle='House Prices (capstone)',
 goals=['Tách và khóa một tập holdout trước khi nhìn dữ liệu, ghi lại dấu vân tay của nó', 'Đo bằng đúng thước đo của cuộc thi (House Prices: RMSE trên log giá)', 'Dựng thang baseline: dự đoán trung bình → tuyến tính một biến → tuyến tính đầy đủ → mô hình mạnh'],
 reads=['Ng (2018), Machine Learning Yearning, các chương về tập dev/test và chọn thước đo một con số', 'Kaufman, Rosset, Perlich & Stitelman (2012), Leakage in Data Mining: Formulation, Detection, and Avoidance, ACM TKDD 6(4)', 'Dwork et al. (2015), The Reusable Holdout: Preserving Validity in Adaptive Data Analysis, Science 349(6248), 636–638', 'Kaggle, House Prices: Advanced Regression Techniques, mục Evaluation'],
 project='Chọn cuộc thi Kaggle cho dự án tổng kết (mặc định House Prices; không cần liên quan luận văn). Mở notebook py30/cloud/Ngay_29_30_Capstone (Kaggle hoặc Colab). Khóa 20% train làm holdout bằng lock_holdout, ghi fingerprint vào README dự án và hứa không đánh giá trên đó cho tới cuối Ngày 30. Báo cáo thang baseline bằng RMSLE 5-fold trên phần dev, rồi nộp submission của baseline tốt nhất để có mốc điểm Kaggle.',
 questions=['Vì sao phải khóa holdout trước khi khám phá dữ liệu?', 'Điểm trên bảng xếp hạng công khai của Kaggle có thay được holdout của bạn không?', 'Vì sao cần baseline "dự đoán trung bình"?'],
 answers=['Mỗi quyết định dựa trên dữ liệu (chọn cột, xử lý ngoại lai, chọn mô hình) là một lần "nhìn"; nếu holdout đã được nhìn, ước lượng cuối cùng sẽ lạc quan.', 'Không hẳn: bảng công khai chỉ dùng một phần test và bị dùng lại nhiều lần nên cũng quá khớp; giữ holdout riêng để có ước lượng không bị nhìn.', 'Nó cho mức điểm của mô hình không học gì; mọi cải tiến phải được so với mốc này và với baseline tuyến tính.'],
 steps=[{'line': 'order = np.random.default_rng(0).permutation(414)', 'state': 'Real estate: 414 giao dịch'},
        {'line': 'holdout, dev = sort(order[:83]), sort(order[83:])', 'state': '83 holdout (20%), 331 dev; holdout bắt đầu [2, 5, 18, 19, 27, 36, …]'},
        {'line': 'fingerprint(holdout)', 'state': 'sha256 của chỉ số đã sắp xếp → 167ee3b5c648a47e (ghi vào README)'},
        {'line': 'cv_rmse(DummyRegressor, X_dev, y_dev, seed=0)', 'state': 'dự đoán trung bình: RMSE 13,82 (≈ độ lệch chuẩn của y, 13,80)'},
        {'line': 'cv_rmse(LinearRegression, log1p(mrt_distance), ...)', 'state': 'tuyến tính 1 biến: 9,22; tuyến tính 6 biến (chuẩn hóa): 9,04'},
        {'line': 'cv_rmse(GradientBoostingRegressor(random_state=0), ...)', 'state': 'gradient boosting: 7,81 — mọi cải tiến sau đều so với các mốc này'}],
 setup=BASE + '''from sklearn.metrics import mean_squared_log_error
def _lock_ref(n, frac, seed):
    o = np.random.default_rng(seed).permutation(n); k = int(round(frac * n)); return np.sort(o[k:]), np.sort(o[:k])
''',
 exercises=[dict(
  id='d29-holdout', title='Khóa holdout và dấu vân tay',
  hint='np.random.default_rng(seed).permutation(n); k = int(round(frac·n)); holdout = k chỉ số đầu, dev = phần còn lại, cả hai sắp xếp tăng dần. fingerprint: hashlib.sha256 trên np.asarray(sorted(idx), dtype=np.int64).tobytes(), lấy 16 ký tự hex đầu.',
  starter='''def lock_holdout(n, frac, seed):
    # Trả (dev_idx, holdout_idx), mỗi mảng đã sắp xếp tăng dần.
    raise NotImplementedError

def fingerprint(idx):
    # Chuỗi 16 ký tự hex, không phụ thuộc thứ tự phần tử trong idx.
    raise NotImplementedError
''',
  solution='''def lock_holdout(n, frac, seed):
    order = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(order[k:]), np.sort(order[:k])

def fingerprint(idx):
    data = np.asarray(sorted(int(i) for i in idx), dtype=np.int64).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]
''',
  test='''_report('d29-holdout',
    _check('Chia đúng cỡ, rời nhau, phủ đủ', lambda: ((lambda d, h: len(h) == 83 and len(d) == 331 and np.array_equal(np.sort(np.r_[d, h]), np.arange(414)) and np.all(np.diff(h) > 0))(*lock_holdout(414, 0.2, 0)), '414 dòng, 20% holdout.')),
    _check('Lặp lại được theo seed và có xáo trộn', lambda: (all(np.array_equal(a, b) for a, b in zip(lock_holdout(414, .2, 0), _lock_ref(414, .2, 0))) and not np.array_equal(lock_holdout(414, .2, 0)[1], lock_holdout(414, .2, 1)[1]), 'Khớp phép chia tham chiếu với seed 0.')),
    _check('Dấu vân tay ổn định và nhạy với thay đổi', lambda: ((lambda h: fingerprint(h) == '167ee3b5c648a47e' and fingerprint(h[::-1]) == fingerprint(h) and fingerprint(np.r_[h[1:], 0]) != fingerprint(h))(lock_holdout(414, .2, 0)[1]), 'Holdout seed 0 có dấu vân tay 167ee3b5c648a47e.')),
)
'''), dict(
  id='d29-baseline', title='Thước đo cuộc thi và RMSE cross-validation',
  hint='rmsle = √mean((log1p(p) − log1p(y))²), chặn p ≥ 0 trước khi lấy log. cv_rmse: với mỗi fold của KFold(5, shuffle=True, random_state=seed), gọi make_model() để có mô hình MỚI, fit trên fold train, tính RMSE trên fold validation, rồi lấy trung bình 5 fold.',
  starter='''def rmse(y, p):
    raise NotImplementedError

def rmsle(y, p):
    # Thước đo của House Prices: RMSE giữa log(1 + dự đoán) và log(1 + giá thật).
    raise NotImplementedError

def cv_rmse(make_model, X, y, seed):
    # X: DataFrame, y: mảng. make_model() trả mô hình chưa fit.
    raise NotImplementedError
''',
  solution='''def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))

def rmsle(y, p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(np.asarray(y, dtype=float))) ** 2)))

def cv_rmse(make_model, X, y, seed):
    scores = []
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        model = make_model().fit(X.iloc[tr], y[tr])
        scores.append(rmse(y[va], model.predict(X.iloc[va])))
    return float(np.mean(scores))
''',
  test=DEV.replace("'''", '') + '''_report('d29-baseline',
    _check('rmsle khớp scikit-learn và chịu được dự đoán âm', lambda: (np.isclose(rmsle([200000, 150000, 310000], [210000, 140000, 300000]), np.sqrt(mean_squared_log_error([200000, 150000, 310000], [210000, 140000, 300000]))) and np.isfinite(rmsle([10.], [-5.])), 'Chặn dự đoán ở 0.')),
    _check('cv_rmse khớp cross_val_score (neg RMSE)', lambda: (np.isclose(cv_rmse(LinearRegression, X_dev, y_dev, 0), -cross_val_score(LinearRegression(), X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=0), scoring='neg_root_mean_squared_error').mean()), 'Tuyến tính 6 biến trên tập dev.')),
    _check('Baseline trung bình ≈ độ lệch chuẩn của y', lambda: ((lambda r: (abs(r - y_dev.std()) < .5, f'RMSE {r:.2f}, std {y_dev.std():.2f}.'))(cv_rmse(DummyRegressor, X_dev, y_dev, 0))[0], 'DummyRegressor dự đoán trung bình của fold train.')),
)
''')]))

# ───────────────────────── Ngày 30 ─────────────────────────
L.append(lesson(
 id=30, title='Dự án tổng kết: ablation, phân tích lỗi và submission', projectTitle='House Prices (capstone)',
 goals=['Đo đóng góp của từng nhóm đặc trưng bằng ablation (bỏ từng nhóm, đo lại CV)', 'Tìm chỗ mô hình sai nhiều nhất theo lát cắt dữ liệu', 'Kiểm tra file submission trước khi nộp, rồi mới mở holdout đúng một lần'],
 reads=['Lipton & Steinhardt (2019), Troubling Trends in Machine Learning Scholarship, ACM Queue 17(1) (mục về ablation)', 'Ng (2018), Machine Learning Yearning, các chương về phân tích lỗi (error analysis)', 'Sculley et al. (2015), Hidden Technical Debt in Machine Learning Systems, NeurIPS 28', 'Recht, Roelofs, Schmidt & Shankar (2019), Do ImageNet Classifiers Generalize to ImageNet?, ICML'],
 project='Trên House Prices: (1) bảng ablation bằng RMSLE 5-fold với các nhóm đặc trưng bạn tạo (diện tích, chất lượng, vị trí, năm xây...), (2) bảng lỗi theo lát cắt (Neighborhood, OverallQual) và 10 nhà sai nhiều nhất kèm giả thuyết, (3) chạy check_submission rồi nộp bản cuối. Cuối cùng mở holdout đúng MỘT lần, báo cáo RMSLE holdout cạnh điểm CV và điểm Kaggle. Viết báo cáo 1 trang: bài toán, baseline, mô hình cuối, ablation, phân tích lỗi, hạn chế.',
 questions=['Bỏ một nhóm mà điểm gần như không đổi thì có nghĩa nhóm đó vô dụng không?', 'Vì sao phân tích lỗi theo lát cắt hữu ích hơn chỉ nhìn một con số tổng?', 'Điểm holdout thấp hơn CV một chút thì nên làm gì?'],
 answers=['Không chắc: thông tin của nhóm đó có thể trùng với nhóm khác (ví dụ tọa độ và khoảng cách tới ga). Ablation đo đóng góp RIÊNG khi các nhóm còn lại vẫn có mặt.', 'Nó chỉ ra nhóm dữ liệu nào bị sai có hệ thống, gợi ý đặc trưng cần thêm hoặc dữ liệu cần thu, thay vì đoán mò.', 'Ghi nhận và báo cáo trung thực; không quay lại chỉnh mô hình theo holdout, vì khi đó holdout không còn là ước lượng chưa bị nhìn.'],
 steps=[{'line': 'full = cv_rmse(GradientBoosting, X_dev, y_dev, 0)', 'state': 'RMSE đầy đủ 6 cột: 7,81'},
        {'line': 'bỏ nhóm "vị trí" (latitude, longitude)', 'state': '7,99 → tăng 0,18: phần lớn thông tin vị trí đã có trong khoảng cách tới ga MRT'},
        {'line': 'bỏ "tiện ích" (mrt_distance_m, stores) / "tuổi nhà" / "thời gian"', 'state': 'tăng 0,40 / 0,42 / 0,41'},
        {'line': 'error_slices(..., by="mrt_distance_m", q=4)', 'state': 'MAE theo tứ phân vị khoảng cách: 6,61 · 5,39 · 4,77 · 3,68 (gần ga sai nhiều nhất)'},
        {'line': 'worst_cases(y_dev, pred, k=3)', 'state': 'dòng 219: giá 117,5, dự đoán 45,5 (một giao dịch bất thường)'},
        {'line': 'check_submission(sub, test_ids)', 'state': '[] nghĩa là hợp lệ; mỗi lỗi trả về một câu mô tả'}],
 setup=BASE + DEV + '''from sklearn.model_selection import cross_val_predict
GROUPS = {'vị trí': ['latitude', 'longitude'], 'tiện ích': ['mrt_distance_m', 'stores'], 'tuổi nhà': ['age_years'], 'thời gian': ['transaction_date']}
_gb = lambda: GradientBoostingRegressor(random_state=0)
cv_rmse = globals().get('cv_rmse') or _cv_rmse_ref
_pred30 = cross_val_predict(_gb(), X_dev, y_dev, cv=KFold(5, shuffle=True, random_state=0))
_ids30 = pd.Series(np.arange(1461, 1471), name='Id')
_good30 = pd.DataFrame({'Id': _ids30, 'SalePrice': np.linspace(120000, 300000, 10)})
''',
 exercises=[dict(
  id='d30-ablation', title='Bảng ablation theo nhóm đặc trưng',
  hint='full = cv_rmse(make_model, X, y, seed). Với mỗi nhóm: X.drop(columns=cột của nhóm), đo lại, delta = rmse_khi_bỏ − full (dương nghĩa là nhóm có ích). Trả DataFrame cột ["nhóm", "rmse", "delta"], sắp delta giảm dần.',
  starter='''def ablation(make_model, X, y, groups, seed):
    # groups: dict tên nhóm → list cột. Dùng cv_rmse (đã có sẵn).
    raise NotImplementedError
''',
  solution='''def ablation(make_model, X, y, groups, seed):
    full = cv_rmse(make_model, X, y, seed)
    rows = []
    for name, cols in groups.items():
        score = cv_rmse(make_model, X.drop(columns=cols), y, seed)
        rows.append({'nhóm': name, 'rmse': score, 'delta': score - full})
    return pd.DataFrame(rows).sort_values('delta', ascending=False).reset_index(drop=True)
''',
  test='''_c30 = {}
def _abl():
    if 't' not in _c30:
        _c30['t'] = ablation(_gb, X_dev, y_dev, GROUPS, 0)
    return _c30['t']
def _abl_ref():
    full = _cv_rmse_ref(_gb, X_dev, y_dev, 0)
    return {g: _cv_rmse_ref(_gb, X_dev.drop(columns=c), y_dev, 0) - full for g, c in GROUPS.items()}
_report('d30-ablation',
    _check('Đúng cột và một dòng cho mỗi nhóm', lambda: ((lambda t: list(t.columns) == ['nhóm', 'rmse', 'delta'] and sorted(t['nhóm']) == sorted(GROUPS))(_abl()), '4 nhóm.')),
    _check('delta khớp cách tính tham chiếu', lambda: ((lambda t, r: all(np.isclose(d, r[g]) for g, d in zip(t['nhóm'], t['delta'])))(_abl(), _abl_ref()), 'delta = RMSE khi bỏ nhóm − RMSE đầy đủ.')),
    _check('Sắp xếp theo delta giảm dần', lambda: (bool(np.all(np.diff(_abl()['delta'].to_numpy()) <= 0)), 'Nhóm quan trọng nhất ở trên cùng.')),
)
'''), dict(
  id='d30-errors', title='Lỗi theo lát cắt và các ca sai nhiều nhất',
  hint='Nếu q được cho, chia cột bằng pd.qcut(df[by], q); ngược lại nhóm theo giá trị của cột. Với mỗi nhóm: n, mae = mean|p − y|, bias = mean(p − y). Sắp mae giảm dần. worst_cases: np.argsort(−|p − y|)[:k].',
  starter='''def error_slices(df, y_true, y_pred, by, q=None):
    # Trả DataFrame index = nhóm, cột ['n', 'mae', 'bias'], sắp theo mae giảm dần.
    raise NotImplementedError

def worst_cases(y_true, y_pred, k):
    # Chỉ số của k mẫu có |sai số| lớn nhất, lớn nhất đứng đầu.
    raise NotImplementedError
''',
  solution='''def error_slices(df, y_true, y_pred, by, q=None):
    key = pd.qcut(df[by], q) if q else df[by]
    err = np.asarray(y_pred) - np.asarray(y_true)
    frame = pd.DataFrame({'g': key.to_numpy(), 'abs': np.abs(err), 'err': err})
    out = frame.groupby('g', observed=True).agg(n=('abs', 'size'), mae=('abs', 'mean'), bias=('err', 'mean'))
    return out.sort_values('mae', ascending=False)

def worst_cases(y_true, y_pred, k):
    return np.argsort(-np.abs(np.asarray(y_pred) - np.asarray(y_true)), kind='stable')[:k]
''',
  test='''def _slices_ref(by, q):
    key = pd.qcut(X_dev[by], q) if q else X_dev[by]
    e = _pred30 - y_dev
    return pd.DataFrame({'g': key.to_numpy(), 'a': np.abs(e), 'e': e}).groupby('g', observed=True).agg(n=('a', 'size'), mae=('a', 'mean'), bias=('e', 'mean')).sort_values('mae', ascending=False)
def _same(a, b):
    return list(a.columns) == ['n', 'mae', 'bias'] and len(a) == len(b) and np.allclose(a[['n', 'mae', 'bias']].to_numpy(float), b.to_numpy(float))
_report('d30-errors',
    _check('Lát cắt theo tứ phân vị khoảng cách MRT', lambda: (_same(error_slices(X_dev, y_dev, _pred30, 'mrt_distance_m', 4), _slices_ref('mrt_distance_m', 4)), 'pd.qcut thành 4 nhóm.')),
    _check('Lát cắt theo giá trị rời rạc (số cửa hàng)', lambda: (_same(error_slices(X_dev, y_dev, _pred30, 'stores'), _slices_ref('stores', None)), 'Mỗi giá trị stores là một nhóm.')),
    _check('Bias có dấu đúng (dự đoán − thật)', lambda: (np.isclose(error_slices(pd.DataFrame({'g': ['a', 'a']}), np.array([10., 10.]), np.array([12., 14.]), 'g').loc['a', 'bias'], 3.0), 'Dự đoán cao hơn thật thì bias dương.')),
    _check('worst_cases trả đúng thứ tự', lambda: (list(worst_cases(y_dev, _pred30, 3)) == list(np.argsort(-np.abs(_pred30 - y_dev), kind='stable')[:3]), 'Ca sai nhiều nhất đứng đầu.')),
)
'''), dict(
  id='d30-submission', title='Kiểm tra file submission trước khi nộp',
  hint='Kiểm tra lần lượt và thêm một câu mô tả vào danh sách cho mỗi lỗi: đúng hai cột [id_col, target_col] theo thứ tự; số dòng bằng số id test; cột id trùng khớp từng dòng với test_ids; không có NaN; cột mục tiêu là số và dương.',
  starter='''def check_submission(sub, test_ids, id_col='Id', target_col='SalePrice'):
    # Trả list các chuỗi mô tả lỗi; list rỗng nghĩa là hợp lệ.
    raise NotImplementedError
''',
  solution='''def check_submission(sub, test_ids, id_col='Id', target_col='SalePrice'):
    problems = []
    if list(sub.columns) != [id_col, target_col]:
        problems.append(f'Cột phải đúng là [{id_col}, {target_col}], đang có {list(sub.columns)}')
        return problems
    if len(sub) != len(test_ids):
        problems.append(f'Cần {len(test_ids)} dòng, đang có {len(sub)}')
    elif not np.array_equal(sub[id_col].to_numpy(), np.asarray(test_ids)):
        problems.append('Cột id không khớp từng dòng với file test')
    if sub[target_col].isna().any():
        problems.append('Có giá trị NaN trong cột dự đoán')
    if not pd.api.types.is_numeric_dtype(sub[target_col]):
        problems.append('Cột dự đoán không phải số')
    elif (sub[target_col].dropna() <= 0).any():
        problems.append('Có dự đoán ≤ 0')
    return problems
''',
  test='''_bad30 = {
    'thiếu cột': _good30[['Id']],
    'sai tên cột': _good30.rename(columns={'SalePrice': 'price'}),
    'thiếu dòng': _good30.iloc[:9],
    'sai thứ tự id': _good30.iloc[::-1].reset_index(drop=True),
    'có NaN': _good30.assign(SalePrice=[np.nan] + list(_good30.SalePrice[1:])),
    'giá âm': _good30.assign(SalePrice=[-1.] + list(_good30.SalePrice[1:])),
    'giá là chữ': _good30.assign(SalePrice=_good30.SalePrice.astype(str)),
}
_report('d30-submission',
    _check('File đúng thì không báo lỗi', lambda: (check_submission(_good30, _ids30) == [], '10 dòng, Id 1461–1470.')),
    _check('Bắt được cả 7 kiểu lỗi', lambda: ((lambda miss: (not miss, 'Bỏ sót: ' + ', '.join(miss) if miss else 'Đủ 7.'))([k for k, v in _bad30.items() if not check_submission(v, _ids30)]))),
)
''')]))

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'src/track-py/week5.json'
out.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding='utf-8')
print('ghi', out, len(L), 'bài,', sum(len(l['exercises']) for l in L), 'bài tập')
