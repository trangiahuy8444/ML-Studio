"""Sinh dữ liệu cho mô phỏng tuần 3 (chạy: python scripts/gen_week3_data.py).

iris2d.json  : Iris (Fisher, 1936) từ scikit-learn, hai cột petal length/width (cm) và nhãn loài.
digits.json  : load_digits (UCI Optical Recognition of Handwritten Digits). PCA tính bằng np.linalg.svd trên
               toàn bộ 1797 ảnh; 400 ảnh mẫu (seed 21) kèm tọa độ trên 30 thành phần đầu; bảng accuracy
               stratified 5-fold (shuffle, seed 0) của Pipeline(PCA(nc, random_state=0), KNN(k)) do scikit-learn tính."""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

out = Path(__file__).resolve().parents[1] / 'src/track-py/data'
iris = load_iris()
(out / 'iris2d.json').write_text(json.dumps({
    'source': 'Iris (Fisher, 1936), bản trong scikit-learn',
    'species': list(iris.target_names),
    'points': [[round(float(a), 2), round(float(b), 2), int(c)] for (a, b), c in zip(iris.data[:, 2:4], iris.target)]}))

X, y = load_digits(return_X_y=True)
mean = X.mean(0)
_, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
var = S ** 2 / (len(X) - 1)
idx = np.sort(np.random.default_rng(21).choice(len(X), 400, replace=False))
Z = (X[idx] - mean) @ Vt[:30].T
grid = []
for nc in (2, 5, 10, 20, 30, 40):
    for k in (1, 3, 5, 9):
        pipe = Pipeline([('pca', PCA(nc, random_state=0)), ('knn', KNeighborsClassifier(k))])
        grid.append([nc, k, round(float(cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()), 4)])
(out / 'digits.json').write_text(json.dumps({
    'source': 'Optical Recognition of Handwritten Digits (Alpaydin & Kaynak, 1998), bản load_digits của scikit-learn',
    'mean': [round(float(v), 4) for v in mean],
    'components': [[round(float(v), 4) for v in row] for row in Vt[:30]],
    'ratio': [round(float(v), 6) for v in var / var.sum()],
    'samples': {'index': idx.tolist(), 'label': y[idx].tolist(), 'pixels': X[idx].astype(int).tolist(),
                'z': [[round(float(v), 2) for v in row] for row in Z]},
    'cv': grid}))
print('iris2d', (out / 'iris2d.json').stat().st_size, 'digits', (out / 'digits.json').stat().st_size)
