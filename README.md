# ML Studio · Python → ML 30 ngày

Web app học Machine Learning cho người đã vững Python: trực quan hóa thuật toán, mô phỏng code từng dòng, bài tập chấm tự động chạy trên kernel Jupyter ở máy bạn, dự án Kaggle, và PyTorch ở tuần 4.

- **Bản web (GitHub Pages):** https://trangiahuy8444.github.io/ML-Studio/ (sau khi bật Pages, xem bên dưới).
- **Đã có:** đủ 30 ngày — tuần 1 hồi quy và tối ưu, tuần 2 phân loại và đánh giá, tuần 3 cây, tổ hợp và học không nhãn, tuần 4 mạng nơ-ron và PyTorch, tuần 5 dự án tổng kết trên Kaggle.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `ml-studio-react/` | Mã nguồn web (React 19 + TypeScript + Vite). Nội dung lộ trình ở `src/track-py/week*.json`, sinh từ `scripts/gen_week*.py`. |
| `linear-regression-lab/` | Launcher Jupyter cục bộ (`start_jupyter_studio.py`, `Mo_Jupyter_Studio.command`) và notebook của khóa học. |
| `linear-regression-lab/py30/` | Notebook bài tập `Ngay_01–30.ipynb`, bộ kiểm tra `verify_py30.py`, lời giải sai cố ý `wrong_cases.py`, notebook GPU cho Kaggle/Colab trong `cloud/`. |

## Bật GitHub Pages (chỉ làm một lần)

Vào **Settings → Pages → Build and deployment → Source** và chọn **GitHub Actions**. Mỗi lần push lên `main`, workflow `.github/workflows/pages.yml` sẽ chạy test, build và đăng trang.

Trên GitHub Pages, toàn bộ bài học, mô phỏng, **6 chương học**, **8 dự án**, **notebook thực hành** và **lộ trình Python → ML 30 ngày** đều có thể chạy và chấm bài Python trực tiếp ngay trên trình duyệt (qua Pyodide/WebAssembly) mà không cần cài đặt gì thêm. Nếu muốn dùng Jupyter Notebook truyền thống và lưu trực tiếp tệp `.ipynb` trên máy, bạn có thể chạy qua launcher cục bộ như dưới đây.

## Chạy trên máy (có bài tập chấm tự động)

```bash
cd ml-studio-react
npm ci
npm run build
cp -R dist ../linear-regression-lab/dist
cd ../linear-regression-lab
python3 -m pip install -r requirements-py30.txt   # Ngày 24–28 cần PyTorch; chạy được trên CPU
python3 start_jupyter_studio.py                   # trên Mac có thể mở Mo_Jupyter_Studio.command
```

Mở địa chỉ launcher in ra, vào mục **Python → ML 30 ngày**, bấm **Kết nối** trong trang của từng ngày.

## Kiểm tra

```bash
cd ml-studio-react && npm test                    # 66 bài test nội dung và giao thức
cd ../linear-regression-lab
python3 py30/verify_py30.py                       # 204 trường hợp: khung chưa làm, lời giải, lời giải sai
```

## Dữ liệu và bản quyền

- Có sẵn trong repo: Real estate valuation (UCI, `py30/data/real_estate.csv`, xem `NGUON.md`); các bộ trong scikit-learn (Breast Cancer Wisconsin, Iris, Digits, Diabetes) được tải qua thư viện.
- **Không có trong repo:** dữ liệu Kaggle (House Prices, Titanic, Digit Recognizer, Credit Card Fraud, IMDB). Tự tải bằng `python py30/download_data.py <tên>` sau khi chấp nhận luật cuộc thi.

## Bảo mật

Không commit `kaggle.json`. Token Kaggle chỉ đặt ở `~/.kaggle/kaggle.json` (chmod 600) trên máy bạn, hoặc trong Colab Secrets. `.gitignore` đã chặn tệp này, dữ liệu Kaggle và thư mục bài làm `py30/bai_lam/`.
