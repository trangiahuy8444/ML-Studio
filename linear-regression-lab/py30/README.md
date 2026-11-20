# Python → ML 30 ngày

Đã có tuần 1 (Ngày 1–7, hồi quy và tối ưu, dự án House Prices), tuần 2 (Ngày 8–14, phân loại và đánh giá, dự án Titanic) tuần 3 (Ngày 15–21, cây, tổ hợp và học không nhãn, dự án Digit Recognizer) tuần 4 (Ngày 22–28, mạng nơ-ron và PyTorch, dự án CNN nhiều seed) và tuần 5 (Ngày 29–30, dự án tổng kết: khóa holdout, baseline, ablation, phân tích lỗi, kiểm tra submission). Ngày 22–23 chỉ cần NumPy; Ngày 24–28 cần PyTorch trong kernel (`python -m pip install -r requirements-py30.txt`), chạy được trên CPU. Phần dự án trên Fashion-MNIST, IMDB dùng notebook trong `py30/cloud/` trên Kaggle hoặc Colab. Mỗi buổi 180 phút.

1. Mở `Mo_Jupyter_Studio.command` ở thư mục cha và giữ Terminal mở. Mở địa chỉ launcher in ra, chọn **Python → ML 30 ngày**. Trang tổng quan có **Bản đồ thuật toán**: mỗi thẻ chạy thuật toán trên dữ liệu minh họa nhỏ, bấm vào để mở buổi học.
2. Trong buổi học, chọn **Kết nối**, viết code, rồi bấm **Chạy và kiểm tra** hoặc Shift+Enter. Trước khi chấm, các hàm của bài bị xóa khỏi kernel để kết quả phản ánh đúng code đang có trong ô. Bài mẫu `Ngay_XX.ipynb` không chứa lời giải; nút lưu tạo bản làm riêng trong `bai_lam/`.
3. Dữ liệu chạy ngay không cần Internet: UCI Real Estate (`data/real_estate.csv`, tuần 1) và Breast Cancer Wisconsin (có sẵn trong scikit-learn, tuần 2). Dữ liệu Kaggle tải bằng `python py30/download_data.py house|titanic|fraud|digits|imdb` sau khi chấp nhận luật cuộc thi; cài `requirements-py30.txt` trước.
4. Ngày 7: `export_house_prices(run_pipeline)` ghi `submission.csv`. Ngày 14: `export_titanic(make_titanic_pipeline, C)` ghi `titanic_submission.csv`. Ngày 21: `export_digits(make_digits_pipeline, nc, k)` ghi `digits_submission.csv` (cần `python py30/download_data.py digits`). Kiểm tra cột và số dòng trước khi tự nộp.
5. Thư mục `cloud/` có bản xuất Kaggle/Colab cho các buổi cần GPU (CNN, Transformer, capstone).

## Cho người soạn nội dung

- Nội dung bài nằm ở `ml-studio-react/src/track-py/week1.json`, `week2.json` (tuần 2 sinh từ `ml-studio-react/scripts/gen_week2.py`).
- `python py30/build_notebooks.py` sinh lại notebook mẫu từ JSON; `--check` chỉ báo notebook lệch.
- `python py30/verify_py30.py [--days 8-14]` chạy mỗi bài trong kernel mới: khung chưa làm phải chưa đạt, lời giải tham khảo phải đạt, mọi lời giải sai trong `wrong_cases.py` phải bị từ chối. Thêm bài mới thì phải thêm lời giải sai.

Tệp HTML mở trực tiếp hỗ trợ lý thuyết và mô phỏng. Ô chạy Python cần địa chỉ launcher của Jupyter cục bộ. Dữ liệu Kaggle, bài đang làm và token tài khoản chỉ nằm trên máy học viên.
