# Cập nhật: thực hành bằng Jupyter NbClassic thật

Mở `../linear-regression-lab/Mo_Jupyter_Studio.command` khi dùng dự án trên máy; trong ZIP mở launcher ở thư mục gốc. Launcher khởi động web và Jupyter Server cục bộ với xác thực. Mỗi bài có nút mở notebook thật ở tab riêng; Python kernel cùng môi trường launcher, lưu trực tiếp `.ipynb`. Phần Pyodide cũ chỉ là tùy chọn chạy thử, không còn là notebook chính. Thư mục runtime, cấu hình và bộ nhớ đệm tách riêng theo phiên; server chỉ lắng nghe 127.0.0.1. Không tắt token hoặc XSRF.

Chỉ mở HTML không khởi động được Notebook Server. Mã cũ trong trình duyệt cần xuất `.ipynb` để chuyển sang Jupyter. Không ghi đè notebook đã sửa khi nâng cấp.

Kiểm tra bổ sung: 32 liên kết đúng tệp; từ chối đường dẫn ngoài danh sách; WebSocket gọi kernel thật trả kết quả NumPy 42; lưu, checkpoint và đọc lại kết quả.

---

# ML Studio 2 — Học AI từ dự án

Ứng dụng React + TypeScript, kiểm tra kiểu nghiêm ngặt, build bằng Vite. Tailwind CSS hỗ trợ bố cục; Plotly vẽ biểu đồ. Python chạy bằng Pyodide 0.28.2 trong Web Worker.

## Dùng ngay

Mở `ML_Studio_React.html` bằng Chrome. Tệp đã có React, dữ liệu và thư viện biểu đồ; không cần cài Node để học. Lần đầu chạy notebook cần Internet để tải Python, NumPy và Matplotlib. Bản trong bộ ZIP có tệp `Mo_ML_Studio.command` để mở nhanh trên Mac.

## Phát triển

```bash
npm ci
npm run dev
```

Mở địa chỉ Vite in ra (mặc định http://127.0.0.1:5173).

```bash
npm test
npm run build
npm run preview
```

`build` gồm kiểm tra TypeScript, build production và tạo HTML một tệp. Bản production nằm trong `dist/`. `ML_Studio_React.html` là bản mở trực tiếp, mở thẳng dự án giá nhà và có bài dự đoán nhanh ở mục Phòng hồi quy.

## Cấu trúc

- `src/App.tsx`: điều hướng, tiến độ, lộ trình, bài học, tìm bài thực hành và học liệu.
- `src/components/HousingProject.tsx`: dự án nhà đất React, tham số/lần học là state, biểu đồ cập nhật theo state.
- `src/components/Notebook.tsx`: giao diện kiểu Jupyter Classic, chọn/sửa/thêm/xóa/di chuyển/chuyển loại ô, chạy mã, nhập/xuất notebook, CSV, checkpoint.
- `src/lib/kernel.ts`: giao thức TypeScript với Python Worker; quản lý chạy, dừng, lỗi và biến chung.
- `src/lib/domain.ts`: kiểu route, kiểm tra dữ liệu nhập, hồi quy, gradient và chia dữ liệu.
- `src/components/Simulation.tsx`: cầu nối có kiểu tới 30 mô phỏng JavaScript đã có. Bộ thuật toán này được tái sử dụng, chưa chuyển toàn bộ sang TypeScript.
- `public/legacy/index.html`: mô phỏng hồi quy nâng cao được tạo bởi `scripts/legacy-embed.mjs` trong workspace gốc. Khung nhúng chỉ tải khi mở phần mô phỏng, tự điều chỉnh chiều cao và dùng các chế độ một biến, nhiều biến, Iris và CSV. Bài dự đoán nhanh và notebook do React hiển thị.
- `src/data/`: 30 bài, 90 bài tập, các notebook và 414 dòng nhà đất UCI.

Các checkpoint và tiến độ giữ khóa cũ để đọc dữ liệu đã lưu khi dùng cùng địa chỉ web. Chuyển từ file:// sang localhost hoặc đổi cổng là đổi vùng lưu của trình duyệt; dùng xuất/nhập tiến độ JSON khi cần chuyển.

## Kiểm tra đã thực hiện

- TypeScript strict: đạt.
- 6 kiểm tra tự động: route của cả 30 ngày và 6 giai đoạn; tiến độ; notebook nhập; lời giải tuyến tính đã biết; giảm loss thật; dữ liệu UCI chia độc lập và chuẩn hóa chỉ fit train.
- MAE validation cho ba đặc trưng bằng 6.1107032454, khớp notebook NumPy độc lập; baseline 10.3449252557. 248 train / 82 validation / 84 test, seed 42.
- Trình duyệt: React cập nhật loss 9.275 → 5.936 sau một cập nhật; notebook chạy đủ 4 ô trong WebAssembly; tìm attention và mở ngày 25; mô phỏng attention tổng trọng số bằng 1; phòng hồi quy cũ hiển thị đủ; bản HTML một tệp chạy và nút so sánh trả 2.831 tỷ cho dữ liệu giả lập.
- Sáu script trong HTML một tệp đều qua kiểm tra cú pháp; thứ tự khởi tạo bảo đảm vùng root và tài nguyên có trước React.

Không suy ra chất lượng định giá thực tế từ một phép chia dữ liệu nhỏ. Dữ liệu diện tích/giá tỷ đồng trong ví dụ đầu là giả lập; bộ UCI thật dùng 10.000 NTD/Ping ở Đài Loan và không có diện tích nhà.

Nguồn: Yeh, I. (2018). Real Estate Valuation. UCI Machine Learning Repository. DOI https://doi.org/10.24432/C5J30W, CC BY 4.0. Plotly.js được phân phối cùng thông báo bản quyền MIT gốc. Các giấy phép React/Vite/TypeScript/Tailwind nằm trong các gói npm tương ứng.


Notebook có tô sáng cú pháp Python bằng highlight.js (grammar Python), cập nhật ngay khi gõ; ô nhập vẫn hỗ trợ Tab, chọn/sao chép và Shift+Enter. Màu theo Jupyter classic: từ khóa xanh, chuỗi đỏ, toán tử tím, chú thích xanh xám. Tô màu không kiểm tra tính đúng đắn của Python; lỗi thực thi vẫn do kernel báo.


## Quan sát thuật toán từng phép tính — 19/11/2026

`AlgorithmLab.tsx` và `algorithmTrace.ts` cung cấp năm mô phỏng sâu cho hồi quy đơn/đa biến, logistic, KNN, K-means. Dùng trực tiếp trên dự án giá nhà và ngày 6/7/8/9/12/13/17. Mọi biểu đồ, bảng gradient và lịch sử dùng chung trace xác định. Đổi dữ liệu/learning rate khởi động lại trạng thái; tua tay dừng phát. Mô phỏng cũ dùng chung bộ điều khiển có thêm tốc độ. Dữ liệu minh họa là giả lập, không tạo metric đánh giá độc lập.

Notebook `Quan_sat_thuat_toan.ipynb` thực hiện lại công thức bằng NumPy, vẽ mô hình và kiểm tra trace tải từ web. Khởi động lại launcher để đăng ký notebook mới (33 mục). Tệp JSON trace phải do người học đưa vào thư mục notebook và chọn đường dẫn; không có đồng bộ kernel/biến giữa web và Jupyter.

Validation: 16 Node tests; React bundle DOM checks cho 5 tab, 7 route, play/pause/seek/reset/speed; 9 ô Python kiểm tra gồm ba trace (linear, multiple, logistic) độc lập bằng NumPy. DOM checks giả lập lệnh Plotly nên không khẳng định đã nghiệm thu hình thức trong trình duyệt thật.


## Đợt 2: CART và PCA (19/11/2026)

`treePcaTrace.ts` tính toàn bộ ứng viên CART và các trạng thái xây cây, cũng như PCA 2D với covariance mẫu và phép chiếu/tái tạo. `TreePcaLab.tsx` hiển thị cây SVG lớn dần, bảng Gini, eigenvectors và các phép tính từng điểm. `ObservationControls.tsx` tách bộ phát/tua dùng chung. Ngày 15 và 18 chuyển sang mô phỏng sâu; dự án giá nhà có tổng 7 lựa chọn.

Notebook `Cay_quyet_dinh_va_PCA.ipynb` đối chiếu công thức bằng NumPy và minh họa cây scikit-learn. Khởi động lại launcher để đăng ký notebook mới, tổng 34 mục. Kiểm tra: build + 24 Node tests; 12 ô Python (6 học liệu + 6 trace), DOM test 9 route. Chưa nghiệm thu hình thức trong trình duyệt thật. Dữ liệu là giả lập; cây không pruning, PCA chỉ 2D.


Mạng nơ-ron (ngày 19–21): bảy pha forward/backward/update, chọn cạnh/bias, gradient số, XOR/moons, 2/4/6 nút, tanh/tuyến tính, trace JSON và notebook NumPy thật. Hiện có 8 mô phỏng sâu ở 12 ngày. 32 Node tests và notebook đối chiếu 6 trace đạt; DOM kiểm tra tương tác có giả lập Plotly, chưa thay nghiệm thu hình thức trên trình duyệt. CNN/attention và nghiệm thu toàn bộ 30 notebook còn tiếp tục.


CNN/attention (ngày 22/25): ma trận tương tác theo từng bước; CNN cho sửa pixel, kernel, stride, padding, bias, ReLU/pooling; attention hiển thị Q/K/V, scale, mask, softmax và tổng trọng số. Trọng số đặt tay, chỉ forward. Notebook CNN_va_Attention_tung_buoc.ipynb đối chiếu JSON bằng NumPy, thực hành A/B/C. Hiện 10 nhóm mô phỏng sâu ở 14 ngày; 40 tests đạt, 6 trace kiểm chứng bằng Python thật. Kiểm tra DOM có giả lập Plotly; hình thức trình duyệt/di động chưa nghiệm thu.


Kiểm tra và dự án (#assessments): ba mốc 7/16/21, 24 câu có dữ liệu trực quan, chấm đủ 8 câu và ngưỡng 6/8, giải thích/ôn lại, lịch sử 20 lần/mốc. Rubric 100 điểm có sáu tiêu chí 0–4, ghi minh chứng, ngưỡng 70 khi đủ tiêu chí. Rubric là tự đánh giá, chưa kiểm chứng chất lượng minh chứng. assessment là trường tùy chọn trong progress version 1; nhập cũ, nhập một phần, xuất JSON và Markdown. Notebook Tu_kiem_tra_va_Rubric.ipynb đối chiếu độc lập bằng Python. 48 Node tests đạt; DOM test nộp/sửa/import/reload/storage failure đạt; notebook 4 ô học liệu + 3 cấu hình đối chiếu chạy thật. Chưa nghiệm thu hình thức trình duyệt/di động hoặc toàn bộ 30 notebook.


Bàn giao tự học: #guide và #guide-1…#guide-30 có giải thích, ví dụ tính tay, A/B/C, đáp án, lỗi thường gặp, thuật ngữ và hướng học tiếp. Cẩm nang Markdown và báo cáo kiểm tra có trong ZIP. 30 notebook được thêm ghi chú, không điền đáp án. 90/90 trường hợp chạy kernel mới đạt: template, lời giải, đáp án sai bị từ chối; 270 ô code. 52 Node tests; hash mã/lời giải khớp báo cáo. Ghi chú được thêm sau lượt chạy; mã không đổi.

Tái kiểm tra (trong môi trường Jupyter đã cài requirements): chuyển vào ml-studio-react, chạy `python scripts/verify_notebooks.py --output ../ket_qua_kiem_thu --workers 2`. Script tạo bản sao và báo cáo, không sửa notebook học viên. Báo cáo không chấm tự luận A/C hoặc giao diện. Giao diện thật/di động và nâng sâu các mô phỏng cơ bản còn lại chưa nghiệm thu.
