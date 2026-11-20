# Lời giải tham khảo bài B

Hãy tự làm trước khi xem. Các lời giải này đã được chạy qua phần kiểm tra; việc qua kiểm tra tính toán chưa thay thế cho giải thích và phân tích kết quả.

## Ngày 1: AI là gì và em sẽ học điều gì?

Gán ba nhiệm vụ dưới đây vào regression, classification hoặc generation.

```python
answer = ['regression', 'classification', 'generation']
```

Nhìn vào kiểu đầu ra, không dựa vào tên ứng dụng.

## Ngày 2: Dùng Python để quan sát dữ liệu

Lọc các điểm hữu hạn và từ 60 trở lên trong scores. Giữ thứ tự xuất hiện.

```python
answer = scores[np.isfinite(scores) & (scores >= 60)]
```

Kết hợp np.isfinite với một điều kiện so sánh bằng toán tử &.

## Ngày 3: Toán cần dùng: vector và ma trận

Tính Xw + b cho cả ba hàng, không dùng vòng lặp.

```python
answer = X @ w + b
```

Nhân ma trận @ rồi cộng một hệ số chặn cho mỗi hàng.

## Ngày 4: Dữ liệu có biến động và bất định

Trả về [mean, median] cho values, rồi giải thích đại lượng nào nhạy với ngoại lệ hơn.

```python
answer = [np.mean(values), np.median(values)]
```

Mean dùng tổng chia số lượng; median là giá trị giữa sau khi sắp xếp.

## Ngày 5: Tại sao mô hình biết phải thay đổi theo hướng nào?

Thực hiện một bước cho L(w)=(w−3)² từ w=0 với alpha=0.1.

```python
answer = w - alpha * 2 * (w - 3)
```

Đạo hàm là 2(w−3); cập nhật trừ alpha nhân gradient.

## Ngày 6: Hồi quy tuyến tính một biến

Với x=3, y=45, b=20, w=7.5, trả về [dự đoán, sai số có dấu, bình phương sai số].

```python
pred = b0 + w0*x0
answer = [pred, pred-y0, (pred-y0)**2]
```

Dùng e = dự đoán − thực tế; đừng bỏ dấu của e.

## Ngày 7: Mốc 1: mô hình đầu tiên của em

Viết bước cập nhật toàn bộ theta bằng gradient của tập train trong ví dụ.

```python
answer = theta - alpha * (2/len(tr) * Z[tr].T @ (Z[tr]@theta-y[tr]))
```

Gradient dùng tất cả các mẫu train; cập nhật đồng thời từ theta cũ.

## Ngày 8: Hồi quy nhiều biến và chuẩn hóa

Đổi theta=[10,6,−4], mu=[3,2], sd=[2,4] về hệ số gốc [b,w1,w2].

```python
raw_w = theta[1:] / sd
answer = np.r_[theta[0] - raw_w @ mu, raw_w]
```

w gốc = theta/sd; b gốc = theta0 − tổng(w gốc × mu).

## Ngày 9: Một lần cập nhật dùng bao nhiêu dữ liệu?

Một mẫu z=[1,2], y=5, theta=[0,0], alpha=0.1: tính theta mới theo SGD và MSE.

```python
answer = theta - alpha * 2 * (z @ theta - target) * z
```

Một mẫu có gradient 2e×z; không chia thêm cho số mẫu của tập đầy đủ.

## Ngày 10: Đánh giá một mô hình có đáng tin không?

Tính [MAE,MSE,R²] cho truth và pred dưới đây.

```python
err = pred-truth
answer = [np.mean(np.abs(err)),np.mean(err**2),1-np.sum(err**2)/np.sum((truth-truth.mean())**2)]
```

R² dùng tổng bình phương sai số và độ biến động quanh trung bình của truth.

## Ngày 11: Overfitting và regularization

Chọn bậc đa thức theo val_loss, không theo train_loss. Trả về số bậc.

```python
answer = int(degrees[np.argmin(val_loss)])
```

Dữ liệu validation đại diện cho dữ liệu không dùng để fit hệ số.

## Ngày 12: Hồi quy logistic cho phân loại

Trả về [sigmoid(0), BCE cho y=1,p=0.8, nhãn của p=0.55 ở ngưỡng 0.6].

```python
answer = [sigmoid(0), -np.log(.8), int(.55 >= .6)]
```

BCE cho nhãn 1 là −log(p); nhãn chỉ được tạo sau bước so với ngưỡng.

## Ngày 13: KNN: học từ các điểm gần nhau

Với X=[0,1,2], nhãn [0,0,1], dự đoán cho 1.8 ở k=1 và k=3.

```python
answer = [knn_predict(X,y,q,k=1)[0],knn_predict(X,y,q,k=3)[0]]
```

k=1 lấy nhãn gần nhất; k=3 bỏ phiếu cả ba điểm.

## Ngày 14: Accuracy có thể đánh lừa như thế nào?

Tính [precision,recall,F1] cho TP=8,FP=2,FN=4,TN=6.

```python
precision=tp/(tp+fp);recall=tp/(tp+fn)
answer=[precision,recall,2*precision*recall/(precision+recall)]
```

Precision hỏi các dự đoán dương đúng bao nhiêu; recall hỏi các mẫu dương được tìm ra bao nhiêu.

## Ngày 15: Cây quyết định và cách mô hình đặt câu hỏi

Tính mức giảm Gini khi [0,0,1,1] được tách thành [0,0] và [1,1].

```python
answer = gini(parent) - .5*gini(left) - .5*gini(right)
```

Hai lá thuần nhất có Gini bằng 0; cần trừ Gini sau chia có trọng số.

## Ngày 16: Mốc 2: tự tổ chức một thí nghiệm ML

Chỉ fit chuẩn hóa từ train=[1,2,3]. Trả về [mu,sd,z] cho một giá trị mới 100.

```python
mu=train_values.mean();sd=train_values.std()
answer=[mu,sd,(new_value-mu)/sd]
```

Độ lệch chuẩn ở đây dùng ddof=0 giống NumPy mặc định.

## Ngày 17: K-means và học không có nhãn

Sau một vòng K-means cho [0,2,8,10] với tâm [0,10], trả về hai tâm mới theo thứ tự.

```python
labels=np.abs(points[:,None]-old_centers[None,:]).argmin(1)
answer=[points[labels==j].mean() for j in range(2)]
```

Gán theo tâm cũ, rồi lấy trung bình từng nhóm để cập nhật đồng thời.

## Ngày 18: PCA: nhìn dữ liệu từ một hướng khác

Với ba điểm cùng đường y=x, tính tỷ lệ phương sai của thành phần PCA đầu tiên.

```python
answer = ratio[0]
```

Dữ liệu nằm hoàn toàn trên một đường nên một thành phần giữ toàn bộ phương sai khác 0.

## Ngày 19: Một mạng nơ-ron tính toán như thế nào?

Tính [logit,ReLU(logit)] với x=[1,3], w=[2,−1], b=0.5.

```python
logit=x@w+b
answer=[logit,max(0,logit)]
```

Tính tổng có trọng số trước khi áp dụng ReLU.

## Ngày 20: Backpropagation: gradient đi ngược ra sao?

Tính dL/dw cho L=(wx−y)² tại w=1,x=2,y=5.

```python
answer = 2*(w*x-y)*x
```

Chain rule nhân đạo hàm của bình phương với đạo hàm wx theo w.

## Ngày 21: Mốc 3: xem mạng nơ-ron học biên quyết định

Tính xác suất forward cho mạng tanh hai nút ẩn được cho, không huấn luyện thêm.

```python
answer = sigmoid(np.tanh(x0 @ W0 + b0) @ V0).ravel()
```

Tính xW+b → tanh → nhân V → sigmoid; chú ý shape.

## Ngày 22: Từ pixel đến ý tưởng CNN

Áp bộ lọc trung bình 3×3 kiểu valid cho image; trả về ảnh 3×3.

```python
answer = conv_valid(image,kernel3)
```

Valid không thêm padding; mỗi vị trí là tổng của một cửa sổ nhân kernel.

## Ngày 23: Biến câu chữ thành đầu vào cho mô hình

Với vocab=[hoc,ai,hay], tạo vector đếm cho câu “hoc ai ai”.

```python
answer = [text.split().count(token) for token in vocab]
```

Đếm theo đúng thứ tự vocabulary, không theo thứ tự từ xuất hiện.

## Ngày 24: Embeddings và ý tưởng Word2Vec

Tính cosine giữa a=[1,0] và b=[1,1].

```python
answer = (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b))
```

Cosine chuẩn hóa dot product bằng tích độ dài; không phải khoảng cách Euclidean.

## Ngày 25: Attention và Transformer ở mức trực giác

Tính attention weights cho query q và K, chia sqrt(d_k) rồi softmax.

```python
answer = softmax(q @ K.T / np.sqrt(K.shape[1]))
```

Softmax tạo các trọng số dương có tổng bằng 1; bài này chỉ là một head nhỏ.

## Ngày 26: Mô hình ngôn ngữ, truy xuất và kiểm chứng

Truy xuất đoạn có nhiều từ truy vấn trùng nhất; trả về chỉ số bắt đầu từ 0.

```python
answer = int(np.argmax(scores))
```

Dùng chỉ số của score lớn nhất; bài không yêu cầu gọi API.

## Ngày 27: Học tăng cường và đánh đổi khám phá/khai thác

Q cũ=0.4 sau 4 lượt, reward mới=1: cập nhật bằng trung bình tăng dần.

```python
answer = old_Q + (reward-old_Q)/(old_count+1)
```

Mẫu thưởng mới là lượt thứ 5; không chia cho số lượt cũ.

## Ngày 28: Dự án cuối, buổi 1: đặt đúng câu hỏi

Tạo ba phần Iris có phân tầng theo iris_split; trả về số mẫu [train,validation,test].

```python
answer = [len(tr), len(va), len(te)]
```

Mỗi lớp có 30 train, 10 validation và 10 test; test được để riêng.

## Ngày 29: Dự án cuối, buổi 2: thí nghiệm và phân tích lỗi

Trong bảng minh họa val_scores, chọn tên mô hình tốt nhất. Không chọn theo train_scores.

```python
answer = model_names[int(np.argmax(val_scores))]
```

Chọn theo validation và tiêu chí đã định; test không tham gia lựa chọn.

## Ngày 30: Dự án cuối, buổi 3: đánh giá và giải thích

Tạo báo cáo [số mẫu test,accuracy,macro-F1] cho mô hình đã chọn bằng validation.

```python
answer = [len(te), accuracy(y[te],final_pred), macro_f1(y[te],final_pred)]
```

Báo cả số mẫu và macro-F1; chất lượng quy trình quan trọng hơn một ngưỡng accuracy cố định.

