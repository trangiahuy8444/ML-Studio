import type {ProjectId} from './ids';

export interface ProjectMeta {
  id:ProjectId|'housing';
  chapter:number;
  title:string;
  question:string;
  summary:string;
  days:number[];
  skills:string[];
  data:string;
}
export interface Chapter {
  id:number;
  title:string;
  days:[number,number];
  outcomes:string[];
  product:string;
  checkpoint?:number;
  projects:(ProjectId|'housing')[];
}

export const CHAPTERS:Chapter[]=[
  {id:1,title:'Nền tảng AI & dữ liệu',days:[1,5],projects:[],
    outcomes:['Đọc một bảng dữ liệu và chỉ ra đâu là đầu vào X, đâu là đáp án y','Biểu diễn một mẫu thành vector và tính khoảng cách giữa hai mẫu','Giải thích bằng lời: mô hình sai bao nhiêu và nên sửa theo hướng nào (một bước gradient)'],
    product:'Bảng quan sát dữ liệu nhà đất và một bước cập nhật em tự tính tay.'},
  {id:2,title:'Mô hình đầu tiên',days:[6,11],checkpoint:7,projects:['housing'],
    outcomes:['Tự huấn luyện hồi quy tuyến tính và đọc được đường loss','Chuẩn hóa dữ liệu nhiều đặc trưng và nói được vì sao cần','Chia train / validation / test và nhận ra overfitting'],
    product:'Dự án giá nhà hoàn chỉnh: từ đoán bằng mắt đến mô hình trên dữ liệu thật.'},
  {id:3,title:'Machine Learning thực hành',days:[12,18],checkpoint:16,projects:['tumor','spam','iris','customers'],
    outcomes:['Xây và so sánh ba kiểu mô hình phân loại: logistic, KNN, cây quyết định','Chọn ngưỡng theo cái giá của sai lầm thay vì chỉ nhìn accuracy','Tìm nhóm trong dữ liệu không nhãn bằng K-means và nhìn dữ liệu nhiều chiều bằng PCA'],
    product:'Bốn dự án phân loại và phân cụm, mỗi dự án có một kết luận em tự viết.'},
  {id:4,title:'Mạng nơ-ron & hình ảnh',days:[19,22],checkpoint:21,projects:['digits'],
    outcomes:['Theo dõi một ảnh đi từ pixel thành vector rồi thành dự đoán','Giải thích mạng học "khuôn mẫu" thế nào khi nhìn vào trọng số','Hiểu bộ lọc tích chập tìm cạnh và nét trong ảnh ra sao'],
    product:'Bộ đọc chữ số viết tay em tự vẽ để thử, kèm bảng các lỗi mô hình hay mắc.'},
  {id:5,title:'Ngôn ngữ & các hướng AI',days:[23,27],projects:['sentiment','rl'],
    outcomes:['Biến câu chữ thành vector bằng túi từ, bigram và embedding','Giải thích vì sao từ gần nghĩa giúp mô hình đoán được câu chưa gặp','Cân bằng khám phá và khai thác, huấn luyện một tác tử học tăng cường nhỏ'],
    product:'Bộ đoán cảm xúc bình luận tiếng Việt và một robot tự học đường đi.'},
  {id:6,title:'Dự án của em',days:[28,30],projects:['gnn'],
    outcomes:['Đặt câu hỏi, chọn dữ liệu và chỉ số đánh giá cho bài toán của riêng em','Chạy thí nghiệm, phân tích lỗi và viết báo cáo chạy lại được','Mở rộng: dùng quan hệ giữa các mẫu (đồ thị) để phân loại'],
    product:'Một notebook dự án riêng đạt từ 70/100 theo rubric tự đánh giá.'},
];

export const PROJECTS:Record<ProjectId|'housing',ProjectMeta>={
  housing:{id:'housing',chapter:2,title:'Dự án giá nhà',question:'Nhà 80 m² giá bao nhiêu?',summary:'Đoán giá, tự vẽ đường thẳng, cho máy học, rồi thử dữ liệu nhà đất thật.',days:[6,7,8,10],skills:['Hồi quy tuyến tính','Gradient descent','Chuẩn hóa','Train/validation'],data:'Dữ liệu giả lập + UCI Real Estate Valuation'},
  tumor:{id:'tumor',chapter:3,title:'Chẩn đoán khối u',question:'Khối u này lành hay ác?',summary:'Vẽ ranh giới, cho máy học logistic và KNN, rồi chọn ngưỡng sao cho ít bỏ sót ca ác tính.',days:[12,13,14],skills:['Hồi quy logistic','KNN','Ngưỡng quyết định','Recall vs accuracy'],data:'Breast Cancer Wisconsin (UCI), 569 ca, 2 đặc trưng đã chuẩn hóa'},
  spam:{id:'spam',chapter:3,title:'Lọc email rác',question:'Email này có phải thư rác?',summary:'Tự lọc thử, đếm từ, để máy học trọng số từng từ, rồi viết một email tìm cách lừa bộ lọc.',days:[12,14,23],skills:['Túi từ','Naive Bayes','Đóng góp của từng từ','Precision'],data:'40 email tiếng Việt tự soạn để minh họa'},
  iris:{id:'iris',chapter:3,title:'Nhận biết loài hoa Iris',question:'Hoa này thuộc loài nào?',summary:'Tự đặt câu hỏi chia nhóm, xem cây quyết định tìm câu hỏi tốt nhất và khi nào cây quá sâu.',days:[15,16],skills:['Cây quyết định','Gini','Độ sâu và overfitting'],data:'Iris (Fisher, 1936), 150 bông, chiều dài/rộng cánh hoa'},
  customers:{id:'customers',chapter:3,title:'Phân nhóm khách hàng',question:'Cửa hàng có những kiểu khách nào?',summary:'Tự chia nhóm, xem K-means chạy từng bước, chọn số nhóm và nhìn 4 chiều bằng PCA.',days:[17,18],skills:['K-means','Chọn k','PCA','Diễn giải cụm'],data:'90 khách hàng giả lập, 4 đặc trưng'},
  digits:{id:'digits',chapter:4,title:'Đọc chữ số viết tay',question:'Máy đọc nét chữ của em thế nào?',summary:'Vẽ một chữ số, xem hàng xóm giống nhất, mạng học khuôn mẫu, bộ lọc tích chập và các lỗi nhầm.',days:[19,20,21,22],skills:['Pixel thành vector','Softmax','Trọng số là khuôn mẫu','Tích chập'],data:'UCI Optical Digits 8×8, 400 ảnh'},
  sentiment:{id:'sentiment',chapter:5,title:'Cảm xúc bình luận tiếng Việt',question:'Bình luận này khen hay chê?',summary:'Từ túi từ đến bigram và embedding: vì sao "không hay" làm khó mô hình và từ gần nghĩa giúp gì.',days:[23,24,25],skills:['Tách từ','Bigram','Embedding','Giới hạn mô hình'],data:'48 phản hồi sinh viên tự soạn theo kiểu UIT-VSFC'},
  rl:{id:'rl',chapter:5,title:'Máy chơi game học dần',question:'Thử cái mới hay giữ cái đang tốt?',summary:'Tự chơi máy đánh bạc, so sánh chiến lược ε-greedy, rồi huấn luyện robot tìm đường bằng Q-learning.',days:[27],skills:['Khám phá/khai thác','ε-greedy','Q-learning','Phần thưởng'],data:'Môi trường mô phỏng, hạt giống cố định'},
  gnn:{id:'gnn',chapter:6,title:'Phân loại bài báo bằng đồ thị',question:'Đọc trích dẫn có đoán được chủ đề?',summary:'Đoán chủ đề từ hàng xóm, so sánh nhìn nội dung với truyền thông điệp, thử thêm cạnh nhiễu.',days:[30],skills:['Đồ thị','Truyền thông điệp','Làm mượt quá mức','Cạnh nhiễu'],data:'Đồ thị trích dẫn 14 bài báo giả lập'},
};
