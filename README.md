# 🧠 reCAPTCHA Solver with Image Processing & Deep Learning
📝 Mô tả dự án
Dự án này nhằm mục đích tự động giải reCAPTCHA ảnh (ví dụ: "Chọn tất cả các hình có xe buýt") thông qua các bước xử lý ảnh và mô hình học sâu. Ứng dụng này hỗ trợ việc tự động hóa các tác vụ cần vượt qua reCAPTCHA, ví dụ như trong các hệ thống kiểm thử tự động.

📌 Quy trình hoạt động
Input:

Ảnh chụp màn hình reCAPTCHA hoặc ảnh chụp riêng phần CAPTCHA.

Có thể là ảnh dạng lưới (3x3, 4x4, v.v.).

+ Xử lý ảnh:

Phân tích và cắt ảnh theo từng ô nhỏ.

Tiền xử lý để đưa ảnh về định dạng phù hợp với mô hình.

+ Dự đoán:

Sử dụng mô hình học sâu để xác định xem ảnh có chứa đối tượng yêu cầu hay không (ví dụ: xe buýt, đèn giao thông,...).

+ Click tự động:

Dựa vào kết quả dự đoán, xác định các vị trí cần click theo tọa độ tuyệt đối trên màn hình hoặc vị trí trong lưới (ví dụ: ô số 2, ô số 5,...).

Có thể sử dụng thư viện như pyautogui để click tự động nếu cần.

🧰 Công nghệ sử dụng
Python

OpenCV cho xử lý ảnh

TensorFlow / PyTorch cho mô hình học sâu

PyAutoGUI (tùy chọn) để tự động click chuột

PIL / NumPy

Các mô hình pretrained (hoặc custom CNN) để nhận dạng đối tượng

🚀 Hướng dẫn sử dụng
Clone repo:
```bash git clone https://github.com/your-username/recaptcha-solver.git cd recaptcha-solver ``` 

Cài đặt thư viện:
```bash pip install -r requirements.txt```

Chạy demo:
```bash python main.py --image path/to/your/captcha.png```
