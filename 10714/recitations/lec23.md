# Model Deployment https://youtu.be/jCBrUisBQ0A

Triển khai mô hình sau huấn luyện tới nhiều môi trường khác nhau (mobile, thiết bị nhúng, servers ...)

Chúng ta sẽ thường phải gói model vào một môi trường chạy native trên thiết bị đó, và gọi các hàm native trên thiết bị đó để sử dụng tối đa phần cứng (ví dụ GPU trên mobile phones). Một vấn đề nữa làm mô hình được huấn luyện thường có kích thước lớn (vài trăm MB hoặc vài GBs), làm thế nào để thu nhỏ kích thước phù hợp cho mobile và thiết bị nhúng cũng là một vấn đề khác cần quan tâm.

Trước khi triển khai mô hình ta sẽ cần cân nhắc:
- Những hạn chế của môi trường triển khai
- Acceleration của local hardwares
- Integration with the app (data preprocessing, post processing)

