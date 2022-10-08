https://dlsyscourse.org

Khóa học cung cấp sự hiểu biết và tổng quan về "toàn bộ" hệ thống học sâu, từ thiết kế mô hình hóa cấp cao của các hệ thống học sâu hiện đại, đến việc triển khai cơ bản các công cụ đạo hàm tự động, đến triển khai các thuật toán hiệu quả trên từng thiết bị. 

Thông qua khóa học, bạn sẽ thiết kế và xây dựng từ đầu một thư viện học sâu hoàn chỉnh - hoạt động hiệu quả trên GPU, đạo hàm tự động tất cả các hàm được triển khai, và phát triển các mô-đun cần thiết để hỗ trợ các lớp được tham số hóa, các hàm mất mát, bộ tải dữ liệu và trình tối ưu hóa.

Sử dụng các công cụ này, bạn sẽ xây dựng một số mô hình học sâu hiện đại, bao gồm mạng tích chập (cnn) để phân loại và phân loại hình ảnh, mạng lặp lại (rnn) và mô hình tự chú ý (self-attention) cho các tác vụ tuần tự như mô hình ngôn ngữ và mô hình sinh ra (generative) để tạo hình ảnh.

Thông qua các bài tập, bạn sẽ xây dựng một thư viện học sâu cơ bản, có thể so sánh với phiên bản tối thiểu của `PyTorch` hoặc `TensorFlow`, có thể mở rộng cho một hệ thống có kích thước vừa phải. 

Dự án cuối cùng, sẽ được thực hiện theo nhóm 2-3 sinh viên, bao gồm việc triển khai một tính năng thực sự mới cho thư viện vừa phát triển, cộng với việc triển khai một mô hình sử dụng tính năng này với thư viện kể trên (mà chưa được thực hiện trong PyTorch / Tensorflow). 

Chúng tôi sẽ cung cấp một số tính năng và mô hình mẫu, bao gồm các phương pháp để tăng tốc phần cứng hơn nữa, đào tạo đối đầu, toán đạo hàm tự động nâng cao (ví dụ: toán tử đại số tuyến tính như system solves / SVD), mô hình xác suất, v.v.


## Additonal Resources

https://github.com/geohot/tinygrad

https://github.com/karpathy/nn-zero-to-hero


## 1/ Intro

https://www.youtube.com/watch?v=ftP5HeOvsI

### Lý do số 1: Để xây dựng hệ thống học sâu

Bất chấp sự thống trị của các thư viện học sâu và TensorFlow và PyTorch, sân chơi này rất linh hoạt (ví dụ: sự xuất hiện gần đây của JAX)

Bạn có thể muốn phát triển các thư viện hiện có (mã nguồn mở) hoặc phát triển các thư viện mới của riêng bạn cho các tác vụ cụ thể.

### Lý do thứ 2: Để sử dụng các hệ thống hiện có hiệu quả hơn

Hiểu cách hoạt động bên trong của các hệ thống học sâu cho phép bạn sử dụng chúng hiệu quả hơn nhiều. Muốn làm cho lớp tùy chỉnh của bạn chạy nhanh hơn (nhiều) trong TensorFlow / PyTorch? … Bạn sẽ muốn hiểu cách từng hoạt động được thực hiện.

Hiểu được các hệ thống học sâu là một “siêu năng lực” sẽ cho phép bạn hoàn thành mục tiêu nghiên cứu hiệu quả hơn nhiều.

### Lý do thứ 3: Hệ thống học sâu rất thú vị!

Bất chấp sự phức tạp của chúng, các thuật toán cốt lõi đằng sau (đạo hàm tự động + tối ưu hóa dựa trên gradient) là cực kỳ đơn giản. Bạn có thể viết một thư viện học sâu với <2000 dòng mã lệnh.

Lần đầu tiên bạn xây dựng thư viện đạo hàm tự động và nhận ra rằng có thể lấy gradient của một gradient mà không cần thực sự biết sẽ triển khai chúng dưới dạng công thức toán học như thế nào.

## [2/ Học máy / hồi quy softmax](lec02.md)

Bài giảng này bao gồm các nguyên tắc cơ bản của học máy (có giám sát), được minh họa bằng thuật toán hồi quy softmax. Chúng ta sẽ tìm hiểu về phương pháp hồi quy softmax và phương pháp suy giảm độ dốc ngẫu nhiên được áp dụng để huấn luyện loại mô hình này.

