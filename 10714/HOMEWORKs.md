## [hw0](https://colab.research.google.com/drive/12hD-w_GHgbLQfYwOXIJUg3jhvH1HQB8-)

Bài tập này cung cấp cái nhìn tổng quát về một số khái niệm và ý tưởng mà bạn nên làm quen _trước khi_ khi tham gia khóa học. Bài tập sẽ yêu cầu bạn xây dựng một thuật toán hồi quy softmax cơ bản, cộng với một mạng nơ-ron hai lớp đơn giản. Bạn sẽ triển khai bằng cả Python (sử dụng thư viện numpy) và C / C++ (đối với hồi quy softmax).

Tất cả quá trình phát triển mã cho các bài tập về nhà có thể được thực hiện trong môi trường Google Colab. Tuy nhiên, thay vì sử dụng các khối mã trực tiếp trong colab, hầu hết mã bạn phát triển sẽ được thực hiện bằng các tệp `.py` được tải xuống (tự động) vào Google Drive của bạn và sử dụng colab để chạy các tập shell để thực thi rồi gửi mã cho máy chấm điểm tự động. Đây là cách sử dụng hơi phi tiêu chuẩn của Colab Notebooks (thường người ta sử dụng chúng giống như môi trường tương tác, với các ô mã được chạy trực tiếp trong sổ tay).

Lý do chúng tôi để sử dụng chúng theo cách này rất đơn giản: ngoài việc là một môi trường sổ tay điện toán đám mây, Colab cũng cung cấp quyền truy cập vào GPU, điều này sẽ cho phép bạn phát triển một số phần mềm GPU sau này mà không yêu cầu bạn phải có GPU vật lý hoặc tự thiết lập các thư viện CUDA.