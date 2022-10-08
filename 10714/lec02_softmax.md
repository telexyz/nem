https://machinelearningcoban.com/2017/02/17/softmax

# [2/ Học máy / hồi quy softmax](https://www.youtube.com/watch?v=MlivXhZFbNA)

## 3 thành tố của 1 thuật toán học máy:

* __Lớp giả thiết__: là cấu trúc của mô hình được định hình bởi một tập các tham số, nó (là một hàm số) mô tả cách chúng ta ánh xạ đầu (ví dụ ảnh của các chữ số từ 0-9) vào thành đầu ra (các chữ số từ 0-9)

* __Hàm mất mát__: một hàm định nghĩa độ tốt của một giả thiết (là 1 lựa chọn của bộ tham số) cho một tác vụ cho trước.

* __Phương thức tối ưu hóa__ một cách để chúng ta tối ưu hóa bộ tham số để tìm ra một bộ tham số tối thiểu (một cách xấp xỉ) tổng mất mát trên một tập dữ liệu huấn luyện.

Mọi thuật toán học máy đều đều được cấu tạo bởi 3 thành tố trên. Để hiểu một thuật toán học máy, trước hết chúng ta hãy tìm hiểu cách chúng cấu thành nên từ các thành tố cơ bản như thế nào.

## Ví dụ thuật toán hồi quy softmax

### Hãy xem xét bài toán phân loại k-lớp, với:

Dữ liệu huấn luyện: $x^{(i)} \in R^n$, $y^{(i)} \in {1, ..., k}$ với $i = 1, ... m$ 
* $n =$ số chiều của không gian chứa dữ liệu đầu vào (mỗi ví dụ huấn luyện tương ứng với một điểm trong không gian dữ liệu đầu vào)
* $k =$ số lượng phân lớp / hay số lượng các nhãn khác nhau
* $m =$ số điểm trong tập huấn luyện

Note: $x^{(i)} = [x^{(i)}_1, x^{(i)}_2, ..., x^{(i)}_n]$

Ví dụ: phân loại các chữ số 28x28 của tập MNIST
* $n = 28 \cdot 28 = 784$
* $k = 10$ (10 nhãn tương đương với các chữ số từ 0-9)
* $m = 60.000$

### Hàm giả thiết tuyến tính

Hàm giả thiết tuyến tính ánh xạ đầu vào $x \in R^n$ vào các vectors $k$ chiều $h: R^n \rightarrow R^k$ sử dụng hàm số $h_i{(x)}$.

Hàm giả thiết $h_i{(x)}$ thể hiện một độ đo sự tin tưởng vào khả năng nhãn của $x$ thuộc về lớp $i$.

Một **hàm giả thiết tuyến tính** sử dụng một toán tử *tuyến tính* (tức là phép nhân ma trận) cho phép biến đổi này ${h}_\theta (x) = \theta^{T}{x}$ cho tham số $\theta \in R^{n \times k}$

### Ký hiệu lô ma trận

<!-- Thường thuận tiện hơn (và hiệu quả hơn) khi viết dữ liệu và hoạt động ở dạng lô ma trận.
$X \in \mathbb{R}^{m \times n}=\left[\begin{array}{c}-x^{(1)^T}- \vdots -x^{(m)^T}\end{array}\right], \quad y \in\{1, \ldots, k\}^m=\left[\begin{array}{c}y^{(1)} \vdots y^{(m)}\end{array}\right]$

Giả thiết tuyến tính được áp dụng cho lô này có thể được viết như sau:
$h_\theta(X)=\left[\begin{array}{c}-h_\theta\left(x^{(1)}\right)^T- \vdots -h_\theta\left(x^m\right)^T-\end{array}\right]=\left[\begin{array}{c}-x^{(1)^T} \theta- \vdots -x^{(m)^T} \theta-\end{array}\right]=X \theta$
 -->

Ta thường áp dụng hàm giả thiết với một điểm dữ liệu nhất định $x^{i}_k$ (với k thuộc 1..n), nhưng hóa ra việc áp dụng hàm giả thiết với nhiều điểm dữ liệu một lúc (lô dữ liệu) sẽ thuận tiện hơn cho việc triển khai trong thực tế, vì thao tác trên ma trận sẽ hiệu quả hơn việc thực hiện nhiều thao tác trên nhiều vector. Cả CPU và GPU đều hoạt động hiệu quả hơn khi ta thực thi các thuật toán dưới dạng lô ma trận (xem các cách cài đặt thuật toán nhân ma trận ảnh hưởng tới hiệu năng như thế nào tại [bài viết này](https://github.com/telexyz/Algorithms-For-Modern-Hardware/blob/master/content/vietnamese/hpc/complexity/languages.md#ng%C3%B4n-ng%E1%BB%AF-th%C3%B4ng-d%E1%BB%8Bch) và [video này](https://youtu.be/o7h_sYMk_oc?t=955)).

Các ký hiệu lô ma trận bao gồm: $X$ là lô ma trận kích thước $m \times n$, với $m$ là số lượng điểm dữ liệu, và $n$ là chiều của điểm dữ liệu. Ma trận này là sự xếp chồng của tất cả các điểm dữ liệu trong tập huấn luyện. Dòng đầu tiên của ma trận tương ứng với điểm dữ liệu đầu tiên $x^{(i)}$ ... Bởi vì ta để từng điểm dữ liệu theo từng hàng nên ta cần chuyển vị nó từ dạng cột sang dạng hàng $x^{(i)^T}$ ...