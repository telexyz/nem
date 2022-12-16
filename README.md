# Học

Thích gì học nấy deep learning, database, intepreter - compiler - programming language, system programming, parallel and distributed programming.

## 09/2022-12/2022

- Hoàn thành `10714`, tự xây dựng thư viện học sâu `kim`

## 12/2022-06/2023

- Hoàn thành `15445`, hiểu cách xây dựng một database cơ bản

- Hoàn thành `15721`, [Advanced Database Systems](https://15721.courses.cs.cmu.edu/spring2020/schedule.html)

Database quan tâm https://github.com/tigerbeetledb/tigerbeetle/blob/main/docs/DESIGN.md: The distributed financial accounting database designed for mission critical safety and performance.

- [cs61c](https://cs61c.org) Great Ideas in Computer Architecture
- [cs4414](https://www.cs.cornell.edu/courses/cs4414) System programming
- [15418](http://15418.courses.cs.cmu.edu) Parallel Computer Architecture and Programming
- [6.824](https://pdos.csail.mit.edu/6.824) Distributed System

> Coi videos ở [15418](http://15418.courses.cs.cmu.edu/spring2016) làm bài tập ở [cs149](https://gfxcourses.stanford.edu/cs149/fall21) vì 2 khóa này của cùng 1 giáo.
_source https://csdiy.wiki_

## 07/2023-xxx

Có thể học viết trình thông dịch theo hướng dẫn trong cuốn https://craftinginterpreters.com bằng ngôn ngữ https://ziglang.org

Hoặc tìm hiểu về trình biên dịch và LLVM qua mã nguồn Zig, bắt đầu từ https://mitchellh.com/zig

### Tại sao chọn Zig?
Zig là một ngôn ngữ lập trình hệ thống hiện đại, đơn giản và mạnh mẽ. Có thể học syntax trong buổi sáng và chiều làm dự án luôn. Zig mới và chưa ra bản 1.0 (có lẽ vào cuối 2023) nhưng đã có 2 startups thành công sử dụng Zig (Bun.js và TigerBeetle). Cộng đồng rất mạnh.

Có thể tham gia sâu hơn và Zig bằng cách:

- Chữa những lỗi đơn giản như một người mới
- Cải tiến nvptx backend chạy mượt [zcuda](https://github.com/gwenzek/cudaz) (xem [video](https://www.youtube.com/watch?v=rvfsWm6TckA&t=5351s))
- ...

## xxx-xxx

Học lập trình hệ thống bằng cách tham gia https://github.com/SerenityOS/serenity
Dự án khởi đầu của một cựu lập trình viên Apple, tự code một hệ điều hành lý tưởng theo cách bạn ấy tưởng tượng (Unix style + retro windows 9.x ui/ux) như là một cách phục hồi sau khi cai nghiện. Bạn này chia sẻ những gì mình làm trên youtube và nhận được hưởng ứng lớn từ cộng đồng với gần 1000 contributors. Mọi thứ được viết từ đầu hết mà ko dùng thư viện có sẵn, kể cả web browser. Nghe có vẻ điên rồ nhưng bạn ấy và cộng đồng đã và đang làm rất tốt.


- - -


# Làm (dự án)

Cần kiến thức lập trình hệ thống, toán, cấu trúc dữ liệu và giải thuật, kiến trúc máy tính cơ bản (phân cấp bộ nhớ, cache, nhân xử lý ...), lập trình hướng dữ liệu, lập trình song song / phân tán, lập trình hiệu năng cao.


## Dùng GPU xử lý dữ liệu lớn (ý tưởng)

Bất kỳ dữ liệu nào, áp dụng hiểu biết về lập trình GPU và database. Có thể liên quan tới thời gian thực và tài chính.


## Bộ gõ tiếng Việt thông minh (xem demo)

https://github.com/telexyz/nem/tree/main/marktone#readme


## Train mô hình ngôn ngữ RWKV cho tiếng Việt

Ưu điểm train nhanh, độ chính xác ~ Transformer, inference cực nhanh (chỉ cần nhân vector với matrix), có thể triển khai trên smartphone.
Nhược điểm không phổ biến và được hỗ trợ rộng rãi như transformer

https://github.com/BlinkDL/RWKV-LM


## Tối ưu hóa xử lý ngữ liệu tiếng Việt (đang làm)

https://github.com/telexyz/bon


## Viết lại Deep Learning Framework bằng ngôn ngữ lập trình hiệu năng cao

- Python baseline https://github.com/telexyz/kim

- Có thể viết bằng https://github.com/exaloop/codon để sử dụng lại Python baseline

- Có thể viết lại bằng Zig nếu Zig hỗ trợ nvptx đủ tốt. Tập trung data-oriented và thuần perf.

- Tham khảo https://github.com/fengwang/ceras 

- - -


### Others

- [Hardware for Machine Learning](https://inst.eecs.berkeley.edu/~ee290-2)

- [TinyML and Efficient Deep Learning](https://efficientml.ai/schedule/)

- [Efficient Computing for Deep Learning](https://www.youtube.com/watch?v=WbLQqPw_n88) | 
[slides](https://www.rle.mit.edu/eems/wp-content/uploads/2020/09/2020_uwisconsin_compressed.pdf)

- - -

- https://github.com/dendibakh/perf-ninja
- https://github.com/pingcap/talent-plan
- https://github.com/codecrafters-io/build-your-own-x

- - -

- [CS61A](https://cs61a.org) Structure and Interpretation of Computer Programs
- [14002](https://github.com/courseworks) Advanced Programming Course
- [CS162](https://cs162.org) Operating Systems and Systems Programming
- [6S081](https://pdos.csail.mit.edu/6.828) Operating System Engineering

- [cs161](https://cs161.org) Computer Security
- [6.858](http://css.csail.mit.edu/6.858) Computer System Security

- [CS186](https://cs186berkeley.net) Introduction to Database Systems
- [CS122](http://courses.cms.caltech.edu/cs122) Database System Implementation
- [CS346](https://web.stanford.edu/class/cs346) Database System Implementation

_source https://csdiy.wiki_
