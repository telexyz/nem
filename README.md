# Học

Thích gì học nấy deep learning, database, intepreter - compiler - programming language, system programming, parallel and distributed programming.

## 09/2022-12/2022

- [x] Hoàn thành [10714](https://dlsyscourse.org) Deep Learning Systems, tự xây dựng thư viện học sâu [kim](https://github.com/telexyz/kim)

- [ ] Học chung với [11785](https://deeplearning.cs.cmu.edu) Introduction to Deep Learning bổ trợ kiến thức rất tốt, nhất là với ai chưa biết nhiều về Deep Learning.

## 12/2022-06/2023

- [ ] Hoàn thành [15445](https://15445.courses.cs.cmu.edu) Database Systems
, hiểu cách xây dựng một database cơ bản

- [ ] Hoàn thành [15721](https://15721.courses.cs.cmu.edu) Advanced Database Systems

Database quan tâm https://github.com/tigerbeetledb/tigerbeetle/blob/main/docs/DESIGN.md: The distributed financial accounting database designed for mission critical safety and performance.

### Nice to learn

- [ ] [15418](http://15418.courses.cs.cmu.edu/spring2016)+[cs149](https://gfxcourses.stanford.edu/cs149) Parallel Computing

- [ ] [CS839](https://pages.cs.wisc.edu/~yxy/cs839-s20) Design the Next-Generation Database

- [ ] [CS764](https://pages.cs.wisc.edu/~yxy/cs764-f22) Topics in Database Management Systems

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

Học lập trình hệ thống bằng cách tham gia https://github.com/SerenityOS (C++ hoặc [Jakt](https://github.com/SerenityOS/jakt))

- Viết bộ gõ tiếng Việt cho Serenity
- ...

- - -


# Làm (dự án)

Cần kiến thức lập trình hệ thống, toán, cấu trúc dữ liệu và giải thuật, kiến trúc máy tính cơ bản (phân cấp bộ nhớ, cache, nhân xử lý ...), lập trình hướng dữ liệu, lập trình song song / phân tán, lập trình hiệu năng cao.


## Dùng GPU xử lý dữ liệu lớn (ý tưởng)

Bất kỳ dữ liệu nào, áp dụng hiểu biết về lập trình GPU và database. Có thể liên quan tới thời gian thực và tài chính.


## Bộ gõ tiếng Việt thông minh (xem demo)

https://github.com/telexyz/nem/tree/main/marktone#readme


## Train mô hình ngôn ngữ RWKV cho tiếng Việt

Ưu điểm train nhanh, độ chính xác ~ Transformer, inference cực nhanh (chỉ cần nhân vector với matrix), có thể triển khai trên smartphone. Nhược điểm không phổ biến và được hỗ trợ rộng rãi như transformer

https://github.com/BlinkDL/RWKV-LM


## Tối ưu hóa xử lý ngữ liệu tiếng Việt (đang làm)

https://github.com/telexyz/bon


## Viết lại Deep Learning Framework bằng ngôn ngữ lập trình hiệu năng cao

- Python baseline https://github.com/telexyz/kim

- Có thể viết bằng https://github.com/exaloop/codon để sử dụng lại Python baseline

- Có thể viết lại bằng Zig nếu Zig hỗ trợ nvptx đủ tốt. Tập trung data-oriented và thuần perf.

- Tham khảo https://github.com/fengwang/ceras 

- - -


### Other Interests

- [ ] [Practical Data Science](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22912b80a3-625d-405d-8905-a8620133666b%22)

- [Hardware for Machine Learning](https://inst.eecs.berkeley.edu/~ee290-2)

- [TinyML and Efficient Deep Learning](https://efficientml.ai/schedule/)

- [Efficient Computing for Deep Learning](https://www.youtube.com/watch?v=WbLQqPw_n88) | 
[slides](https://www.rle.mit.edu/eems/wp-content/uploads/2020/09/2020_uwisconsin_compressed.pdf)

- [Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html)

- - -

- https://github.com/dendibakh/perf-ninja
- https://github.com/pingcap/talent-plan
- https://github.com/codecrafters-io/build-your-own-x
- [Making a RISC-V Operating System using Rust](https://osblog.stephenmarz.com)
- [More challenging projects every programmer should try](https://austinhenley.com/blog/morechallengingprojects.html)

- - -

- [CS61A](https://cs61a.org) Structure and Interpretation of Computer Programs
- [14002](https://github.com/courseworks) Advanced Programming Course

- [cs61c](https://inst.eecs.berkeley.edu/~cs61c/fa20/#lectures) Great Ideas in Computer Architecture (fun)
- [cs4414](https://www.cs.cornell.edu/courses/cs4414) System programming

- [CS162](https://cs162.org) Operating Systems and Systems Programming
- [6S081](https://pdos.csail.mit.edu/6.828) Operating System Engineering

- [cs161](https://cs161.org) Computer Security
- [6.858](http://css.csail.mit.edu/6.858) Computer System Security

- [CS186](https://cs186berkeley.net) Introduction to Database Systems
- [6.5830](http://dsg.csail.mit.edu/6.5830/assign.php) Database Systems
- [CS122](http://courses.cms.caltech.edu/cs122) Database System Implementation
- [CS346](https://web.stanford.edu/class/cs346) Database System Implementation
  
- [6.824](https://pdos.csail.mit.edu/6.824) Distributed System ([more](https://www.youtube.com/watch?v=rZPRjLMWOao&list=PLNPUF5QyWU8PydLG2cIJrCvnn5I_exhYx))
