# Học

Thích gì học nấy deep learning, database, intepreter - compiler - programming language, system programming, parallel and distributed programming.

## 09/2022-01/2023

- [x] Hoàn thành [10714](https://dlsyscourse.org) Deep Learning Systems, tự xây dựng thư viện học sâu [kim](https://github.com/telexyz/kim)

## 01/2023-06/2023

- [ ] [cs234](https://stanford-cs324.github.io/winter2023/syllabus) Foundation Models

- [ ] [cs25](https://web.stanford.edu/class/cs25) Transformers United V2

> AI / Deep learning phát triển quá nhanh. Ý tưởng sau 6 tháng không thực hiện đã trở nên lạc hậu. Big foundation models + fine-tune (a.k.a foundation models) trở thành xu hướng tất yếu. Cần nắm được xu hướng này nhất là Large Language Model.

- - -

## 01/2023-06/2023

- [ ] [15721](https://15721.courses.cs.cmu.edu) Advanced Database Systems

__Database quan tâm__:

- https://github.com/tigerbeetledb/tigerbeetle/blob/main/docs/DESIGN.md: The distributed financial accounting database designed for mission critical safety and performance.

- https://github.com/unum-cloud/UKV Universal Keys & Values, Unum tận dụng thế mạnh phần cứng hiện đại để lưu trữ và xử lý dữ liệu đơn giản [hiệu quả hơn nhiều](https://unum.cloud/post/2022-09-13-ucsb-10tb)

![](https://github.com/unum-cloud/UKV/raw/main/assets/UKV.png)

### Nice to learn

- [ ] [15418](http://15418.courses.cs.cmu.edu/spring2016)+[cs149](https://gfxcourses.stanford.edu/cs149) Parallel Computing

- [ ] [CS839](https://pages.cs.wisc.edu/~yxy/cs839-s20) Design the Next-Generation Database

- [ ] [CS764](https://pages.cs.wisc.edu/~yxy/cs764-f22) Topics in Database Management Systems

- [ ] [15445](https://15445.courses.cs.cmu.edu) Database Systems, hiểu cách xây dựng một database cơ bản

## xxx-xxx

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


## 1/ Dùng GPU xử lý dữ liệu lớn

Bất kỳ dữ liệu nào, áp dụng hiểu biết về lập trình GPU và database. Có thể liên quan tới thời gian thực và tài chính.


## 2/ Bộ gõ tiếng Việt thông minh (xem demo)

https://github.com/telexyz/nem/tree/main/marktone#readme


## 3/ Train mô hình ngôn ngữ RWKV cho tiếng Việt

Ưu điểm train nhanh, độ chính xác ~ Transformer, inference cực nhanh (chỉ cần nhân vector với matrix), có thể triển khai trên smartphone. Nhược điểm không phổ biến và được hỗ trợ rộng rãi như transformer

https://github.com/BlinkDL/RWKV-LM


## 4/ Tối ưu hóa xử lý ngữ liệu tiếng Việt (đang làm)

https://github.com/telexyz/bon


## 5/ Viết Deep Learning Framework hiệu năng cao

- Python baseline https://github.com/telexyz/kim

- Có thể viết bằng https://github.com/exaloop/codon để sử dụng lại Python baseline

- Có thể viết lại bằng Zig nếu Zig, tập trung multi-CPUs + data-oriented và thuần perf.

- Tham khảo https://github.com/fengwang/ceras 

## 6/ Tiny projects

- https://github.com/karpathy/nanoGPT

- [bert-one-day-one-gpu.md](./tiny-projects/bert-one-day-one-gpu.md)

- [lsm-in-a-week.md](./tiny-projects/lsm-in-a-week.md)
