# Học

Thích gì học nấy từ deep learning, database, intepreter - compiler - programming language cho tới system programming.

## 09/2022-12/2022

- Hoàn thành `10714`, tự xây dựng thư viện học sâu `kim`

## 12/2022-03/2023

- Hoàn thành `15445`, hiểu cách xây dựng một database cơ bản

## 03/2023-06/2023

- Hoàn thành `15721`, [Advanced Database Systems](https://15721.courses.cs.cmu.edu/spring2020/schedule.html)

Database quan tâm https://github.com/tigerbeetledb/tigerbeetle/blob/main/docs/DESIGN.md: The distributed financial accounting database designed for mission critical safety and performance.

## 07/2023-xxx

Học viết trình thông dịch theo hướng dẫn trong cuốn https://craftinginterpreters.com bằng ngôn ngữ https://ziglang.org

Rồi tìm hiểu sâu hơn về trình biên dịch và LLVM qua mã nguồn Zig bắt đầu từ https://mitchellh.com/zig

### Tại sao chọn Zig?
Zig là một ngôn ngữ lập trình hệ thống hiện đại, đơn giản và mạnh mẽ. Có thể học syntax trong buổi sáng và chiều làm dự án luôn. Zig mới và chưa ra bản 1.0 (có lẽ vào cuối 2023) nhưng đã có 2 startups thành công sử dụng Zig (Bun.js và TigerBeetle). Cộng đồng rất mạnh.

Có thể tham gia sâu hơn và Zig bằng cách:

- Chữa những lỗi đơn giản như một người mới
- Cải tiến nvptx backend chạy mượt [zcuda](https://github.com/gwenzek/cudaz) (xem [video](https://www.youtube.com/watch?v=rvfsWm6TckA&t=5351s))
- ...

## xxx-xxx

Tìm hiểu deep learning compiling có thể là Triton, MLIR, ...


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

- [Hardware for Machine Learning](https://inst.eecs.berkeley.edu/~ee290-2/sp21)

- [TinyML and Efficient Deep Learning](https://efficientml.ai/schedule/)

- [Efficient Computing for Deep Learning](https://www.youtube.com/watch?v=WbLQqPw_n88) | 
[slides](https://www.rle.mit.edu/eems/wp-content/uploads/2020/09/2020_uwisconsin_compressed.pdf)

- - -

#### CS Self-study Guide https://csdiy.wiki

- [CS61A](https://cs61a.org) Structure and Interpretation of Computer Programs

- [14002](https://github.com/courseworks) Advanced Programming Course

- [CS162](https://cs162.org) Operating Systems and Systems Programming
- [6S081](https://pdos.csail.mit.edu/6.828) Operating System Engineering

- [15418](http://15418.courses.cs.cmu.edu) Parallel Computer Architecture and Programming
- [CS149](https://gfxcourses.stanford.edu/cs149) Parallel Computing
- [6.824](https://pdos.csail.mit.edu/6.824) Distributed System

- [cs161](https://cs161.org) Computer Security
- [6.858](http://css.csail.mit.edu/6.858) Computer System Security

- [CS186](https://cs186berkeley.net) Introduction to Database Systems
- [CS122](http://courses.cms.caltech.edu/cs122) Database System Implementation
- [CS346](https://web.stanford.edu/class/cs346) Database System Implementation


> Professor Kayvon Fatahalian previously opened the [15418](http://15418.courses.cs.cmu.edu/spring2016) course at CMU, and later he opened a similar course [cs149](https://gfxcourses.stanford.edu/cs149/fall21) after he became an assistant professor at Stanford. But in general, 15-418 contains richer course content and has course replays, but CS149's programming assignments are more fashionable. Personally, I watched the 15-418 course video but completed the CS149 homework.

> If you can build the entire CS basic building in less than three years, have a relatively solid mathematical foundation and coding ability, have experienced the baptism of dozens of projects with thousands of lines of code, and master at least C /C++/Java/JS/Python/Go/Rust and other mainstream languages, for algorithms, circuits, systems, networks, operating systems, compilation, artificial intelligence, machine learning, computer vision, natural language processing, reinforcement learning, cryptography, information theory , Game theory, numerical analysis, statistics, distribution, database, graphics, Web development, cloud services, supercomputing, etc. have been involved. I think you will have enough confidence and confidence to choose the direction you are interested in. Whether it is employment or scientific research, you will be quite competitive.

> The biggest advantage of self-study is that you can adjust the learning speed completely according to your own progress. For some difficult knowledge points, I can watch the video repeatedly, Google related content on the Internet, and _ask questions on StackOverflow_ until I fully understand it. As for the content that you have mastered relatively quickly, you can skip it at double or even triple speed.


- - -


- [System programming](https://www.cs.cornell.edu/courses/cs4414/2021fa/Schedule.htm)

- [Performance Ninja Class](https://github.com/dendibakh/perf-ninja)

- [Open source training courses about distributed database and distributed systems](https://github.com/pingcap/talent-plan)

- https://github.com/codecrafters-io/build-your-own-x
	* https://github.com/codecrafters-io/build-your-own-sqlite
	* https://github.com/codecrafters-io/sqlite-starter-rust
