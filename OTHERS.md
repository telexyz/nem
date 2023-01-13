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

# Other projects

## Train mô hình ngôn ngữ RWKV cho tiếng Việt

Ưu điểm train nhanh, độ chính xác ~ Transformer, inference cực nhanh (chỉ cần nhân vector với matrix), có thể triển khai trên smartphone. Nhược điểm không phổ biến và được hỗ trợ rộng rãi như transformer

https://github.com/BlinkDL/RWKV-LM

## Viết Deep Learning Framework hiệu năng cao

- Python baseline https://github.com/telexyz/kim

- Có thể viết bằng https://github.com/exaloop/codon để sử dụng lại Python baseline

- Có thể viết lại bằng Zig nếu Zig, tập trung multi-CPUs + data-oriented và thuần perf.

- Tham khảo https://github.com/fengwang/ceras 

## Tiny projects

- [lsm-in-a-week.md](./tiny-projects/lsm-in-a-week.md)

# Other Interests

- [ ] [6.S898](https://phillipi.github.io/6.s898/#schedule) Deep Learning

- [ ] [11785](https://deeplearning.cs.cmu.edu) Introduction to Deep Learning

- [ ] [CS246](http://web.stanford.edu/class/cs246) Mining Massive Data Sets [videos](https://www.youtube.com/watch?v=jofiaetm5bY&list=PLoCMsyE1cvdVnCgHk43vRy7PVTVWJ6WVR)

- [ ] [AIR](https://github.com/sebastian-hofstaetter/teaching#lectures) Advanced Information Retrieval

- [ ] [15388](http://www.datasciencecourse.org/lectures) Practical Data Science [2019-videos](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22618ea253-ca45-4b14-9f1d-aab501543bd2%22) | [2018-videos](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22912b80a3-625d-405d-8905-a8620133666b%22)

- [ ] [6.S965](https://efficientml.ai/schedule) TinyML and Efficient Deep Learning

- [ ] [Deep Learning in Practice](https://www.youtube.com/playlist?list=PLVSIY7rG0A3e2OU8oqM1ASyecPON8s8E1)

- [ ] [Machine Learning Compilation](https://mlc.ai/summer22)

- [ ] [cs285](http://rail.eecs.berkeley.edu/deeprlcourse) Deep Reinforcement Learning

- [ ] [NPFL122](https://ufal.mff.cuni.cz/courses/npfl122) Deep Reinforcement Learning

- [Hardware for Machine Learning](https://inst.eecs.berkeley.edu/~ee290-2)

- [Efficient Computing for Deep Learning](https://www.youtube.com/watch?v=WbLQqPw_n88) | 
[slides](https://www.rle.mit.edu/eems/wp-content/uploads/2020/09/2020_uwisconsin_compressed.pdf)

- [Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html)

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
