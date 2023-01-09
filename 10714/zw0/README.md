- softmax regression cần learning_rate giảm dần sao mỗi epoc
- nn cần giá trị ma trận là `f64` nếu không sẽ bị `nan`
- nn cần nhiều dữ liệu và tốn thời gian để train hơn rất nhiều so với softmax
- softmax đạt độ chính xác 91.57% sau 05 epochs
- nn      đạt độ chính xác 95.68% sau 10 epochs

- [ ] áp dụng softmax_stable để xem có giảm xuống `f32` hoặc `f16` được ko?
- [ ] tối ưu hóa [nhân ma trận](https://en.algorithmica.org/hpc/algorithms/matmul)
