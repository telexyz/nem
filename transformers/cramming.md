https://twitter.com/giffmana/status/1608568387583737856

https://arxiv.org/pdf/2212.14034v1.pdf

https://github.com/jonasgeiping/cramming

## Khai thác luật mở rộng

Luật mở rộng dường như cản trở chúng ta thu hoạch lớn từ việc thay đổi kích cỡ và và loại tfm, vì hiệu năng trên mỗi token bị buộc chặt vào kích cỡ mô hình (chưa hiểu lắm). Không có tiến bộ:
- Khi dùng kiến trúc tranformer hình phễu
- Hay khi bỏ FFN layers
- Hay khi dùng recurrent layers thậm chí là huấn luyện với BPTT (backprob through time)
- Hay thay đổi quy mô kiến trúc thành deep-narrow

Khi nguyên tắc này đóng một cánh cửa để có thể scaling down hiệu quả thì nó mở ra một cách cửa khác. Bởi vì hiệu năng per-gradient gần ngữ không đổi cho mọi mô hình của cùng kích cỡ, chúng ta có thể khai thác luật mở rộng bằng cách nhanh chóng tìm kiếm một lựa chọn kiến trúc làm tăng tốc độ tính toán trong khi giữ kích cỡ mô hình không đổi. Một số những cách tối ưu rõ thấy nằm trong lớp này:

1. Attn block:
  - Tắt toàn bộ QKV bias giúp khai thác luật mở rộng bằng cách loại bọ một tầng tính toán, làm cho forward và backward pass nhanh hơn trong khi giữ cho kích thước mô hình không đổi.
  - Giảm gradient costs bằng cách giảm số lượng attn heads đồng thời tăng hiệu năng thêm một chút nhưng lại làm giảm hiệu năng finetune, vì thế chúng tôi quyết định giữ lại cả 12 heads.

2. Feedforward block:
  - Gain nhờ loại bỏ toàn bộ bias
  - Sử dụng GeLU
  - We do see small improvements from re-ordering the block into a gated linear unit
  - we do not increase the number of parameters in the FFN block to compensate for the halving of the hidden dimensionality due to gating

3. Embedding: 
  - We implement scaled sinusoidal positional embeddings, finding incremental benefits over learned or unscaled sinusoidal embeddings
  - We include a layer normalization at the end of the embedding block

4. Layer Structure: As observed in many studies, we find that pre-normalization with Layer Norms is beneficial over post Layer Norms. We note that the key effect of pre-normalization is to stabilize training and enable larger learning rates and reduced warmup, and we see limited benefits from including it by itself.

5. Head Block: We find that we can remove the nonlinear head without ill effect. We can further drop the decoder bias and gain in memory using sparse token prediction. We add a final Layer Norm to stabilize training further.

