![](files/transformer.png)

# Transformer implementation
- 10714 https://www.youtube.com/watch?v=OzFmKdAHJn0
- 11785 https://www.youtube.com/watch?v=X2nUH6fXfbc

- - -

# Transformer
10714 https://www.youtube.com/watch?v=IFKRf-BAqZo&t=1479s
- 2 cách tiếp cận để mô hình hóa chuỗi thời gian (time series)
- tự chú ý (self-attn) và Transformers
- transformers ứng dụng khác ngoài chuỗi thời gian

## 2 cách tiếp cận time series: lattent state approach
![](files/tfm-00.jpg)

Ta có thể mô hình hóa bài toán mô hình hóa chuỗi thời gian bằng cách dự đoán `y_1:T = f_theta(x_1:T)` với y_t chỉ phụ thuộc vào x_1:t. 
f_theta có thể là bất cứ hàm nào miễn là nó giúp chúng ta dự đoán output từ input. Có nhiều cách để làm được điều đó, có thể không dùng tới biểu diễn latent state của RNNs.

__bản chất của RNN là "latten state" approach__ chúng lấy chuỗi đầu vào tới thời điểm x_t chẳng hạn rồi tổng kết / nhúng toàn bộ chuỗi đó vào một latent state h_t. Và trạng thái h_t đó summarize all information chúng ta quan sát được cho tới thời điểm t đó. RNN sẽ biểu diễn toàn bộ chuỗi thời gian (tới thời điểm t) bằng một latent (không quan sát được trực tiếp) state h.
- pros: RNN có thể capture infinite history, và có compact representation
- cons: compute path quá dài từ time bắt đầu cho tới current => vanishing / exploding gradients, hard to learn

__Điều này khiến latent state aproarch works in therory nhưng trong thực tế sẽ có vấn đề__

## 2 cách tiếp cận time series: direct prediction approach

![](files/tfm-01.jpg)

- - -

# "Phillip Isola" on transformer
- video https://www.youtube.com/watch?v=Smav86u60FM
- notes https://phillipi.github.io/6.s898/materials/notes/09_transformers.pdf
- slide https://phillipi.github.io/6.s898/materials/slides/9_transformers.pdf

2 ý tưởng chính về transformers là __Tokens và Self-Attention__. Và thêm vài thứ khác :) 

Trong AI, nơ-ron là số thực, just scalar numbers, và tokens là vectors. Đó là sự khác biệt mà tôi concerned.
Như vậy __token__ chỉ là lingo cho __a vector of neurons__. Mà nơ-ron là scalar number =>
__token__ đơn giản là __vector__: kiểu dữ liệu 1D array of scalar numbers.

- Trong NLP thì token khởi đầu bằng one-hot vectors.
- Trong Vision token là patch, tức là ta duỗi điểm ảnh của một ô đơn vị ảnh từ 2D array ra 1D array

## Tokenizing input data

Bạn có thể biến mọi data của mình thành tokens.

![](files/tfm-02.jpg)

Mục tiêu của việc tokenization là biến data thành dữ liệu dạng chuỗi - a sequence of vectors. Transformer thực sự treat tokens như là một set, nhưng mọi người thường nói về nó như là một chuỗi (đó là lý do transformer cần thêm positional embedding để đánh dấu thứ tự của token trong chuỗi).

![](files/tfm-03.jpg)

Trong mạng nơ-ron nhân tạo tiêu chuẩn thường có 2 tầng chính là tầng linear combination hay tầng linear, và tầng thứ 2 là pointwise (đầu vào là từng điểm 1 trên input tensor) nonlinearity, có thể là giá trị của hàm sigmoid.

![](files/tfm-04.jpg)

Tương tự như vậy khi deal with tokens, ta có khái niệm token-wise nonlinearity, 
`t_out = [F_theta(t_1.z),..., F_theta(t_N.z)]` với F thường là MLP.
Khái niệm này tương đương với 1 CNN với 1x1 kernels, chạy trên chuỗi tokens đầu vào.

=> Transformer and CNN, it's all the same idea of rehashed.

```
Q: z ở đây là gì?
A: z is the token code vector, meaning the vector of neurons inside that token. 
`token.z` ở đây hàm ý z là thuộc tính con của token, ám chỉ this code token that lives inside it.

Q: analogy to convolution here là gì?
A: conv apply the same operator tới một phần tử của chuỗi. Và minh họa ở trên làm điều tương tự. So it's like slide a (non-linear) filter across the sequence, vì thế có thể nghĩ nó như là 1x1 kernels
```

![](files/tfm-05.jpg)

Ở hình minh họa trên hàm token-wise nonlinear F_theta là biến đổi từng token vector input thành token vector output. Nó giống như cửa sổ trượt của CNN. 

## Token nets

![](files/tfm-06.jpg)

![](files/tfm-07.jpg)

Hóa ra GNN cũng có cùng cấu trúc với Transformer.
__token nets, graph nets, transformers are all the same thing__ 

Bạn có thể coi transformer là một dạng đặc biệt của graph nets, nếu bạn thích nhìn theo cách đó hơn.

## Ý tưởng thứ 2: attention

Như vậy ý tưởng đầu tiên là build things out of token - vector-valued units, thay vì nơ-ron - scalar-valued units. Ý tưởng thứ 2 làm nên tên tuổi cho transformer là attention, có hẳn 1 layer gọi là attention trong transformers.

![](files/tfm-08.jpg)

- trong linear bạn có một weighted combination với W là free parameters và có thể được cập nhật (trong quá trình huấn luyện)
- trong Attn bạn cũng có một weighted combination của inputs để tạo ra output, nhưng các tham số - giá trị của ma trận A không học được (không được cập nhật), thay vào đó nó sẽ phụ thuộc vào dữ liệu (data-dependent), nó sẽ là một hàm của một loại dữ liệu khác.

So when you have data-dependent linear combination where the weights are functions of some other data, then that's called attention. A is a weight matrix that's a function of something else, it tells you how much weight to apply to each token in the input sequence, you'll take a weighted sum, which is just matrix multiply A by the tokens. 


## Intuition of attn https://youtu.be/Smav86u60FM?t=656

![](files/tfm-09.jpg)

Ta có thể có attn give me by some other branch of my NN. Nó có thể là một nhánh NN để phân tích câu hỏi. Và câu hỏi đó sẽ cho ta biết patches nào của ảnh đầu vào ta nên chú ý đến. Các patches đầu vào được biểu diễn bởi tokens. Và ta có thể nói đây là các patches tôi sẽ cho trọng số cao hơn, và sau đó tôi sẽ tính weight sum of the code vectors inside those patches to produce an output.

Ta có thể đặt câu hỏi "màu sắc của đầu con chim trong hình là gì?", tức là ta sẽ chú ý tới đầu chim. Hoặc "màu sắc của cỏ là gì?" tức là ta sẽ chú ý tới nền cỏ. So it's just saying which tokens or patches there are going to be getting a lot of weight to make my decision. And then we'll report best green because the token code vectors will have represented some information like the color of the token.

## Query-key-value attention https://youtu.be/Smav86u60FM?t=717

![](files/tfm-10.jpg)

Một attention điển hình sẽ như hình trên, nó cũng làm nổi rõ notation `token.z` để thể hiện vector live in side a token. Ý muốn nói là trong token chứa nhiều loại thông tin:
- `token.z` là giá trị vector của nó
- `token.query() = W_q token.z`
- `token.key()   = W_k token.z`
- `token.value() = W_v token.z`

Ta có thể thấy .query(), .key(), .token() là 3 hàm f_q(t), f_k(t) và f_v(t). Và q, k, v là 3 giá trì đầu ra khi áp dụng 3 hàm đó vào 1 token t nhất định. Ở ví dụ trên W_q, W_k, W_v là 3 ma trận trọng số có thể học được - là triển khai cụ thể của cơ chế attn.

Tổng quát hơn nữa ta thấy với transformers có thể quy mọi thứ về token. t_question cho câu hỏi, t_input cho dữ liệu cần tìm câu trả lời trên nó rồi t_out cho câu trả lời đầu ra (input là t_question và t_in, output là t_out).

Diễn giải bằng lời: this is like attn from the question branch, but i could __just have the data choose what in its own__ token sequence to attent to. Each token chooses which other tokens to attent to, and that's called self-attention.

![](files/tfm-11.jpg)
This picture here where this patch sẽ quyết định, cái gì sẽ được chú ý đến để có được một biểu diễn tốt hơn của that patch, essentially.

![](files/tfm-12.jpg)
Hàm f ám chỉ sự phụ thuộc của weight vào data. Và vì f không đến từ bên ngoài nên cơ chế này được gọi là self-attention (no external source).

## Demo https://youtu.be/Smav86u60FM?t=945
(xem video để thấy chuyển động của mô hình)

To connect it to one of the most common demonstrations of transformers is to do sequence modeling. Nhưng thực sự transformers are more about set to set operations. Bởi bạn có thể biểu diễn seuquence như là một tập hợp, so it's fine.

![](files/tfm-13.jpg)
Ở ví dụ trên, từ chuỗi đầu vào "colorless green ideas sleep", mô hình dự đoán token tiếp theo là "furiously", và token này lại nối tiếp với chuỗi đầu đang có trở thành chuỗi đầu vào tiếp theo "colorless green ideas sleep furiously" để tiếp tục dự đoán. Và thay vì dự đoán từ tiếp theo, ở các bài toán khác bạn có thể dự đoán protein tiếp theo, hay là sound wave tiếp theo ...

```Q&A
Q: lịch sử của term key, query, value là gì?
A: tôi nghĩ nó đến từ data retriveval database. Bạn có database lưu trữ tri thức trong các cells. Và bạn có thể query database đó. Bạn nói tôi muốn tìm những thứ liên quan tới "drafts". Và mọi cell đều có nhãn gọi là key, ví dụ như đây là cell cho mammal, đây là cell cho giraffes, và đây là cell cho plants. Và bạn match query vào key. Và những thứ có trong cell mà bạn thu về được gọi là value.

Q: giả sử câu hỏi là "đâu là màu sắc của đầu con gà tây", và vì một vài lý do nào đó, đầu gà tây là một đặc trưng đủ phức tạp mà nó không thể được biểu diễn tốt trong một token, điều đó có gây ra vấn đề gì cho việc attending to the right things, or weighting the right thing properly?
A: Một trong những câu trả lời là chúng ta có multi-head attention, nơi mà bạn sẽ lấy sequence của mình hay your set of tokens và bạn sẽ có n query vectors khác nhau, n value vectors khác nhau, và n key vectors khác nhau và mỗi query có thể hỏi một kiểu khác nhau kiểu như: một query sẽ hỏi tôi muốn match the same color, query khác hỏi tôi muốn match the same geometry. Và khi bạn tối ưu các tham số của mạng, bằng một cách nào đó chúng sẽ tự tổ chức, well, if it's useful to factor things into geometry and color, thì sẽ có một attention head take cares about color, and one that cares about geometry.

Q: Tôi có câu hỏi về phi tuyến tính trong transformer. Nếu tôi nhớ không nhầm thì bạn sẽ chuẩn hóa cho mỗi token, intution khi làm việc đó là gì? Tôi không thấy người ta làm việc đó nhiều trước khi transformer xuất hiện.
A: Đó là một câu hỏi tuyệt vời. Thông thường bạn sẽ weighted sum to add up to one, so the weight should add up to one. Và chúng ta làm việc đó thông qua softmax over the weights.

Q: Ồ, tôi muốn hỏi về residual connection cơ.
A: À đó là layernorm. Theo hiểu biết của mình, tôi thấy laynorm, residual connection chỉ là mẹo để giúp network có thể train. It's good to normalize things.
Brian Cheung: Transformers rất khó để train, đặc biệt là một vài tầng, như là tầng thứ 12 trở lên. Nếu bạn không có residual connections hay layernorms.

Q: Làm thế nào để quyết định dùng patch nào? Ví dụ trong ảnh hoặc text you also had what seemed to be arbitrary segmentation of data.
A: Đó là một câu hỏi tuyệt vời. Like how do you do the tokenization? Like how do you design that. I think it's super hacky right now. So that feel like somewhere that people could do a lot of work. One thing does seem to happen is that __the smaller you make your token, if you have token that are a single pixel, it's just better you get__. So maybe what happen is just stop having clever tokenization and just go down to whatever atomic units of the data are, like a single character, a single pixel, and that just makes the choice for us because you can't go below that, really. Vì thế tôi nghĩ (tokens) càng nhỏ hơn trong ViT, thing i've seen is that smaller the tokens are, the better they tend to perform. Bởi vì do cơ chế attn là every tokens attends to every token, nó là n^2, nên nếu bạn chọn tokens quá nhỏ, có quá nhiều tokens bạn sẽ bị run out of memory. Vì thế có những cách tknz thông minh như superpixels, hay segmentation. Hay như trong NLP, có rất nhiều cách tknz như bype-pair-encoding, bạn có thể dùng chúng như là tầng tknz đầu tiên. But yeah, that feels like a hackey area right now to me.

Q: Can you try to usually leverage at least some degree of topology by tokenizing it, that respects spatial coordinates for images, or language order for words. Because there's also smt called position encoding, which give you knowledge about the topology of why this element is in this position, versus this other position, and that tell you a lot about the structure of the image, or at least where this oken is with respect to the overall structure of the image.
A: Yeah, và một thứ nữa có vẻ missing right now is people usually envision transformers which i'm most familiar with, that usually break into non-overlapping tokens. But we know from ConvNets and signal processing that you'll get a ton of aliasing in the filtering operations if you have these huge strides, you're bringing in non-overlaping patches, you really want to have overlapping patches, or blur the signal. So all that signal processing stuff I think has been thrown away, and probably it's a good idea to put it back in there.

someone has tried overlapping patches, and it does help. Everythin i say, as you probably know, there's 10k transformer papers now. So i'm sure everthing that you could imagine has been tried.  
```

## Positional encoding

![](files/tfm-14.jpg)

Nếu bạn không muốn bị shift invariant. Một là bạn dùng một kiến trúc không bị shift invariant như MLP. Hai là bạn cho thêm thông tin về vị trí vào input của convolution filters, cách này gọi là positional encoding. Ý tưởng này đã có từ trước.

![](files/tfm-15.jpg)

Nếu bạn không muốn to be permutation invariant. Một là bạn sử dụng kiến trúc không bị permutation invariant như MLP. Hai là bạn cho thêm thông tin vị trí vào token code vectors, đó gọi là positional encoding. Và khi làm điều đó bạn được một sequence model. Nếu không làm thế bạn sẽ có property of permuatition equivariance (tương đương hoán vị). This is why I said it's really a set-to-set operation. Vì thế nếu bạn không cho token thông tin nó đến từ đâu, bạn không cho nó positional codes, và khi ta lấy tokens từ input layer, và ta giao hoán chúng theo bất cứ thứ tự nào bạn chỉ đơn giản đang hoán đổi tokens của layer đầu ra. The mapping will be the same up to permutation. Sẽ cần suy nghĩ một chút để thấy điều này là đúng, nhưng lý do cốt lõi là bởi vì attention layer is permutation invariant bởi vì cái cách mà attn hoạt động. Và token-wise (tương đương point-wise) layer is permutation invariant too, so transformer is a permutation invariant function (hàm bất biến hoán vị).

![](files/tfm-16.jpg)

Như vậy nếu có positional encoding, thì nó sẽ là sequence. Còn không transformers sẽ là set to set mapping.

Q: Bạn bắt đầu với ý tưởng về token như một loại cấu trúc dữ liệu mới, but just a few sencences before, you said smt that may sounds like the idea of tokens is rather weakened that the transformers work better when you have individual bits of data, like pixels. Vì thế tôi càm thấy ý tưởng về token không phải là cốt lõi của toàn bộ ý tưởng này, mà attn mới là phần quan trọng nhất. Và ý tưởng có thể attend dynamically anywhere in the picture là ý tưởng gives power in these algorithms, chứ không phải là tokens?
A: Tôi không biết. Và tôi nghĩ rằng that's kind of the open debate. It's fun to keep discussing it. So I think the field is kind of split right now between people thinking that it's the tokens and the vector, encapsulated vectors that are important to this versus the attention, and really if you just have intentional mechanism, you can still be operating over neurons. It doesn't have to be about tokens. Tôi nghĩ rằng cả 2 đều quan trọng để thành công. Một ví dụ phản công là nếu attention matter is there are these other architectures now, sometimes called MLP-mixer, nó sử dụng token chứ không phải attention. Cơ bản MLP-Mixer là ConvNet over tokens, one-by-one conv over tokens, họ gọi nó là MLP-mixer khiến nó trở nên khó hiểu, but anyway MLP-Mixer không có attn. But it's still, in my view, it's a token net, and that seems competitive on some tasks with attention networks. So maybe attention is, maybe it's not.

- - -

# Attention and Transformer
- 11785 https://youtu.be/amqFoES3c1w?t=663

## Q, K, V in Attention

Database Analogy
- Q(uery) is query you typed in to get the data
- K(eys) point to where information is
- V(alues) are information itself

=>

- `Query` is what **pays attention**: what looking for where I get information from?
- `Values` are what are **paid attention to**
- `Keys` helps queries figure out **how much attention to pay** to each value

- `Attention Weights` how much attention to pay

![](files/qkv0.png)
The red guy is the current token, it wonders how do i adapt to my context? And all the bule guys are there to help. They said "hey, this is me, here is my information!"

![](files/qkv1.png)
The red guy goal is to get a contextualized representation. And the guy think: "may be I can do a weighted sum? May be I can use all of this information and do a weighted some to get my contextualized representation."

![](files/qkv2.png)

![](files/qkv3.png)
Use enery function $e_ij =  q_i^T k_j$, then softmax to normalize 

![](files/qkv4.png)


![](files/qkv5.png)
`Query` is the guy are wondering "How do I contextualize my own represention?".
`Keys` are all other guys. They said here is my information, try to use it to contextualize your represention!

![](files/qkv6.png)

![](files/qkv7.png)
What we use to multiple with attenion weights are called `Values`. The weighted sum to get contextualized representation.

![](files/qkv8.png)
`h` is a vector representation of token, it can be one-hot or anything ...
W_Q, W_K, W_V are just matrix to do linear projection to other dimentions.

![](files/qkv9.png)
We take one query, one key and calculate energy score or one attention score. And do 
the same for other keys. And get scores for each of (key) tokens. We pass everything to the softmax. And we have `attention weights`.

> `attention weight` $alpha_m,n$ is how important token n to token m's contextual meaning?

![](files/qkv10.png)
Now we take all of the `values` (h @ W_v), multiple them with each of attention weights that we found, take a weighted some to get the output o (contextualized representation).

## Scaled-dot attention
https://youtu.be/amqFoES3c1w?t=1054

![](files/scaled-dot-attention0.png)

![](files/scaled-dot-attention1.png)
Scaled mean divide by dimensionality of the keys. This is done before we do softmax. 
Why we do it? => To keep values numberically stable (not too extreme ...)

![](files/scaled-dot-attention2.png)
dot product are enery function, the function we use to score each of the key using a query.

![](files/scaled-dot-attention3.png)
3 is dimensionality of the key.

## Self-attention
![](files/self-attention-single-head.png)

![](files/self-attention-multiple-head.png)

## Attention is all we need
![](files/attention-is-all-we-need.png)
