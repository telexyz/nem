https://youtu.be/8HwHGGb1zpQ

https://people.cs.umass.edu/~miyyer/cs685_f22/slides/prompt_learning.pdf

# What does all this scaling buy us?
https://youtu.be/8HwHGGb1zpQ?t=348

- - -

https://www.youtube.com/watch?v=5ef83Wljm-M

https://phontron.com/class/anlp2022/assets/slides/anlp-09-prompting.pdf

Survey https://arxiv.org/pdf/2107.13586.pdf

# Prompting + Sequence-to-sequence Pre-training
- Prompting Methods
- Sequence-to-sequence Pre-training
- Prompt Engineering
- Answer Engineering
- Multi-prompt Learning
- Prompt-aware Training Methods

## Objective engineering
Paradigm: Pre-train, Fine-tune (2017-now)
- Use pre-train LMs as initialization
- Less work on architecture design, but __engineer objective functions__

## Prompt engineering
Paradigm: Pre-train, Prompt, Predict (2019-now)
- NLP tasks are modeled __entirely relying on LMs__
- The tasks of shallow and deep feature extraction, and prediction of the data are __all given to the LM__
- Engineering of prompts is required

## Prompting là gì?
Khuyến khích một mô hình được đào tạo trước để đưa ra những dự đoán cụ thể bằng cách
cung cấp một 'dấu nhắc' chỉ định nhiệm vụ sẽ được thực hiện.
![](files/12-00.jpg)

## Prompting workflow
- Prompt Addition
- Answer Prediction
- Answer-Label Mapping

![](files/12-01.jpg)

Từ prompt đầu vào x, ta dùng template để biến đổi nó thành prompting (query context).
Sau đó mô hình đưa ra dự đoán, và ta "mapping" dự đoán đó ra kết quả cuối cùng (label).

## Types of prompts
- __Prompt__: I love this movie. Overall it was a [z] movie
- __Filled Prompt__: I love this movie. Overall it was a boring movie
- __Answered Prompt__: I love this movie. Overall it was a fantastic movie
- __Prefix Prompt__: I love this movie. Overall this movie is [z]
- __Cloze Prompt__: I love this movie. Overall it was a [z] movie

# Design Considerations for Prompting
- Pre-trained Model Choice
- Prompt Engineering
- Answer Engineering
- Expanding the Paradigm
- Prompt-based Training Strategies (very interesting)

Có rất nhiều lựa chọn với prompt-based !!
![](files/12-02.jpg)

# Pre-trained Language Models
![](files/12-07.jpg)

- __Autogressive LM__: left-to-right, suitable to highly larger-scale LMs, use prefix prompts
- __Masked LM__: bidirectional, Suitable for NLU tasks, use cloze prompt
- __Prefix LM__: a combination of masked and left-to-right, 
- __Encoder-decoder__: translate, summarize, info extract, q&a

![](files/12-03.jpg)

![](files/12-04.jpg)

![](files/12-05.jpg)

![](files/12-06.jpg)

### MASS, Bart, mBart, UniLM, T5 (Deberta?)
https://youtu.be/5ef83Wljm-M?t=1243

### Application of Prefix LM/Encoder-Decoders in Prompting
- Conditional Text Generation
  - Translation
  - Text Summarization (T5)

- Generation-like Tasks
  - Information Extraction
  - Question Answering

- - -

# Prompt Engineering
![](files/12-08.jpg)

## Traditional Formulation V.S Prompt Formulation
![](files/12-09.jpg)

![](files/12-10.jpg)

![](files/12-11.jpg)

![](files/12-12.jpg)

![](files/12-13.jpg)
Tự động tìm kiếm một tổ hợp các keywords tốt nhất từ inputs. 
Cách này tốt hơn các cách domain-based ở trên.

![](files/12-14.jpg)
- Tại sao ta không trực tiếp tối ưu hóa tham số thay vì chỉ đổi keywords?
- Prompt-tuning chỉ tối ưu hóa tầng embedding!
- Prefix-tuning thay vì chỉ tối ưu hóa tầng embedding, nó tuning a prefix that you append to every layer of the model.

## DEMO :D
https://demo.allennlp.org/masked-lm

# Answer Engineering

Why do we need answer engineering?

We have reformulate the task! 
=> We also should re-define the “ground truth labels”

![](files/12-15.jpg)

![](files/12-16.jpg)

Think about answer space, then think about label space. Có thể dùng many-to-many mapping và gán cho trọng số (giống softmax).

## Why do we need answer engineering?
- We have reformulate the task! => We also should re-define the “ground truth labels”
- Definition:
	- aims to search for an answer space and a map to the original output Y that results in an effective predictive model

![](files/12-17.jpg)

## Answer Shape
- Token: Answers can be one or more __tokens in the pre-trained language model vocabulary__

- Chunk: Answers can be chunks of words __made up of more than one tokens__
  - Usually used with cloze prompt

- Sentence: Answers can be a sentence of arbitrary length
  - Usually used with prefix prompt

![](files/12-18.jpg)

![](files/12-19.jpg)

# Expanding the Paradigm

![](files/12-20.jpg)

![](files/12-21.jpg)

![](files/12-22.jpg)
Bạn không cần tìm 1 prompts tốt mà bạn có thể sinh ra nhiều prompts và hy vọng chúng giúp tìm ra một kết quả tốt.

![](files/12-23.jpg)

![](files/12-24.jpg)
Lý giải tại sao works? Có nhiều patterns như thế trên Internet!

![](files/12-25.jpg)
Why it works? Có sẵn patterns đó trên Internet! :D
Có sẵn đâu đó trong dataset, câu "let's think step by step" và sau đó là câu trả lời đúng.

# Prompt-based Training Strategies
- Data Perspective: _How many training samples are used?_
  - Zero-shot: without any explicit training of the LM for the downstream task
  - Few-shot: few training samples (e.g., 1-100) of downstream tasks
  - Full-data: lots of training samples (e.g., 10K) of downstream tasks

- Parameter Perspective: _Whether/How are parameters updated?_

![](files/12-26.jpg)

![](files/12-27.jpg)
https://youtu.be/5ef83Wljm-M?t=4206