https://arxiv.org/pdf/2005.14165.pdf

# Language Models are Few-Shot Learners

![](files/lec02-r00.jpg)
Figure 1.1: Language model meta-learning. During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. It then uses these abilities at inference time to rapidly adapt to or recognize the desired task. We use the term “in-context learning” to describe the inner loop of this process, which occurs within the forward-pass upon each sequence. The sequences in this diagram are not intended to be representative of the data a model would see during pre-training, but are intended to show that there are sometimes repeated sub-tasks embedded within a single sequence.

![](files/lec02-r01.jpg)
Figure 1.2: Larger models make increasingly efficient use of in-context information. We show in-context learning performance on a simple task requiring the model to remove random symbols from a word, both with and without a natural language task description (see Sec. 3.9.2). The steeper “in-context learning curves” for large models demonstrate improved ability to learn a task from contextual information. We see qualitatively similar behavior across a wide range of tasks.

![](files/lec02-r02.jpg)
Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning. The panels above show
four methods for performing a task with a language model – fine-tuning is the traditional method, whereas zero-, one-,
and few-shot, which we study in this work, require the model to __perform the task with only forward passes at test
time__. We typically present the model with a few dozen examples in the few shot setting. Exact phrasings for all task
descriptions, examples and prompts can be found in Appendix G.

![](files/lec02-r03.jpg)


GPT-3 architect is same as GPT-2 [RWC+19], including the modified initialization, pre-normalization,
and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse
attention patterns in the layers of the transformer, similar to the Sparse Transformer [CGRS19].

Datasets for language models have rapidly expanded, culminating in the Common Crawl dataset2
[RSR+19] constituting nearly a trillion words.

![](files/lec02-r05.jpg)
Figure 2.2: Total compute used during training. Based on the analysis in Scaling Laws For Neural Language Models
[KMH+20] we train much larger models on many fewer tokens than is typical. As a consequence, although GPT-3 3B
is almost 10x larger than RoBERTa-Large (355M params), both models took roughly 50 petaflop/s-days of compute
during pre-training. Methodology for these calculations can be found in Appendix D.

![](files/lec02-r06.jpg)
Table 3.2: Performance on cloze and completion tasks. GPT-3 significantly improves SOTA on LAMBADA while
achieving respectable performance on two difficult completion prediction datasets. a[Tur20] b[RWC+19] c[LDL19] d[LCH+20]

![](files/lec02-r07.jpg)
Figure 3.2: On LAMBADA, the few-shot capability of language models results in a strong boost to accuracy. GPT-3
2.7B outperforms the SOTA 17B parameter Turing-NLG [Tur20] in this setting, and GPT-3 175B advances the state of
the art by 18%. Note zero-shot uses a different format from one-shot and few-shot as described in the text.

The few-shot setting instead allows us to “frame” the task as a cloze-test and allows the language model to infer from examples that a completion of exactly one word is desired. We use the following fill-in-the-blank format:
- Alice was friends with Bob. Alice went to visit her friend ______. → Bob
- George bought some baseball equipment, a ball, a glove, and a ______. →

When presented with examples formatted this way, GPT-3 achieves 86.4% accuracy in the few-shot setting, an increase
of over 18% from the previous state-of-the-art. We observe that few-shot performance improves strongly with model
size. While this setting decreases the performance of the smallest model by almost 20%, for GPT-3 it improves accuracy
by 10%. Finally, the fill-in-blank method is not effective one-shot, where it always performs worse than the zero-shot
setting. Perhaps this is because all models still require several examples to recognize the pattern.

![](files/lec02-r08.jpg)
Figure 3.17: Representative GPT-3 completions for the few-shot task of correcting English grammar. Boldface
is GPT-3’s completions, plain text is human prompts. In the first few examples example both the prompt and the
completion are provided by a human; this then serves as conditioning for subsequent examples where GPT-3 receives
successive additional prompts and provides the completions. Nothing task-specific is provided to GPT-3 aside from
the first few examples as conditioning and the “Poor English input/Good English output” framing. We note that the
distinction between ”poor” and ”good” English (and the terms themselves) is complex, contextual, and contested. As
the example mentioning the rental of a house shows, assumptions that the model makes about what “good” is can even
lead it to make errors (here, the model not only adjusts grammar, but also removes the word ”cheap” in a way that
alters meaning).

