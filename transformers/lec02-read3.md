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

