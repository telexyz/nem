http://karpathy.github.io/2022/03/14/lecun1989

Reflections. Let’s summarize what we’ve learned as a 2022 time traveler examining state of the art 1989 deep learning tech:

- First of all, not much has changed in 33 years on the macro level. We’re still setting up differentiable neural net architectures made of layers of neurons and optimizing them end-to-end with backpropagation and stochastic gradient descent. Everything reads remarkably familiar, except it is smaller.


- The dataset is a baby by today’s standards: The training set is just 7291 16x16 greyscale images. Today’s vision datasets typically contain a few hundred million high-resolution color images from the web (e.g. Google has JFT-300M, OpenAI CLIP was trained on a 400M), but grow to as large as a small few billion. This is approx. ~1000X pixel information per image `(384*384*3/(16*16))` times 100,000X the number of images (1e9/1e4), for a rough 100,000,000X more pixel data at the input.


- The neural net is also a baby: This 1989 net has approx. 9760 params, 64K MACs, and 1K activations. Modern (vision) neural nets are on the scale of small few billion parameters (1,000,000X) and O(~1e12) MACs (~10,000,000X). Natural language models can reach into trillions of parameters.


- A state of the art classifier that took 3 days to train on a workstation now trains in 90 seconds on my fanless laptop (3,000X naive speedup), and further ~100X gains are very likely possible by switching to full-batch optimization and utilizing a GPU.


- I was, in fact, able to tune the model, augmentation, loss function, and the optimization based on modern R&D innovations to cut down the error rate by 60%, while keeping the dataset and the test-time latency of the model unchanged.


- Modest gains were attainable just by scaling up the dataset alone.


- Further significant gains would likely have to come from a larger model, which would require more compute, and additional R&D to help stabilize the training at increasing scales. In particular, if I was transported to 1989, I would have ultimately become upper-bounded in my ability to further improve the system without a bigger computer.

- - -

Suppose that the lessons of this exercise remain invariant in time. What does that imply about deep learning of 2022? What would a time traveler from 2055 think about the performance of current networks?

- 2055 neural nets are basically the same as 2022 neural nets on the macro level, except bigger.
- Our datasets and models today look like a joke. Both are somewhere around 10,000,000X larger.
- One can train 2022 state of the art models in ~1 minute by training naively on their personal computing device as a weekend fun project.
- Today’s models are not optimally formulated, and just changing some of the details of the model, loss function, augmentation or the optimizer we can about halve the error.
- Our datasets are too small, and modest gains would come from scaling up the dataset alone.
- Further gains are actually not possible without expanding the computing infrastructure and investing into some R&D on effectively training models on that scale.

- - -

But the most important trend I want to comment on is that __the whole setting of training a neural network from scratch on some target task (like digit recognition) is quickly becoming outdated due to finetuning, especially with the emergence of foundation models like GPT.__ These foundation models are trained by only a few institutions with substantial computing resources, and most applications are achieved via lightweight finetuning of part of the network, prompt engineering, or an optional step of data or model distillation into smaller, special-purpose inference networks. I think we should expect this trend to be very much alive, and indeed, intensify. In its most extreme extrapolation, you will not want to train any neural networks at all. In 2055, you will ask a 10,000,000X-sized neural net megabrain to perform some task by speaking (or thinking) to it in English. And if you ask nicely enough, it will oblige. Yes you could train a neural net too… but why would you?

