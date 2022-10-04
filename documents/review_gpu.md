https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning

## The Most Important GPU Specs for Deep Learning Processing Speed

### Tensor Cores

- Tensor Cores reduce the used cycles needed for calculating multiply and addition operations, `16-fold` — in my example, for a 32×32 matrix, from 128 cycles to 8 cycles.

- Tensor Cores reduce the reliance on repetitive shared memory access, thus saving additional cycles for memory access.

- Tensor Cores are so fast that computation is no longer a bottleneck. The only bottleneck is getting data to the Tensor Cores.

Here are some important cycle timings or latencies for operations:
- Global memory access (up to 48GB): `~200 cycles`
- Shared memory access (up to 164 kb per Streaming Multiprocessor): `~20 cycles`
- Fused multiplication and addition (FFMA): `4 cycles`
- Tensor Core matrix multiply: `1 cycle`

### Shared Memory / L1 Cache Size / Registers
- `Volta`:   96 kb shared memory / 32 kb L1
- `Turing`:  64 kb shared memory / 32 kb L1
- `Ampere`: 164 kb shared memory / 32 kb L1

We see that Ampere has a much larger shared memory allowing for larger tile sizes, which reduces global memory access. Thus, Ampere can make better use of the overall memory bandwidth on the GPU memory. This improves performance by roughly 2-5%. The performance boost is particularly pronounced for huge matrices.

### Estimating Ampere Deep Learning Performance

Debiased benchmark data suggests that the Tesla A100 compared to the V100 is `1.70x faster for NLP` and 1.45x faster for computer vision.


### Additional Considerations for Ampere / RTX 30 Series

The new NVIDIA Ampere RTX 30 series has additional benefits over the NVIDIA Turing RTX 20 series, such as `sparse network` training and inference. Other features, such as the new data types, should be seen more as an ease-of-use-feature as they provide the same performance boost as Turing does but without any extra programming required.

### New Fan Design / Thermal Issues

If you want to buy 1 GPU or 2 GPUs in a 4 PCIe slot setup, then there should be no issues.

The components’ maximum power is only used if the components are fully utilized, and in deep learning, the CPU is usually only under weak load. With that, a 1600W PSU might work quite well with a 4x RTX 3080 build, but for a 4x RTX 3090 build, it is better to look for high wattage PSUs (+1700W). 

### GPU Deep Learning Performance
![](https://timdettmers.com/wp-content/uploads/2020/09/Normalized-GPU-Performance-Ampere-1.svg)

### GPU Deep Learning Performance per Dollar

Here some rough guidelines for memory:
- Kaggle competitions `>= 8 GB`
- Prototyping neural networks (either transformer or convolutional nets) `>= 10 GB`
- Using pretrained transformers; training small transformer `>= 11GB`
- Training large transformer in research / production: `>= 24 GB`

![](https://timdettmers.com/wp-content/uploads/2020/09/Normalized-1-and-2-GPU-Performance-per-Dollar-Ampere-1.svg)
*Normalized deep learning 1-2 GPU performance-per-dollar relative to RTX 3080.*

### When do I need >= 11 GB of Memory?

I mentioned before that you should have at least 11 GB of memory if you work with transformers, and better yet, >= 24 GB of memory if you do research on transformers. This is so because most previous models that are pretrained have pretty steep memory requirements, and these models were trained with at least RTX 2080 Ti GPUs that have 11 GB of memory. Thus having less than 11 GB can create scenarios where it is difficult to run certain models.

### When is <11 GB of Memory Okay?

The RTX 3070 and RTX 3080 are mighty cards, but they lack a bit of memory. For many tasks, however, you do not need that amount of memory.

The RTX 3070 is perfect if you want to learn deep learning. This is so because the basic skills of training most architectures can be learned by just scaling them down a bit or using a bit smaller input images. **If I would learn deep learning again, I would probably roll with one RTX 3070, or even multiple if I have the money to spare.**

The RTX 3080 is currently by far the most cost-efficient card and thus ideal for prototyping. For prototyping, you want the largest memory, which is still cheap. With prototyping, I mean here prototyping in any area: Research, competitive Kaggle, hacking ideas/models for a startup, experimenting with research code. For all these applications, the RTX 3080 is the best GPU.

### How can I fit +24GB models into 10GB memory?

It is a bit contradictory that I just said if you want to train big models, you need lots of memory, but we have been struggling with big models a lot since the onslaught of BERT and solutions exists to train 24 GB models in 10 GB memory. If you do not have the money or what to avoid cooling/power issues of the RTX 3090, you can get RTX 3080 and just accept that you need do some extra programming by adding memory-saving techniques. There are enough techniques to make it work, and they are becoming more and more commonplace.

- FP16/BF16 training
- Gradient checkpointing (only store some of the activations and recompute them in the backward pass)
- GPU-to-CPU Memory Swapping (swap layers not needed to the CPU; swap them back in just-in-time for backprop)
- Model Parallelism (each GPU holds a part of each layer; supported by fairseq)
- Pipeline parallelism (each GPU hols a couple of layers of the network)
- ZeRO parallelism (each GPU holds partial layers)
- 3D parallelism (Model + pipeline + ZeRO)
- CPU Optimizer state (store and update Adam/Momentum on the CPU while the next GPU forward pass is happening)

If you are not afraid to tinker a bit and implement some of these techniques — which usually means integrating packages that support them with your code — you will be able to fit that 24GB large network on a smaller GPU. **With that hacking spirit, the RTX 3080, or any GPU with less than 11 GB memory, might be a great GPU for you.**

## => Kết luận

2 RTX 3080 build, 1000W PSU, Ryzen