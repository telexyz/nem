https://analyticsindiamag.com/can-mxnet-stand-up-to-tensorflow-pytorch


> “MXNet, born and bred here at CMU, is the most scalable framework for deep learning I have seen and is a great example of what makes this area of computer science so beautiful – that you have different disciplines which all work so well together: imaginative linear algebra working in a novel way with massive distributed computation leading to a whole new ball game for deep learning,” - said Andrew Moore, former dean of Computer Science at the Carnegie Mellon University.

`Amazon chose MXNet` for three reasons:
- Development speed and programmability
- Portable enough to run on a broad range of devices and platforms and locations with different network facilities
- Scalable to multiple GPUs to train larger and more sophisticated models with bigger datasets.

### MXNet vs TensorFlow & PyTorch

MXNet scores big on two fronts – `ease of learning` and `speed`. Easy to learn as pytorch and as fast as tensorflow (even better gpu utilization).


- - -


https://mxnet.apache.org/get_started/build_from_source

```sh
git clone --recursive https://github.com/apache/incubator-mxnet mxnet

brew install cmake ninja ccache opencv llvm openblas gfortran

mkdir -p build && cd build

cmake .. -DUSE_CUDA=0 -DUSE_CPP_PACKAGE=1
# https://mxnet.apache.org/versions/master/api/cpp.html

python3 -m pip install --user -e ./python

```

```sh
git clone --recursive https://github.com/apache/tvm.git
```

MXNet relies on the BLAS and LAPACK. MXNet is tested with:
- Apple Accelerate
- ATLAS
- Intel MKL
- OpenBLAS

MXNet recommends OpenBLAS as it typically outperforms ATLAS, is portable across many platforms, provides a LAPACK implementation and has a permissive license.

- - -

The Gluon Python API lets you use Apache MXNet in a fully imperative manner. It also allows you to simply switch to symbolic mode by calling the hybridize functionality. The symbolic execution provides faster and more optimized execution as well as the ability to export the network for inference in different language bindings like java or C++.


MXNet allows you to make the most out of your hardware, whether it is multi-gpu or multi-host training with near-linear scaling efficiency. Apache MXNet recently introduced support for Horovod, the distributed learning framework developed by Uber.


## [mxnet vs pytorch](https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/getting-started/to-mxnet/pytorch.html)

PyTorch is a popular deep learning framework due to its easy-to-understand API and its completely imperative approach. MXNet includes the Gluon API which gives you the simplicity and flexibility of PyTorch and allows you to hybridize your network to leverage performance optimizations of the symbolic graph. As of April 2019, NVidia performance benchmarks show that Apache MXNet outperforms PyTorch by `~77%` on training ResNet-50: `10,925` images per second vs. `6,175`.

Both PyTorch and Apache MXNet relies on multidimensional matrices as a data sources. While PyTorch follows Torch’s naming convention and refers to multidimensional matrices as `tensors`, Apache MXNet follows NumPy’s conventions and refers to them as `NDArrays`.

## [Performance](https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/performance/index.html)

### Mixed precision training using float16
Need Volta range of Nvidia GPUs (e.g. AWS P3 instance)

### Gradient Compression

*(( speedup training by about 2x, accuracy loss was as low as 1% ))*

When training models whose architectures include large fully connected components, it can be helpful to use gradient compression. For larger models, as well as recurrent neural networks, the communication cost becomes a major factor. Such models stand to benefit greatly with gradient compression.

When the training is configured to use device to device communication on a single node with multiple GPUs, gradient compression can be used to reduce the cost of communication. This can provide about 20% speedup for large models using older generation architectures. However, speed benefits may be negligible on a machine with a newer generation architecture where GPUs can communicate at low latency.

### Optimizing Deep Learning Computation Graphs with TensorRT¶

NVIDIA’s TensorRT is a deep learning library that has been shown to provide large speedups when used for network inference.