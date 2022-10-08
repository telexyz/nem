# pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
# pip3 install pytest numpy numdifftools pybind11 requests

# python3 -m pytest

# python3 -m pytest -l -k "test_op_logsumexp_backward_5"
# python3 -m pytest -v -k "test_nn_layernorm"
# python3 -m pytest -v -k "test_nn_batchnorm"
# python3 -m pytest -v -k "test_optim_adam"
# python3 -m pytest -v -k "test_optim_sgd"

# python3 -m pytest -v -k "test_dataloader"
python3 -m pytest -v -k "test_dataloader_ndarray"

# python3 -m pytest -v -k "test_mlp"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "mlp_resnet"

# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "init" -s
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_linear"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_relu"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_sequential"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "op_logsumexp"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_softmax_loss"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_flatten"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_dropout"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_residual"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "flip_horizontal"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "random_crop"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "test_mnist_dataset"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "test_dataloader"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k ""
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k ""

# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "optim_sgd"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "optim_adam"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_batchnorm"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_layernorm"
