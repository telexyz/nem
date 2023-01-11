# pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
# pip3 install pytest numpy numdifftools pybind11 requests

# python3 -m pytest -k "linear"

# python3 -m pytest -l -k "test_op_logsumexp_backward_5"
# python3 -m pytest -v -k "test_nn_layernorm"
# python3 -m pytest -v -k "test_nn_batchnorm"
# python3 -m pytest -v -k "test_optim_adam"
# python3 -m pytest -v -k "test_optim_sgd"

# python3 -m pytest -v -k "test_dataloader"
# python3 -m pytest -v -k "test_dataloader_ndarray"

# python3 -m pytest -v -k "test_mlp"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "mlp_resnet"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_dropout"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "mnist_dataset"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "dataloader"

# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "optim_sgd"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "optim_adam"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_batchnorm"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_layernorm"
