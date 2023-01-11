# pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
# pip3 install pytest numpy numdifftools pybind11 requests
# python3 -m pytest -k "parse_mnist"
# python3 -m pytest # -k "softmax_loss"
# python3 -m pytest -k "softmax_regression_epoch and not cpp"
# make && python3 -m pytest -k "softmax_regression_epoch_cpp"
python3 -m pytest -k "nn_epoch"
python3 -m pytest

cp src/simple_ml* /Volumes/GoogleDrive/My\ Drive/10714/hw0/src
cp /Volumes/GoogleDrive/My\ Drive/10714/hw0/hw0.ipynb .