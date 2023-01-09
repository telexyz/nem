# pip3 install --upgrade --no-deps git+https://github.com/dlsyscourse/mugrade.git
# pip3 install pytest numpy numdifftools pybind11 requests

# cp -rf apps python /Volumes/GoogleDrive/My\ Drive/10714/hw1
# cp /Volumes/GoogleDrive/My\ Drive/10714/hw1/hw1.ipynb .

# python3 -m pytest
# python3 -m pytest -l -v -k "backward"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "backward"

python3 -m pytest -l -k "nn_epoch_ndl"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_epoch_ndl"