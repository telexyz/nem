# For window wsl, update Windows nvidia driver, Restart computer.
# Then install nvidia cuda via apt, then link libcuda.so and libnvidia-ptxjitcompiler.so.1 using below commands

# find /usr/lib/wsl/ -name "libcuda.so"
# /usr/lib/wsl/lib/libcuda.so

# find /usr/lib/wsl/ -name "*nvidia-ptxjitcompiler*"
# /usr/lib/wsl/drivers/nvam.inf_amd64_6c91eefecceede9a/libnvidia-ptxjitcompiler.so.1
# /usr/lib/wsl/drivers/nvamsig.inf_amd64_8f9166072f173168/libnvidia-ptxjitcompiler.so.1

sudo ln -s /usr/lib/wsl/lib/libcuda.so /usr/local/cuda/lib64/
sudo ln -s /usr/lib/wsl/drivers/nvam.inf_amd64_6c91eefecceede9a/libnvidia-ptxjitcompiler.so.1 /usr/local/cuda/lib64/
sudo cp /usr/local/cuda/lib64/libnvidia-ptxjitcompiler.so.1 /usr/local/cuda/lib64/libnvidia-ptxjitcompiler.so

git submodule init; git submodule update