# sudo apt-get -y install \
#       build-essential \
#       clang-12 \
#       clang-format-12 \
#       clang-tidy-12 \
#       cmake \
#       doxygen \
#       git \
#       g++-12 \
#       pkg-config \
#       zlib1g-dev

export CC=/usr/bin/clang-12
export CXX=/usr/bin/clang++-12
# export CC="/usr/local/bin/zig cc"
# export CXX="/usr/local/bin/zig c++"
mkdir -p build && rm -rf build && mkdir -p build && cd build
cmake ..
# cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
