# mv _git .git && git pull
# mv .git _git

# https://15445.courses.cs.cmu.edu/fall2022/project0/
cd build
make format
make starter_trie_test
make check-clang-tidy-p0
make check-lint
./test/starter_trie_test
# https://juejin.cn/post/7139572163371073543

