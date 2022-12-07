mv _git .git && git pull
mv .git _git

# https://15445.courses.cs.cmu.edu/fall2022/project0/
cd build
make format
make starter_trie_test
./test/starter_trie_test
make check-clang-tidy-p0
