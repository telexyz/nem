# mv _git .git && git pull
# mv .git _git

cd build
make format
make check-lint

# Project 0
# - - - - -
# https://15445.courses.cs.cmu.edu/fall2022/project0/
# make starter_trie_test
# make check-clang-tidy-p0
# ./test/starter_trie_test

# https://juejin.cn/post/7139572163371073543

# valgrind \
#     --error-exitcode=1 \
#     --leak-check=full \
#     ./test/starter_trie_test

# zip project0-submission.zip \
#     src/include/primer/p0_trie.h 

# Project 1
# - - - - -
# https://15445.courses.cs.cmu.edu/fall2022/project1/
make extendible_hash_table_test -j8
make check-clang-tidy-p1
# ./test/extendible_hash_table_test
# make submit-p1

# https://skyfan2002.github.io