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
# https://15445.courses.cs.cmu.edu/fall2022/project1

rm ./test/extendible_hash_table_test
rm ./test/lru_k_replacer_test
rm ./test/buffer_pool_manager_instance_test
make buffer_pool_manager_instance_test -j8
# make extendible_hash_table_test lru_k_replacer_test buffer_pool_manager_instance_test -j8
# ./test/extendible_hash_table_test
# ./test/lru_k_replacer_test
./test/buffer_pool_manager_instance_test

# make format && make check-lint && make check-clang-tidy-p1

# rm ../project1-submission.zip; make submit-p1
