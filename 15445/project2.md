Skip https://github.com/cmu-db/bustub/pull/452

In general, as a note to everyone, the p2 project is hard. Most people struggle to complete it perfectly. 
If you want to skip ahead to p3 and p4, Chi wrote something that you can use instead.

.. which I think might be worth pinning; p3 and p4 are pretty cool IMO and you may want to tackle those first if you're frustrated with p2. IMO, if you can pass the single threaded versions of p2 / can pass p2 using a global latch, you've understood "enough" about how a B-Tree works -- the rest is careful engineering, you may want to reimplement p2 in the future, for now it would also be OK to just skip ahead 

- - -

https://15445.courses.cs.cmu.edu/fall2022/project2/


## [Task 1](https://15445.courses.cs.cmu.edu/fall2022/project2/#b+tree-pages)

### B+Tree Parent Page

`b_plus_tree_page`

`b_plus_tree_internal_page`

`b_plus_tree_leaf_page`

`b_plus_tree_concurrent_test`

- - -

https://15445.courses.cs.cmu.edu/fall2022/slides/08-trees.pdf

https://dichchankinh.com/~galles/visualization/BPlusTree.html

A B+Tree is a self-balancing tree data structure that keeps data sorted and allows searches, sequential access, insertions, and deletions always in O(log n).

- - -

https://discord.com/channels/724929902075445281/1014055948098220062/1030890179893936128

When the b-tree is empty, concurrent Insert may change the root_page_id many times without a mutex. In this situation, are we still not allowed to add a mutex? 

You should add a mutex for root_page_id, but you'll need to unlatch it as soon as possible.

- - -

https://github.com/cmu-db/bustub/pull/389/commits/e69779152a129bb413db337cc3120a5b4d54b83b

It's a bit strange to me because the project page still references the FindLeafPage method in the BPlusTree class, even though it had been removed in the PR. 

Back when we provided all the functions, students complained that it felt like we were asking them to read our minds (of the exact semantics of MoveHalfTo etc). Ultimately, we just want students to write a working B-Tree, implementation details can vary a little. The project page does sometimes get out of sync with the code, unfortunately.

- - -

https://www.skyzh.dev/posts/articles/2022-10-05-bustub-query-processing

![](files/overview.png)