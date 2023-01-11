https://www.youtube.com/watch?v=FThvfkXWqtE&t=858s

![](files/fa-00.jpg)

![](files/fa-01.jpg)

Dealing with n x n matrix, so can write to the same location. In practice people write to new mem since they need the old one for gradient. But you can write to same (input) mem, it will save mem and increase speed(?) nope: since you still need the same number of read / write.

Why dropout and softmax take so much time?
