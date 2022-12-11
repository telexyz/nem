# http://ruiblog.xyz/2022/06/25/DB_Notes/project2-Hash_Index

# https://www.youtube.com/watch?v=Y9H2HaRKOIw

# https://www.geeksforgeeks.org/extendible-hashing-dynamic-approach-to-dbms/

- - -

# https://discord.com/channels/724929902075445281/1014055928619872276

I wrote a short post on extendible hashing at some point, you may find it helpful

![](files/p1-00.png)
```c
int mask = (1 << global_depth_) - 1; // righ shift global_depth_
return std::hash<K>()(key) & mask; // IndexOf(key)
```
=> băm key vào index < 2^global_depth_
`global_depth_ = 0` => `mask =  0b` => `IndexOf(key) = 0`
`global_depth_ = 1` => `mask =  1b` => `IndexOf(key) = 0, 1`
`global_depth_ = 2` => `mask = 11b` => `IndexOf(key) = 0, 1, 2, 3`

![](files/p1-01.png)
![](files/p1-02.png)
![](files/p1-03.png)
