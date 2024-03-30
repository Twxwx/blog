---
title: Python基础语法
date: 2024-01-17 14:36:23
categories:
    - 基础语法
tags:
---

## python基础语法

### python数组
``` python
multilist = [[0 for _ in range(3)] for _ in range(5)] # 5*3 的二维数组初始化为0

```

### python字典
```python
# 空的字典，可以用来存储键值对。两种方式是等价的
hashtable = {} 
hashtable = dict()

hashtable['one'] = "1"

# 根据 key 查找字典中的 value，如果指定键的值不存在时，返回该默认值
hashtable.get(key, default=None) 

# 删除值为key的value
value = hashtable.pop(key)

```

### python defaultdict

- 如果 key 是 list 需要先转换成 tuple 才能进行哈希

- collections.defaultdict(list)
``` python

import collections
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)] # tuple-list
 
# defaultdict
d = collections.defaultdict(list)
for k, v in s:
    d[k].append(v)
print(d.items())
print(d.keys())
print(d.values())

# dict_items([('yellow', [1, 3]), ('blue', [2, 4]), ('red', [1])])
# dict_keys(['yellow', 'blue', 'red'])
# dict_values([[1, 3], [2, 4], [1]])

```

- collections.defaultdict(set)
``` python

import collections
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = collections.defaultdict(set)
for k, v in s:
    d[k].add(v)
print(d.items())

# dict_items([('yellow', {1, 3}), ('blue', {2, 4}), ('red', {1})])

```

- collections.defaultdict(int)
``` python
import collections
string = 'nobugshahaha'
count = defaultdict(int)
for key in string:
	count[key] += 1
 
print(count.items())

# dict_items([('n', 1), ('o', 1), ('b', 1), ('u', 1), ('g', 1), ('s', 1), ('h', 3), ('a', 3)])

```



### python字符串
```python
str = "Hello, World!" # 字符串的索引从0开始，第一个字符是"H"
print(str[0]) # H
print(str[-1]) # !  倒数第一个字符是! 

# 切片操作： str[start:end] ,包含start，不包含end
print(str[0:5]) # Hello

# 字符串长度： len()
print(len(str)) # 13

# 字符串拼接
"".join("AB") # AB 

# 字符串排序
sorted(str) # 返回的是字符的集合

# 字符串和ASCII码转换
print(ord('A')) # 65
print(chr(97)) # a

# 判断字符是否是数字
c.isdigit()

# 遍历字符串
for ch in str:
    print(ch)
```

### 遍历方法
``` python
# 用range来倒序遍历 len(nums) - 1 对应 start，-1为结束点但取不到（python遵循左闭右开），-1是步长，即每次迭代时减1。
for i in range(len(nums) - 1, -1, -1):
    print(nums[i])
``` 

