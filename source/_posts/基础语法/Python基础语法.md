---
title: Python基础语法
date: 2024-01-17 14:36:23
categories:
    - 基础语法
tags:
---

## python数组

``` python
# 5*3 的二维数组初始化为0
nums = [[0 for _ in range(3)] for _ in range(5)] 
nums = [[0] * 3 for _ in range(5)]
```

## python字典

```python
# 空的字典，可以用来存储键值对。两种方式是等价的
hashtable = {} 
hashtable = dict()

# 给字典赋值
hashtable['one'] = "1"

# 根据 key 查找字典中的 value，如果指定键的值不存在时，返回该默认值
hashtable.get(key, default=None) 

# 删除值为key的value
value = hashtable.pop(key)
```

## python 队列

- 三种实现方式：Queue(较慢)、list(最慢)、deque(最快)

### deque

```python
import collections
# 创建双端队列
q = collections.deque()
# 入队，从队列右端(队尾)插入
q.append(1) 
# 入队，从队列左端(队头)插入
q.appendleft(1)
# 判断空
if q
# 出队，从队列左端(队头)删除一个元素，并返回该元素
a = q.popleft()
# 出队，从队列右端(队尾)删除一个元素，并返回该元素
a = q.pop()
# 队列大小
len(q)
```

### list

``` python
# 从队尾插入
q.append() 
# 删除队头
del q[0] 
# 队列大小
len(q) 
# 队列为空
if not q 
```

### Queue

``` python
import queue
q = queue.Queue()
# 从队尾插入
q.put() 
# 从队头删除，并返回
q.get() 
# 队列大小
q.qsize() 
# 队列是否为空
q.empty() 
```

## python defaultdict

### collections.defaultdict(list)

``` python
import collections
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)] # tuple-list
d = collections.defaultdict(list)
for k, v in s:
    d[k].append(v)
print(d.items()) # dict_items([('yellow', [1, 3]), ('blue', [2, 4]), ('red', [1])])
print(d.keys()) # dict_keys(['yellow', 'blue', 'red'])
print(d.values()) # dict_values([[1, 3], [2, 4], [1]])
```

### collections.defaultdict(set)

``` python
import collections
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = collections.defaultdict(set)
for k, v in s:
    d[k].add(v)
print(d.items()) # dict_items([('yellow', {1, 3}), ('blue', {2, 4}), ('red', {1})])
```

### collections.defaultdict(int)

``` python
import collections
string = 'nobugshahaha'
count = defaultdict(int)
for key in string:
	count[key] += 1
print(count.items()) # dict_items([('n', 1), ('o', 1), ('b', 1), ('u', 1), ('g', 1), ('s', 1), ('h', 3), ('a', 3)])
```

## python字符串

``` python
s = ' Hello world   '
print(s.strip()) # 去除开头和结尾的空格 Hello world
print(s.split()) # 以空格分割字符串 ['Hello', 'world']
print(s[::-1]) # 反转字符串，空格会保留

str = "Hello, World!" # 字符串的索引从0开始，第一个字符是"H"
print(str[0]) # H
print(str[-1]) # !  倒数第一个字符是! 

# 切片操作： str[start:end] ,包含start，不包含end
print(str[0:5]) # Hello

# 字符串长度： len()
print(len(str)) # 13

# 连接字符序列
words = ['Hello', 'World']
print(" ".join(words)) # Hello World 

# 字符串排序
words = ['Hello', 'World']
word = 'dhdkhkd'
print(sorted(words)) # ['Hello', 'World']
print(sorted(word)) # ['d', 'd', 'd', 'h', 'h', 'k', 'k']

# 字符串和ASCII码转换
print(ord('A')) # 65
print(chr(97)) # a

# 判断字符是否是数字
c.isdigit()

# 遍历字符串
for ch in str:
    print(ch)
```

## 遍历方法

``` python
# 用range来倒序遍历 len(nums) - 1 对应 start，-1为结束点但取不到（python遵循左闭右开），-1是步长，即每次迭代时减1。
# 等效于 for (int i = len(nums) - 1, i > -1, i--)
for i in range(len(nums) - 1, -1, -1):
    print(nums[i])

# 得到数组元素的下标和值
for i, num in enumerate(nums):
    print(i, num)
``` 

## 表示最大值

```python
maxx = float('inf')
maxx = math.inf
minn = float('-inf')
minn = -math.inf
```

## 排序函数

```python
nums = [2, 24, 8, 6, 35, 7, 22, 30]
nums.sort() # sort作用在list上，不产生新的列表 [2, 6, 7, 8, 22, 24, 30, 35]
nums.sort(reverse=True) # [35, 30, 24, 22, 8, 7, 6, 2]
new_nums = sorted(nums) # sorted可以对所有可迭代的对象进行排序，产生新的列表 [2, 6, 7, 8, 22, 24, 30, 35]
```

```python
# 如果对元组列表进行排序，首先比较第一个元组元素，如果相同则再比较第二个元组元素，以此类推
nums = [(0, 0, 1), (-1, 1, 2), (1, 1, 3), (-2, 2, 4), (0, 2, 6), (0, 2, 5), (2, 2, 7)]
nums.sort()
print(nums) # [(-2, 2, 4), (-1, 1, 2), (0, 0, 1), (0, 2, 5), (0, 2, 6), (1, 1, 3), (2, 2, 7)]
```