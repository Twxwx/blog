---
title: python基础语法
date: 2024-01-17 14:36:23
categories:
tags:
---

## python基础语法

### python数组
``` python
multilist = [[0 for _ in range(3)] for _ in range(5)] # 5*3 的二维数组初始化为0

```

### python字典
```python
dict = {} # 空的字典，可以用来存储键值对。
dict['one'] = "1 - 菜鸟教程"
print(dict) # {'one': '1 - 菜鸟教程'}
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

# 字符串和ASCII码转换
print(ord('A')) # 65
print(chr(97)) # a
```

### 遍历方法
``` python
# 用range来倒序遍历 len(nums) - 1 对应 start，-1为结束点但取不到（python遵循左闭右开），-1是步长，即每次迭代时减1。
for i in range(len(nums) - 1, -1, -1):
    print(nums[i])
``` 

