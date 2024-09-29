---
title: DINO
date: 2024-09-29 16:39:17
categories:
    - CV
tags:
---

## 方法概述

- 解决DETR训练收敛慢和小目标检测性能差等问题

![](/img/note/202409292110.png)

## 改进点

### CDN对比去躁训练

![](/img/note/202409292111.png)

- 避免模型重复预测
- 避免远距离的query被选中来预测结果

### Mixed Query Selection

![](/img/note/202409292112.png)

- Static Queries：如DAB-DETR中的Anchor和DETR中的Position Queries都只是学习数据集的分布，与输入是无关的。另外Content Queries更是直接初始化为0。

- Pure Query Selection：如Deformable-DETR和Efficient DETR中从Encoder的输出中筛选了一部分特征经过全连接层后直接作为decoder的Position和Content Queries。此时的Queries的位置和内容都是与输入有关的。

- Mixed Query Selection（proposed）：Position Queries来源于Query Selection后的特征，Content Queries仍然采用静态的。作者认为没有没有经过精细化的特征如果作为Content Queries可能会误导decoder中的训练。

### Look Forward Twice

![](/img/note/202409292113.png)
