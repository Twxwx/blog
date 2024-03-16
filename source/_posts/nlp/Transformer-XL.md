---
title: Transformer-XL
date: 2024-03-11 16:51:57
categories:
    - NLP
tags:
---

[论文链接](https://arxiv.org/abs/1901.02860)

## 前言

- Transformer-XL（extra long）是为了进一步提升 Transformer 建模长期依赖的能力。它的核心算法包含两部分：片段递归机制（segment-level recurrence）和相对位置编码机制 (relative positional encoding)。

- Transformer-XL带来的提升包括：1. 捕获长期依赖的能力；2. 解决了上下文碎片问题（context segmentation problem）；3. 提升模型的预测速度和准确率。

## 片段递归

- Transformer-XL 在训练的时候是以固定长度的片段的形式进行输入的，Transformer-XL 的上一个片段的状态会被缓存下来，然后在计算当前段的时候再重复使用上个时间片的隐层状态。因为上个片段的特征在当前片段进行了重复使用，这也就赋予了 Transformer-XL 建模更长期的依赖的能力。

![](/img/note/202403161514.png)

- 另一个好处是带来的推理速度的提升，对比Transformer的自回归架构每次只能前进一个时间片，Transfomer-XL的推理过程通过直接复用上一个片段的表示而不是从头计算，将推理过程提升到以片段为单位进行推理，这种简化带来的速度提升是成百上千倍的。

## 相对位置编码

- 上面所说的循环机制还有个问题待解决，就是位置编码，我们知道，原生的Transformer使用的是绝对位置编码，但绝对位置编码跟循环机制结合会导致问题

![](/img/note/202403161515.png)


