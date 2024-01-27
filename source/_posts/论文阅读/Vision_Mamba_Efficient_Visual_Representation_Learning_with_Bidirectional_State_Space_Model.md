---
title: Vision Mamba Efficient Visual Representation Learning with Bidirectional State Space Model
date: 2024-01-20 10:12:54
categories:
    - 计算机视觉
tags:
---

## 贡献
- Vim实现了与成熟ViT(如DeiT)相比更高的性能，同时**显著提高了计算和内存效率**。Vim具有克服ViT处理高分辨率图像时的计算和内存限制的潜力，并有可能成为下一代视觉基础模型
![](/img/paper/202401231429.png)

## 方法
- 我们提出了一个新的基于双向曼巴块（Vim）的通用视觉骨干，该模型通过位置嵌入标记图像序列并通过双向状态空间模型压缩视觉表示
![](/img/paper/202401231438.png)

## 代码实现
![](/img/paper/202401231441.png)

