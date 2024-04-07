---
title: Side Adapter Network for Open-Vocabulary Semantic Segmentation
date: 2024-04-05 17:28:03
categories: 
    - 论文阅读
tags: 
---

[参考链接](https://zhuanlan.zhihu.com/p/662894016)

[代码链接](https://github.com/MendelXu/SAN)

-  the CLIP model is trained by image-level contrastive learning. Its learned representation lacks the pixel-level recognition capability that is required for semantic segmentation

- The side adapter network has two branches: one predicting mask proposals, and one predicting attention biases that
are applied to the self-attention blocks of CLIP for mask class recognition.

- we use low-resolution images in the CLIP model and high-resolution images in the side adapter network.

## 方法概述

![](/img/paper/202404061735.png)

![](/img/paper/202404061736.png)

![](/img/paper/202404061737.png)