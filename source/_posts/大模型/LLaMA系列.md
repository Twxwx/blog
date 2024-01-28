---
title: LLaMA系列
date: 2024-01-28 10:35:02
categories:
    - 大模型
tags:
---

## LLaMA

[论文链接](https://arxiv.org/pdf/2302.13971v1.pdf)

### 核心思想
- 大部分用户没有训练LLM的资源，更多的是拿着训好的LLM来推理。首选的模型应该不是训练最快的，而应该是推理最快的小LLM。

### 摘要
- LLaMA（Large Language Model Meta AI），共有 7B、13B、33B、65B 四种版本。

- 关于模型性能，LLaMA 的性能非常优异：具有 130 亿参数的 LLaMA 模型「在大多数基准上」可以胜过 GPT-3（ 参数量达 1750 亿），而且可以在单块 V100 GPU 上运行；而最大的 650 亿参数的 LLaMA 模型可以媲美谷歌的 Chinchilla-70B 和 PaLM-540B。

- 训练集的来源都是公开数据集。整个训练数据集在 token 化之后大约包含 1.4T 的 token。其中，LLaMA-65B 和 LLaMA-33B 是在 1.4万亿个 token 上训练的，而最小的模型 LLaMA-7B 是在 1万亿个 token 上训练的。

### 模型结构
![](/img/note/202401281517.png)


## LLaMA 2

[论文链接](https://arxiv.org/pdf/2307.09288.pdf)

### LLaMA 2

- 训练数据从 1.4T tokens 增加到 2.0 tokens
- 上下文窗口从 2k 增加到 4k
- 采用分组查询注意力（ Grouped-Query Attention）：对于更大参数量、更大的 context length、更大的 batchsize 来说，原始的MHA（multi-head attention）的内存占用会更高（因为在计算时要缓存pre token的K、V矩阵）。

### LLaMA 2-CHAT

#### 训练流程
![](/img/note/202401281548.png)




