---
title: 智谱系列
date: 2024-02-28 09:50:28
categories:
    - 大模型
tags:
---

## GLM

### 背景

- GLM的核心是：自回归空白填充（Autoregressive Blank Infilling）
- Prefix LM 架构

### 技术原理

- GLM 在只使用 Transformer 编码器的情况下，自定义 attention mask 来兼容三种模型结构，使得前半部分互相之间能看到，等效于编码器(BERT)的效果，侧重于信息提炼、后半部分只能看到自身之前的，等效于解码器(GPT)的效果，侧重于生成。这样综合起来实现的效果就是，将提炼信息作为条件，进行有条件地生成

![](/img/note/202403062208.png)

### GLM 的预训练

![](/img/note/202403062209.png)

### GLM 的微调

![](/img/note/202403062210.png)

## CodeGeeX

[论文链接](https://arxiv.org/pdf/2303.17568.pdf)

- 一个 13B 参数的 23 语言代码生成模型，其在代码生成和翻译上超过了同等规模的多语言基线。

### 训练数据

- 训练语料库，包含了 158B（1580亿）个标记，覆盖了 23 种编程语言。在训练过程中，为了帮助模型区分多种语言，每个语言段之前都添加了一个特定于语言的标签，例如 # language: Python

![](/img/note/202403161724.png)

### 模型架构

- 顶层查询层（Top Query Layer）：在所有其他 Transformer 层之上，CodeGeeX 使用了一个额外的查询层来获取最终的嵌入。这个查询层通过注意力机制将输入与模型的参数进行关联。
- 解码策略：CodeGeeX 支持多种解码策略，包括贪婪采样（Greedy Sampling）、温度采样（Temperature Sampling）、Top-k 采样、Top-p 采样和束搜索（Beam Search）。
- 词汇表和位置嵌入：模型使用可学习的词嵌入和位置嵌入来处理输入的标记。词嵌入用于将输入标记转换为向量表示，而位置嵌入则用于捕捉序列中标记的位置信息。

![](/img/note/202403161725.png)







