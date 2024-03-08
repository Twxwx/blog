---
title: GLM系列
date: 2024-02-28 09:50:28
categories:
    - 大模型
tags:
---

## 背景

- GLM的核心是：自回归空白填充（Autoregressive Blank Infilling）
- Prefix LM 架构

## 技术原理

- GLM 在只使用 Transformer 编码器的情况下，自定义 attention mask 来兼容三种模型结构，使得前半部分互相之间能看到，等效于编码器(BERT)的效果，侧重于信息提炼、后半部分只能看到自身之前的，等效于解码器(GPT)的效果，侧重于生成。这样综合起来实现的效果就是，将提炼信息作为条件，进行有条件地生成

![](/img/note/202403062208.png)

## GLM 的预训练

![](/img/note/202403062209.png)

## GLM 的微调

![](/img/note/202403062210.png)







