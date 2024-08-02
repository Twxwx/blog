---
title: CodeGeeX
date: 2024-07-25 15:07:27
categories:
    - 大模型
tags:
---

## 概述

- CodeGeeX，是一个具有130亿个参数的多语言模型，用于代码生成，在23种编程语言的8500亿个token上进行了预训练，具有8K的上下文窗口

- CodeGeeX的特点：除了代码生成和代码补全，也支持代码解释和代码翻译。

## 架构

![](/img/note/202407251536.png)

1. 使用类GELU的FastGELU

![](/img/note/202407251537.png)

2. Top Query层和解码层

- 原始的GPT使用pooler函数来获得最终的输出。我们在所有transformer层之上使用一个额外的查询层，通过attention获得最终的embedding。top query层的输入被替换为位置n+1的query embedding。最后的输出再乘以词嵌入矩阵的转置，得到输出概率。对于解码策略，贪心、温度采样、top-k采样、top-p采样和beam search。最后，去标记化将把选中的tokenID变成一个实际的单词