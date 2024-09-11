---
title: NLP评价指标
date: 2024-04-04 14:58:41
categories:
    - NLP
tags:
---

## BLEU(Bilingual Evaluation Understudy)

- 定义：BLEU是一种用于评估机器翻译质量的指标，它通过比较机器翻译输出和一组参考翻译之间的n-gram重叠来评分。

- 计算方式：BLEU分数是通过计算机器翻译输出和参考翻译之间的n-gram精确匹配度，并通过短语长度惩罚因子来调整得到的。BLEU分数的范围从0到1，其中1表示完美的匹配。

- 应用场景：BLEU主要用于机器翻译任务，但也可用于其他文本生成任务，如文本摘要。

![](/img/note/202408261717.png)

![](/img/note/202408261718.png)

![](/img/note/202408261719.png)

## Rouge(Recall-Oriented Understudy for Gisting Evaluation)

- 定义：ROUGE是一组用于评估自动文摘和机器翻译质量的指标，它主要关注召回率，但也考虑精确率。

- 计算方式：ROUGE包括多种指标，如ROUGE-N（基于n-gram的重叠）、ROUGE-L（基于最长公共子序列）和ROUGE-W（加权最长公共子序列）。每种指标都有其特定的计算方式，主要关注生成的文本和参考文本之间的重叠程度。

- 应用场景：ROUGE广泛用于自动文摘和机器翻译任务，特别是在需要考虑文本的整体相似性时。

![](/img/note/202408261720.png)

![](/img/note/202408261721.png)

## perplexity（困惑度、复杂度）

![](/img/note/202408261722.png)

## 分类评价指标

### 准确率（Acc）、错误率

![](/img/note/202409051328.png)

### 精确率（Precision）、召回率（Recall）、F1值

![](/img/note/202409051329.png)