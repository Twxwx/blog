---
title: BERT Pre-training of Deep Bidirectional Transformers for Language Understanding
date: 2024-01-12 21:42:09
categories:
    - NLP
tags:
---

[论文链接](https://arxiv.org/pdf/1810.04805.pdf)

## Abstract
1. BERT 用无标签文本训练深度双向表示, 在所有层中综合利用文本左右边两个方向的上下文信息
2. 预训练的BERT模型可以通过仅仅增加一个额外的输出层, 然后微调, 就能在很多任务上达到最高水平, 比如问答/语言推断,这些任务, 不需要根据大量的任务进行模型结构的修改

## 先前方法的不足
- 采用 feature-based 方法的 ELMo 模型，它是通过从左向右(LTR)和从右向左(RTL)两个模型的输出拼接获得词的表示；
- 采用预训练加 fine-tune 的 OpenAI GPT，它是通过从左向右的单向模型来训练。
- 单向模型的主要缺点在于不能获得足够好的词表示，在句子级任务以及分词级任务的效果都是不够好的，同时模型训练的速度也会受到限制。

![](/img/paper/202401241821.png)

## 训练方法
Bert是一种基于Transformer Encoder的自监督语言表征模型，Bert模型包括预训练(Pre-trained)和微调(Fine-tuned)两个阶段，其中Pre-trained主要是利用海量预料库进行自监督学习从而获取到预料库词表征向量，而Fine-tuned主要是针对不同的任务在Pre-trained模型的基础上进行有监督的训练。

### Pre-training BERT
Bert在预训练阶段基于Transformer Encoder利用自监督的方式对模型参数和词向量Embedding层进行学习，Bert利用无标签数据构建了两个不同的有监督任务，分别是Masked Language Model和Next Sentence Prediction。

1. Masked Language Model(MLM)：MLM通过利用类似完形填空任务(多分类Task，损失函数使用交叉熵)来进行自监督学习，首先将Sentence中的Token被随机的Mask后的数据作为输入数据，然后通过利用Mask Token的上下文信息对其原始Token进行预测。Pre-training会对原始输入的头和尾进行标识词添加，标识词包括[CLS]和[SEP]，其中[CLS]用于输入数据的开头，[SEP]用于输入数据的中间分割前后两个不同的句子或者结尾。

Bert Mask策略：随机选取15%的Token进行Mask，在选取的15%的Token中其中有80%是直接进行Mask，10%保持不变，10%替换成其他Token。其中10%为了保持Pre-training和Fine-tuning时数据分布的一致性，而10%替换成其他Token是为了让模型避免只关注被Mask的部分。

2. Next Sentence Prediction(NSP)：NSP通过利用类似句子配对任务(二分类Task，损失函数使用交叉熵)来进行自监督学习，首先从预料库中的构建同等比例的成对句子与非成对句子，成对句子作为正样本(A and B label is 0)，非成对句子作为负样本(A and B label is 1)，然后在加入标识词之后([CLS] A [SEP] B [SEP])将数据输入到多层Transformer Encoder中进行特征提取，最后将[CLS] Token的表征向量作为整体句子的表征信息输入到Softmax层进行二分类。

![](/img/paper/202401251127.png)

### Fine-tuning BERT
在Pre-training BERT的基础上针对特定任务进行有监督训练，比如对于文本匹配任务，只需要将预训练阶段的输出层的参数重新初始化然后开始端到端的有监督训练即可。

![](/img/paper/202401261415.png)

### BERT input representation
Bert模型输入数据要经过三种不同的表征层，如下所示分别是Token 、Segment 、Position Embedding，其中Token Embedding是对Token进行向量化，Segment Embedding是对句子左右位置的向量化，Position Embedding是对位置信息进行向量化即使得序列的位置信息获取方式不再是一种硬编码方式，而是使得变为一种可学习的方式，相比硬编码方式这种向量化的方式更具有泛化性。

![](/img/paper/202401251545.png)


