---
title: RNN和LSTM
date: 2024-09-22 16:36:24
categories:
    - NLP
tags:
---

## 什么是RNN和LSTM

- RNN（Recurrent Neural Network），循环神经网络，是一种用来处理序列数据的深度学习模型。这里的序列问题大致可以分为两类：时间序列和文本。需要指出的是，当前RNN 在很大程度上正在被基于transform的大型语言模型（LLM）所取代，后者在顺序数据处理中的效率要高得多。

- LSTM （Long short-term memory）长短期记忆神经网络，是一种特殊的RNN，常用于处理序列问题。

- 当然，RNN也能处理视频和图像，视频本质上是图像序列。同理，CNN也能处理序列数据。因为不论是CNN或者RNN，其输入都是矩阵。

## 理解时间序列问题

![](/img/note/202409221645.png)

## RNN和LSTM的结构

### RNN结构

![](/img/note/202409221646.png)

### LSTM结构

- RNN的长期依赖会导致梯度消失和梯度爆炸的问题，影响了RNN模型的训练。为了解决该问题，LSTM应用而生。

![](/img/note/202409221647.png)

![](/img/note/202409221648.png)

![](/img/note/202409221649.png)
