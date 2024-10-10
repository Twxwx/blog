---
title: Prompt Engineering
date: 2024-02-25 23:09:40
categories:
    - 大模型
tags:
---

## 概述

- 提示工程（Prompt Engineering），是指如何针对当前任务生成prompt模板，在不更新模型权重的情况下与 LLM 交互引导其行为以获得所需结果。在提示工程中，任务的描述会被嵌入到输入中，不是隐含地给予模型一定的参数，而是以问题的形式直接输入。

- 提示工程不仅仅是关于设计和研发提示词。它包含了与大语言模型交互和研发的各种技能和技术。提示工程在实现和大语言模型交互、对接，以及理解大语言模型能力方面都起着重要作用。用户可以通过提示工程来提高大语言模型的安全性，也可以赋能大语言模型，比如借助专业领域知识和外部工具来增强大语言模型能力。

## 人工构造prompt

- 基于人工知识来定义prompt模板

### Zero-shot

![](img/note/202403062043.png)

### Few-shot

![](img/note/202403062044.png)

### 思维链（Chain-of-Thought, CoT）

![](img/note/202403062045.png)

### 指令提示（Instruction Prompting）

![](img/note/202403062046.png)

### 自我提示工程师（APE）

![](img/note/202403062047.png)

## 自动生成prompt

![](img/note/202410092150.png)

![](img/note/202410092151.png)

## 隐空间中的prompt

- 上面介绍prompt模板都是具体文本的prompt，另一种类型的prompt是在隐空间的prompt。相比于文本prompt，隐空间的prompt不需要强制让prompt模板必须是真实的文本表示，而是在隐空间学习一个文本向量，它可能无法映射到具体的单词，但是和各个词的embedding在同一个向量空间下。这种自动生成的prompt也可以不用保证必须是真实的文本，给prompt的生成带来了更大的灵活空间。例如：prefix-tuning、prompt-tuning、p-tuning
