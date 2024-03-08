---
title: DeepSpeed
date: 2024-01-20 17:47:51
categories:
    - 大模型
tags:
---

[参考链接](https://blog.csdn.net/zwqjoy/article/details/130732601)

## 概览

### 简介

- DeepSpeed是一个由微软开发的开源深度学习优化库，旨在提高大规模模型训练的效率和可扩展性。它通过多种技术手段来加速训练，包括模型并行化、梯度累积、动态精度缩放、本地模式混合精度等。

- DeepSpeed作为一个大模型训练加速库，位于模型训练框架和模型之间，用来提升训练、推理等。

### 核心思想
- **GPU显存不够，CPU内存来凑**。具体点说，DeepSpeed将当前时刻，训练模型用不到的参数，缓存到CPU中，等到要用到了，再从CPU挪到GPU。
- 越多的参数挪到CPU上，GPU的负担就越小；但随之的代价就是，更为频繁的CPU，GPU交互，极大增加了训练推理的时间开销。因此，DeepSpeed需要对时间开销和显存占用的进行权衡。

    1. Optimizer state partitioning (ZeRO stage 1)  只对optimizer进行切片后分布式保存
    2. Gradient partitioning (ZeRO stage 2)   对optimizer和grad进行切片后分布式保存
    3. Parameter partitioning (ZeRO stage 3)  对optimizer、grad和模型参数进行切片后分布式保存
    4. 混合精度训练
    5. A range of fast CUDA-extension-based optimizers
    6. ZeRO-Offload to CPU and NVMe：offload就是将forward中间结果保存到内存、硬盘（NVMe）等缓存中，然后在需要时进行加载或重计算，进一步降低显存占用

### 软件架构

- 主要包含三部分：
    1. Apis。提供易用的api接口，训练模型、推理模型只需要简单调用几个接口即可。其中最重要的是initialize接口，用来初始化引擎，参数中配置训练参数及优化技术等。配置参数一般保存在config.json文件中。
    2. runtime。运行时组件，是deepspeed管理、执行和性能优化的核心组件。如部署训练任务到分布式设备、数据分区、模型分区、系统优化、微调、故障检测、checkpoints保存和加载等。该组件使用python语言实现。
    3. ops。用c++和cuda实现底层内核，优化计算和通信，例如ultrafast transformer kernels, fuse LAN kernels, customary deals等。

![](/img/note/202402282209.png)

## 核心技术

### ZeRO

- ZeRO Stage 1: 划分optimizer states。优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数。
- ZeRO Stage 2: 划分gradient。每个memory，只保留它分配到的optimizer state所对应的梯度。这很合理，因为梯度和optimizer是紧密联系在一起的。只知道梯度，不知道optimizer state，是没有办法优化模型参数的。
- ZeRO Stage 3: 划分模型参数，或者说，不同的layer. ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory。

### 用 3D 并行化实现万亿参数模型训练

- DeepSpeed 实现了三种并行方法的灵活组合：ZeRO 支持的数据并行、流水线并行和张量切片模型并行。

### ZeRO-Offload 使 GPU 单卡能够训练 10 倍大的模型

- 为了同时利用 CPU 和 GPU 内存来训练大型模型，扩展了 ZeRO-2。用户在使用带有单张英伟达 V100 GPU 的机器时，可以在不耗尽显存的情况下运行多达 130 亿个参数的模型，模型规模扩展至现有方法的10倍，并保持有竞争力的吞吐量。

### 通过 DeepSpeed Sparse Attention 用6倍速度执行10倍长的序列

- DeepSpeed提供了稀疏 attention kernel（一种工具性技术，可支持长序列的模型输入，包括文本输入，图像输入和语音输入）。与经典的稠密 Transformer 相比，它支持的输入序列长一个数量级，并在保持相当的精度下获得最高 6 倍的执行速度提升。它还比最新的稀疏实现快 1.5–3 倍。

### 1 比特 Adam 减少 5 倍通信量
- Adam 是一个在大规模深度学习模型训练场景下有效的优化器。然而，它与通信效率优化算法往往不兼容。因此，在跨设备进行分布式扩展时，通信开销可能成为瓶颈。我们推出了一种 1 比特 Adam 新算法，以及其高效实现。该算法最多可减少 5 倍通信量，同时实现了与Adam相似的收敛率。在通信受限的场景下，我们观察到分布式训练速度提升了 3.5 倍，这使得该算法可以扩展到不同类型的 GPU 群集和网络环境。

