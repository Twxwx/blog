---
title: PyTorch
date: 2024-03-27 12:01:28
categories:
    - 分布式训练框架
tags:
---

## 简介

- PyTorch 分为 torch.nn.parallel.DataParallel (DP) 和 torch.nn.parallel.DistributedDataParallel (DDP)。

## DDP与DP的区别

- DP是单进程多线程的，只能在单机上工作；DDP是多进程的，可以在多级多卡上工作。DP通常比DDP慢，主要原因有：
    1. DP是单进程的，受到GIL的限制；
    2. DP每个step都需要拷贝模型，以及划分数据和收集输出；
- DDP可以与模型并行相结合；
- DP的通信成本随着卡数线性增长，DDP支持Ring-AllReduce，通信成本是固定的。

## DP模式

- DP是较简单的一种数据并行方式，直接将模型复制到多个GPU上并行计算，每个GPU计算batch中的一部分数据，各自完成前向和反向后，将梯度汇总到主GPU上。其基本流程：
    1. 加载模型、数据至内存；
    2. 创建DP模型；
    3. DP模型的forward过程：
        1. 一个batch的数据均分到不同device上；
        2. 为每个device复制一份模型；
        3. 至此，每个device上有模型和一份数据，并行进行前向传播；
        4. 收集各个device上的输出；
    4. 每个device上的模型反向传播后，收集梯度到主device上，更新主device上的模型，将模型广播到其他device上；
    5. 3-4循环。
- 在DP中，只有一个主进程，主进程下有多个线程，每个线程管理一个device的训练。因此，DP中内存中只存在一份数据，各个线程间是共享这份数据的。DP和Parameter Server的方式很像。

## DDP模式

- DDP，顾名思义，即分布式的数据并行，每个进程独立进行训练，每个进程会加载完整的数据，但是读取不重叠的数据。
- DDP执行流程为：
    - 准备阶段
        - 环境初始化
            - 在各张卡上初始化进程并建立进程间通信，对应代码：init_process_group。
        - 模型广播  
            - 将模型parameter、buffer广播到各节点，对应代码：model = DDP(model).to(local_rank)。
        - 创建管理器reducer，给每个参数注册梯度平均hook。

    - 准备数据
        - 加载数据集，创建适用于分布式场景的数据采样器，以防不同节点使用的数据不重叠。

    - 训练阶段
        - 前向传播
            - 同步各进程状态（parameter和buffer）；
            - 当DDP参数find_unused_parameter为true时，其会在forward结束时，启动一个回溯，标记未用到的参数，提前将这些设置为ready。
        - 计算梯度
            - reducer外面：各进程各自开始反向计算梯度；
            - reducer外面：当某个参数的梯度计算好了，其之前注册的grad hook就会触发，在reducer里把这个参数的状态标记为ready；
            - reducer里面：当某个bucket的所有参数都是ready时，reducer开始对这个bucket的所有参数开始一个异步的all-reduce梯度平均操作；
            - reducer里面：当所有bucket的梯度平均都结束后，reducer把得到的平均梯度正式写入到parameter.grad里。
        - 优化器应用梯度更新参数。

