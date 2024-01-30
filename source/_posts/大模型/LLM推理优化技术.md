---
title: LLM推理优化技术
date: 2024-01-28 15:36:23
categories:
    - 大模型
tags:
---

## Transformer结构优化

![](/img/note/202401301711.png)

![](/img/note/202401301808.png)

- 减少头的数量，减少 kv cache的 size，达到减小带宽的压力的目的，那么推理速度势必更快。

### Multi Head Attention

- 标准的多头注意力机制，h 个 Query、Key 和 Value 矩阵。

### Multi Query Attention

- MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。

### Group Query Attention

- GQA将查询头分成 G 组，每个组共享一个 Key 和 Value 矩阵。GQA-G 是指具有 G 组的 grouped-query attention。GQA-1 具有单个组，因此具有单个 Key 和 Value，等效于 MQA。而 GQA-H 具有与头数相等的组，等效于MHA。