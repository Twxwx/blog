---
title: Bert及其变种
date: 2024-03-02 23:01:08
categories:
    - NLP
tags:
---

## Bert

[论文链接](https://arxiv.org/pdf/1810.04805.pdf)

- BERT的基础结构仍然是Transformer，并且仅有Encoder部分，因为它并不是生成式模型；
- BERT是一种双向的Transformer，这其实是由它的语言模型性质决定，它提出了一种掩码语言模型 MLM（masked language model）
- BERT 用无标签文本训练深度双向表示, 在所有层中综合利用文本左右边两个方向的上下文信息
- 预训练的BERT模型可以通过仅仅增加一个额外的输出层, 然后微调, 就能在很多任务上达到最高水平, 比如问答/语言推断,这些任务, 不需要根据大量的任务进行模型结构的修改

### 先前方法的不足
- 采用 feature-based 方法的 ELMo 模型，它是通过从左向右(LTR)和从右向左(RTL)两个模型的输出拼接获得词的表示；
- 采用预训练加 fine-tune 的 OpenAI GPT，它是通过从左向右的单向模型来训练。
- 单向模型的主要缺点在于不能获得足够好的词表示，在句子级任务以及分词级任务的效果都是不够好的，同时模型训练的速度也会受到限制。

![](/img/note/202401241821.png)

### BERT input representation

- Bert模型输入数据要经过三种不同的表征层，如下所示分别是Token 、Segment 、Position Embedding，其中Token Embedding是对Token进行向量化，Segment Embedding是对句子左右位置的向量化，Position Embedding是对位置信息进行向量化即使得序列的位置信息获取方式不再是一种硬编码方式，而是使得变为一种可学习的方式，相比硬编码方式这种向量化的方式更具有泛化性。将三种embedding相加得到最终输入。

![](/img/note/202401251545.png)

- 其中[CLS]代表初始token，该 token 可以用于表征整个句子，用于下游的分类任务
- 模型为了能够处理句子间的关系，两个句子的结尾都插入特殊的token [SEP] 进行分隔，并且两个句子拥有自己的 Segment Embedding
- 使用 WordPiece 来对句子进行token切分，对于英文模型，存在于词表中的则会直接映射，否则会拆分为Subword

### 训练方法

- Bert是一种基于Transformer Encoder的自监督语言表征模型，Bert模型包括预训练(Pre-trained)和微调(Fine-tuned)两个阶段，其中Pre-trained主要是利用海量预料库进行自监督学习从而获取到预料库词表征向量，而Fine-tuned主要是针对不同的任务在Pre-trained模型的基础上进行有监督的训练。

### Pre-training BERT

![](/img/note/202401251127.png)

- MLM（masked language model）：训练词的语意理解能力
- NSP（Next Sentence Prediction）：预测两个句子是否为来自连续的段落，使模型学会捕捉句子间的语义联系

#### Masked LM（masked language model）

![](/img/note/202403042144.png)

#### Next Sentence Prediction（NSP）

![](/img/note/202403042145.png)

### Fine-tuning BERT

- 模型微调则是非常直接，与预训练模型使用完全相同的网络结构，并且预训练模型的参数作为微调模型的初始化。

- 因为Transformer的self-attention机制，使得BERT能够对许多下流任务进行建模，只需要适当的变换输入和输出，不管是包含单个句子或者是句子对。

- 一般来说，对于分类任务，与预训练的 NSP 任务一样，使用初设token [CLS]的表征，来喂入到一个分类输出网络中，如 K 分类任务，则是先将[CLS]的表征向量映射到维度为K的向量，再使用softmax；

- 对于序列标注（sequence tagging），其实也是类似，使用序列中token的表征来，喂入到一个token级别的分类输出网络，如命名实体识别，将每个token的表征也是先映射到K维的向量，K为实体的数量。

![](/img/note/202401261415.png)

### 参数量计算

- bert的参数主要可以分为四部分：embedding层的权重矩阵、multi-head attention、layer normalization、feed forward
- Base：110M、Large：340M

#### embedding 层的权重矩阵

- 词向量包括三个部分的编码：词向量参数，位置向量参数，句子类型参数（bert用了2个句子，为0和1），Bert 采用的 vocab_size = 30522，hidden_size = 768，max_position_embeddings = 512，token_type_embeddings = 2。这就很显然了，embedding参数 = （30522 + 512 + 2）* 768

#### multi-head attention

- 单 head 的参数：768 * 768 / 12 * 3，multi-heads 的参数：768 * 768 / 12 * 3 * 12，之后将12个头concat后又进行了线性变换，用到了参数Wo，大小为 768 * 768，那么最后multi-heads的参数：768 * 768 / 12 * 3 * 12 + 768 * 768

#### layer normalization

- 但是参数都很少，gamma 和 beta 的维度均为 768。因此总参数为 768 * 2 + (768 * 2 + 768 * 2) * 12（层数）

#### feed forward

- 由两个全连接层组成，用到了两个参数 W1 和 W2，Bert 沿用了惯用的全连接层大小设置，即 4 * d_model = 3072，因此 W1，W2 大小为 768 * 3072，2 个为 2 * 768 * 3072

## RoBERTa

[论文链接](https://arxiv.org/abs/1907.11692)

- RoBERTa 其实就是选取更大的数据，对 BERT 的预训练模型的各种超参数进行调优，对其训练过程稍作修改所得到的。认为 BERT 模型缺乏充分的训练，在对其超参数进行调优，且经过充分训练后，模型的效果得到明显提升。该论文本身没有太大的创新性，它主要就是重新研究了BERT模型，提出一些超参数、训练过程上的改进，具体修改主要有：
    1. 将原 BERT 中的静态 mask 改为动态 mask
    2. 改变数据的输入格式，移除任务 Next Sentence Prediction
    3. 用更大的batch size进行训练
    4. 采用 Byte-Pair Encoding 进行文本编码

### 动态掩码

- 原 BERT 的 MLM 使用的是静态掩码，即数据首先会进行掩码处理，然后再喂入模型中，这样在每个 epoch 我们喂入的都是同样的数据，这意味着每次同一句子都对同一个单词进行相同的掩码，这样做没有充分利用到语料信息。为了避免在每个 epoch 中对同一实例进行相同的掩码，我们将训练数据重复10次，这样在预处理时对同一句子就可能有10种不同的掩码方式，这就是动态掩码方法。

### 改变输入格式，移除任务NSP

- 原 BERT 模型的输入数据是两个连接的文档片段，有50%的可能两个片段本身是相连的，有50%的可能两个片段本身不相连。RoBERTa 模型移除了任务 NSP，并且修改了输入格式，原文中说到，主要用到的格式有两种：第一种 Full-Sentences 格式，每个输入都包含一个或多个文档连续采样的句子（总长度不超过512），即输入会“跨文档”进行采样；第二种 Doc-Sentences 格式，每个输入都只采样一个文档连续的句子（总长度不超过512），和 Full-Sentences 相比，它的不同之处就在于不会进行跨文档采样。这两种输入格式中，Doc-Sentences 效果更好，但是这种格式下的数据输入可能少于512个 token，这就要求我们动态的调整 batch_size的大小，以使每个 batch 的标记总数相同。

### 更大的 batch size

- 文章经过实验分析，发现大的 batch size 使得模型可以获得更好的效果。

### Text Encoding

- 原 BERT 模型对 token 使用的是 WordPiece 编码，RoBERTa 模型对 token 采用 byte 级BPE编码。

## Deberta 

[论文链接](https://arxiv.org/abs/2006.03654)

- Deberta 模型是 2021 年由微软提出的模型，它的全名为 Decoding-enhanced BERT with disentangled attention，它主要针对 BERT, Roberta 模型从三个方面进行了改进：

    1. 解耦注意力机制，将每个 token 的词向量分别用两个向量表示，即内容向量和位置向量，而 Bert 的输入是一个向量。
    2. 增强掩码解码器，向模型中添加了 token 的绝对位置
    3. 一种用于微调的虚拟对抗训练方法（SiFT）

![](/img/note/202401261416.png)

### 解耦注意力机制（disentangled attention）

![](/img/note/202403052243.png)

- 与 BERT 不同，DeBERTa 中每个词使用两个对其内容和位置分别进行编码的向量来表示，使用分解矩阵分别根据词的内容和相对位置来计算词间的注意力权重。采用这种方法是因为：词对的注意力权重（衡量词与词之间的依赖关系强度）不仅取决于它们的内容，还取决于它们的相对位置。例如，「deep」和「learning」这两个词在同一个句子中接连出现时的依赖关系要比它们出现在不同句子中强得多。

### 增强型掩码解码器

![](/img/note/202403052242.png)

- 从前面解耦注意力机制我们知道模型在输入时考虑的是单词之间的相对位置，但是有时候仅仅只考虑相对位置是不够的，因此需要引入单词的绝对位置。在 Deberta 模型中，我们仍然是使用的 MLM，即对序列中某个单词进行掩码，根据上下文来推断这个单词是什么。下面举个例子，说明一下为什么要引入绝对位置。

- 例如句子「a new store opened beside the new mall」其中，「store」和「mall」在用于预测时被掩码操作。尽管两个词的局部语境相似，但是它们在句子中扮演的句法作用是不同的。（例如，句子的主角是「store」而不是「mall」）。这些句法上的细微差别在很大程度上取决于词在句子中的绝对位置，因此考虑单词在语言建模过程中的绝对位置是非常重要的。DeBERTa 在 softmax 层之前合并了绝对词位置嵌入，在该模型中，模型根据词内容和位置的聚合语境嵌入对被掩码的词进行解码。

### 用于微调的虚拟对抗训练方法

- SiFT 全称为 Scale-invariant Fine-Tuning，是本文提出的一种新的虚拟对抗学习方法，对抗学习，其实就是向数据中加入一些干扰信息，模型在真实数据和虚假数据中进行动态博弈，通过这样的训练可以增强模型的鲁棒性。

- 主要思想是：在标准化后的 word embedding 中添加扰动信息。当我们将 Deberta 应用到下游任务中时， SiFT 首先将 word embedding 变为标准化随机向量，然后在标准化后的向量中添加随机扰动信息。

## Deberta v3

### DeBERTa with Replaced token detection (RTD)

![](/img/note/202403061100.png)

### Gradient-Disentangled Embedding Sharing

1. Embedding Sharing (ES)

![](/img/note/202403061101.png)

2. No Embedding Sharing (NES)

![](/img/note/202403061102.png)

3. Gradient-Disentangled Embedding Sharing (GDES)

![](/img/note/202403061103.png)

## Text to Text Transfer Transformer（T5）

### 前言

- 该模型将所有自然语言问题都转化成文本到文本的形式，并用一个统一的模型解决。
- T5最核心的理念是：使用前缀任务声明及文本答案生成，统一所有自然语言处理任务的输入和输出。T5不需要对模型做任何改动，只需要提供下游任务的微调数据；不需要添加任何非线性层，唯一需要做的就是在输入数据前加上任务声明前缀。

![](/img/note/202403161631.png)

### 核心思想

- 模型结构采用了传统的 Transformer 结构

- 采用 BERT 式的训练方法，采用若干个连续词一起替换的效果最好
