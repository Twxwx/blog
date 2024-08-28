---
title: Transformer
date: 2024-03-02 23:03:21
categories:
    - NLP
tags:
---

## 模型架构

![](/img/note/202403032100.png)

## Embedding

- Embedding 可以将高维的离散文本数据映射到低维的连续向量空间。这不仅减小了输入数据的维度，也有助于减少数据的稀疏性，提高模型的性能和效率。
- 同时，词嵌入可以捕捉单词之间的语义关系，相似的单词在嵌入空间中会更接近。

``` python
class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)
```

## Positional Encoding

![](/img/note/202403032103.png)

``` python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model = 512, max_len = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe:[max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
```

## Mask

- 在 Encoder 和 Decoder 中，Mask 会遮住用于 Padding 的位置。
- 在 Decoder 中，Mask 会遮住预测剩余位置，防止 Dcoder 提前得到信息。

## Multi-Head Attention

![](/img/note/202403032101.png)

![](/img/note/202403042000.png)

![](/img/note/202403042001.png)

``` python 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = self.d_model // self.heads

        if self.heads * self.d_k != self.d_model:
            raise ValueError(
                f"d_model must be divisible by heads (got `d_model`: {self.d_model}"
                f" and `heads`: {self.heads})."
            )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz = q.shape[0]

        # (bsz, seq_len, d_model) -> (bsz, seq_len, heads, d_k) -> (bsz, heads, seq_len, d_k) 
        q = self.q_proj(q).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.k_proj(v).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 scores: [bsz, heads, seq_len, seq_len] 
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        # 生成 casual mask
        if mask is not None:
            # mask: [bsz, max_seq_len, max_seq_len] -> [bsz, 1, max_seq_len, max_seq_len]
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        scores = F.softmax(attn, dim = -1)
        # [bsz, heads, seq_len, d_k]
        out = torch.matmul(scores, v)
        # concat: [bsz, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        output = self.o_proj(out)
        return output
```

### 为什么在进行 softmax 之前需要对 attention 进行 scaled 以及为什么除以 d_k 的平方根？

- 向量点积后的结果数量级变大，经历过softmax函数的归一化之后，softmax将几乎全部的概率分布都分配给了最大值对应的标签，后续反向传播的过程中梯度会很小，进行scaled能够缓解这种情况。
- 假设q和k的各个分量是互相独立的随机变量，均值为0，方差为1。点积后q·k的均值为0，方差为 d_k，所以对于点积后的数除以 sqrt(d_k)，相当于方差除以d_k，使得方差缩放为1。

### 为什么使用多头注意力？

- 多头自注意力可以得到更具有解释性的模型。不仅每个注意力头清晰地学会了执行不同的任务，许多注意力头似乎表现出与句子的句法和语义结构相关的行为。

## Feed-Forward Network

![](/img/note/202403042002.png)

``` python
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```

## EncoderLayer

``` python 
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, heads: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiheadAttention(d_model, heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.dropout1(self.attn(x, x, x, mask))
        x = self.norm1(x)

        x = x + self.dropout2(self.ffn(x))
        x = self.norm2(x)
        return x  
```

## DecoderLayer

``` python 
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, heads: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn1 = MultiheadAttention(d_model, heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = MultiheadAttention(d_model, heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: torch.Tensor, 
            enc_output: torch.Tensor, 
            src_mask: torch.Tensor, 
            tgt_mask: torch.Tensor
        ):
        x = x + self.dropout1(self.attn1(x, x, x, tgt_mask))
        x = self.norm1(x)

        x = x + self.dropout2(self.attn2(x, enc_output, enc_output, src_mask))
        x = self.norm2(x)

        x = x + self.dropout3(self.ffn(x))
        x = self.norm3(x)
        return x
```

## Encoder

``` python 
class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int, 
            N: int = 6,  
            d_model: int = 512, 
            max_seq_len: int = 2048, 
            heads: int = 8, 
            d_ff: int = 2048, 
            dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)]
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

## Decoder

``` python 
class Decoder(nn.Module):
    def __init__(
            self,  
            vocab_size: int, 
            N: int = 6,
            d_model: int = 512, 
            max_seq_len: int = 2048, 
            heads: int = 8, 
            d_ff: int = 2048, 
            dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)]
        )

    def forward(
            self, 
            tgt: torch.Tensor, 
            enc_output: torch.Tensor, 
            src_mask: torch.Tensor, 
            tgt_mask: torch.Tensor
        ):
        x = self.embed(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
```

## Transformer

``` python 
class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab: int, 
            tgt_vocab: int, 
            N: int, 
            d_model: int = 512, 
            max_seq_len: int = 2048, 
            heads: int = 8, 
            d_ff: int = 2048, 
            dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab, N, d_model, max_seq_len, heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, N, d_model, max_seq_len, heads, d_ff, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor, 
            src_mask: torch.Tensor, 
            tgt_mask: torch.Tensor
        ): 
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = F.softmax(self.out(dec_output), dim=-1)
        return output
```

## 输入输出测试

``` python
if __name__ == "__main__":
    bsz = 4
    heads = 8
    max_seq_len = 1024
    src_vocab = 128
    tgt_vocab = 64
    N = 3
    d_ff = 2048
    d_model = 512
    model = Transformer(src_vocab, tgt_vocab, N=N, d_model=d_model, max_seq_len=max_seq_len, heads=heads, d_ff=d_ff)
    # (bsz, max_seq_len)
    src = torch.randint(low=0, high=src_vocab, size=(bsz, max_seq_len))
    # (bsz, max_seq_len)
    tgt = torch.randint(low=0, high=tgt_vocab, size=(bsz, max_seq_len))
    src_mask = torch.ones(size=(bsz, max_seq_len, max_seq_len))
    tgt_mask = torch.ones(size=(bsz, max_seq_len, max_seq_len))
    res = model(src, tgt, src_mask, tgt_mask)
    print(f"Output data shape is: {res.shape}")
```

## 参数量

- 参数量指的是深度神经网络中需要学习的参数数量。在深度学习中，每个神经元都有一个权重，这些权重是需要通过训练来确定的。深度神经网络中的参数量是指所有权重的数量之和，其中包括连接输入和输出的权重，以及所有神经元的偏置项。

![](/img/note/202408082049.png)

![](/img/note/202408082050.png)

## 计算量

- 计算量指的是在模型中进行前向传播和反向传播所需的浮点运算次数(通常将相乘后相加看做一次操作，乘法消耗大于加法消耗)。在深度学习中，神经网络的计算量通常是指卷积、乘法和加法操作的数量。由于深度神经网络具有非常大的计算量，因此需要强大的计算能力才能对其进行训练和推理。

![](/img/note/202408082102.png)

![](/img/note/202408082103.png)

![](/img/note/202408082104.png)

## 中间激活值分析

- 除了模型参数、梯度、优化器状态外，占用显存的大头就是前向传递过程中计算得到的中间激活值了，需要保存中间激活以便在后向传递计算梯度时使用。这里的激活（activations）指的是：前向传递过程中计算得到的，并在后向传递过程中需要用到的所有张量。这里的激活不包含模型参数和优化器状态，但包含了dropout操作需要用到的mask矩阵。

- 在一次训练迭代中，模型参数（或梯度）占用的显存大小只与模型参数量和参数数据类型有关，与输入数据的大小是没有关系的。优化器状态占用的显存大小也是一样，与优化器类型有关，与模型参数量有关，但与输入数据的大小无关。**中间激活值与输入数据的大小（批次大小 b 和序列长度 s）是成正相关的**。

## KV cache

- 在推断阶段，transformer模型加速推断的一个常用策略就是使用 KV cache。一个典型的大模型生成式推断包含了两个阶段：

    1. 预填充阶段：输入一个prompt序列，为每个transformer层生成 key cache和 value cache。

    2. 解码阶段：使用并更新KV cache，一个接一个地生成词，当前生成的词依赖于之前已经生成的词。
