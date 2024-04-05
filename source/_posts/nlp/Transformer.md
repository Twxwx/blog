---
title: Transformer
date: 2024-03-02 23:03:21
categories:
    - NLP
tags:
---

## 概念理解

- 参数量越小需要显存越少，计算量越小算的时间越少

### 参数量

- 参数量指的是深度神经网络中需要学习的参数数量。在深度学习中，每个神经元都有一个权重，这些权重是需要通过训练来确定的。深度神经网络中的参数量是指所有权重的数量之和，其中包括连接输入和输出的权重，以及所有神经元的偏置项。

### 计算量

- 计算量指的是在模型中进行前向传播和反向传播所需的浮点运算次数(通常将相乘后相加看做一次操作，乘法消耗大于加法消耗)。在深度学习中，神经网络的计算量通常是指卷积、乘法和加法操作的数量。由于深度神经网络具有非常大的计算量，因此需要强大的计算能力才能对其进行训练和推理。

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
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int = 512, max_seq_len: int = 2048, base: int = 10000) -> None:
        super().__init__()
        self.d_model = d_model
        inv_freq_half = 1.0 / (
            base ** torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        )
        inv_freq = torch.arange(0, d_model, dtype=inv_freq_half.dtype)
        inv_freq[..., 0::2] = inv_freq_half
        inv_freq[..., 1::2] = inv_freq_half
        pos = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        pe = torch.einsum("i, j -> ij", pos, inv_freq)
        pe[..., 0::2] = pe[..., 0::2].sin()
        pe[..., 1::2] = pe[..., 1::2].cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        pe = self.pe[:seq_len].to(dtype=x.dtype)
        return x + pe
```

## Mask

- 在 Encoder 和 Decoder 中，Mask 会遮住用于 Padding 的位置。
- 在 Decoder 中，Mask 会遮住预测剩余位置，防止 Dcoder 提前得到信息。

## Multi-Head Attention

![](/img/note/202403032101.png)

![](/img/note/202403042000.png)

``` python 

class MultiheadAttention(nn.Module):
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

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz = q.shape[0]

        # translate to [bsz, heads, seq_len, d_k]
        q = self.q_proj(q).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.k_proj(v).view(bsz, -1, self.heads, self.d_k).transpose(1, 2)

        # calculate attention
        # out: [bsz, heads, seq_len, d_k]
        out = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concat multi-heads
        # concat: [bsz, seq_len, d_model]
        concat = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        output = self.o_proj(concat)
        return output
```

![](/img/note/202403042001.png)

``` python
    def attention(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            d_k: int, 
            mask: Optional[torch.Tensor] = None, 
            dropout: Optional[nn.Dropout] = None
        ):
        # calculate the scores
        # q: [bsz, heads, seq_len, d_k]
        # k: [bsz, heads, d_k, seq_len]
        # v: [bsz, heads, seq_len, d_k]
        # scores: [bsz, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            # tanslate [bsz, seq_len, seq_len] to [bsz, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # scores: [bsz, heads, seq_len, seq_len]
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        # output: [bsz, heads, seq_len, d_k]
        output = torch.matmul(scores, v)
        return output

```

### Attention 的计算开销

![](/img/note/202403032102.png)

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
        self.pe = PositionalEncoder(d_model, max_seq_len)
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
        self.pe = PositionalEncoder(d_model, max_seq_len)
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

if __name__ == '__main__':
    bsz = 4
    max_seq_len = 1024
    src_vocab = 128
    tgt_vocab = 64
    N = 3
    d_ff = 512
    model = Transformer(src_vocab, tgt_vocab, N=N, max_seq_len=max_seq_len, d_ff=d_ff)
    # (bsz, max_seq_len)
    src = torch.randint(low=0, high=src_vocab, size=(bsz, max_seq_len))
    # (bsz, max_seq_len)
    tgt = torch.randint(low=0, high=tgt_vocab, size=(bsz, max_seq_len))
    src_mask = torch.ones(size=(bsz, max_seq_len, max_seq_len))
    tgt_mask = torch.ones(size=(bsz, max_seq_len, max_seq_len))
    res = model(src, tgt, src_mask, tgt_mask)
    print(f"Output data shape is: {res.shape}")

```




