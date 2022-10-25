import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from utils import *


class MyDataset(data.Dataset):  # 自定义dataset
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):  # 每次返回一个sample
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]


class Embedding(nn.Module):  # 初始化 Embeddings
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(args.vocab_size, args.d_model)  # token embedding
        self.pos_embed = nn.Embedding(args.maxlength, args.d_model)  # position embedding
        self.seg_embed = nn.Embedding(args.n_segments, args.d_model)  # segment embedding
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x, seg):  # x, seg [batch_size, maxlength]
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)  # [batch_size, seq_len, d_model]

        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, args):  # Q:[batch_size, n_heads, seq_len, d_k]  attn_mask:[batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args.d_k)  # [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, float('-inf'))  # mask中取值为True位置对应于scores的相应位置用value填充
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, seq_len, d_k]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.d_k = args.d_k
        self.n_heads = args.n_heads
        self.d_v = args.d_v
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

    def forward(self, Q, K, V, attn_mask):
        # Q:[batch_size, seq_len, d_model], K:[batch_size, seq_len, d_model], V:[batch_size, seq_len, d_model] attn_mask:[batch_size, maxlength, maxlength]
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.args)  # [batch_size, n_heads, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)  # [6, 30, 768]  拼接
        output = nn.Linear(self.n_heads * self.d_v, self.d_model)(context)  # [batch_size, seq_len, d_model]
        return nn.LayerNorm(self.d_model)(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = args.d_model
        self.fc1 = nn.Linear(args.d_model, args.d_ff)
        self.fc2 = nn.Linear(args.d_ff, args.d_model)

    def forward(self, x):
        residual = x
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        output = self.fc2(gelu(self.fc1(x)))
        return nn.LayerNorm(self.d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # Q, K, V 形状相同
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.embedding = Embedding(args)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])  # 保存modules
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(args.d_model, 2)
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):  # input_ids:[batch_size, maxlength], segment_ids:[batch_size, maxlength], masked_pos:[batch_size, mask单词数]
        output = self.embedding(input_ids, segment_ids)  # [batch_size, maxlength, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlength, maxlength]
        for layer in self.layers:
            # output: [batch_size, maxlength, d_model]
            output = layer(output, enc_self_attn_mask)

        h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [6, 5, 40]
        return logits_lm, logits_clsf
