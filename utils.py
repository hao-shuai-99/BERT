import torch
import numpy as np
from string import punctuation
import random
import math


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            line = line.lower()
            for c in punctuation:
                line = line.replace(c, ' ')
            data.append(line)
    return data


def make_data(args, sentences, word2idx, token_list):
    batch = []
    positive = negative = 0
    while positive != args.batch_size / 2 or negative != args.batch_size / 2:
        tokens_a_index, tokens_b_index = random.randrange(len(sentences)), random.randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        n_segments_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(args.max_pred, max(1, int(len(input_ids) * 0.15)))  # 15%
        cand_maked_pos = [i for i, token in enumerate(input_ids) if
                          token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # MASK时忽略CLS和SEP
        random.shuffle(cand_maked_pos)
        masked_token, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_token.append(input_ids[pos])
            if random.random() < 0.8:  # 生成0-1随机数
                input_ids[pos] = word2idx['[MASK]']  # 80% MASK
            elif random.random() > 0.9:  # 10% 随机替换
                index = random.randint(0, args.vocab_size - 1)
                while index < 4:  # 不能包括 ‘CLS’， ‘SEP’， ‘PAD’
                    index = random.randint(0, args.vocab_size - 1)
                input_ids[pos] = index

        # Zero Paddings input_id padding to max_length
        n_pad = args.maxlength - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)  # [0] * n_pad
        n_segments_ids.extend([0] * n_pad)

        # Zero Paddings 预测词数, masked_token padding to max_pred
        if args.max_pred > n_pred:
            n_pad = args.max_pred - n_pred
            masked_token.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < args.batch_size/2:
            batch.append([input_ids, n_segments_ids, masked_token, masked_pos, True])  # zip
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < args.batch_size/2:
            batch.append([input_ids, n_segments_ids, masked_token, masked_pos, False])
            negative += 1

    return batch


def gelu(x):
    # 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))  OpenAI GPT-2
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]

    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]