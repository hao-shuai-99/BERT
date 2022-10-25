import torch
import numpy as np
import argparse
from utils import *
from torch.utils.data import DataLoader
from Bert import MyDataset, BERT


def train(args):
    sentences = load_data(args.path)  # 加载数据
    word_list = list(set("".join(sentences).split()))  # 生成词汇表
    word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # PAD, CLS, SEP token

    for i, w in enumerate(word_list):  # words to tokens
        word2idx[w] = i + 4
    idx2word = {i: w for i, w in enumerate(word2idx)}  # tokens to words

    vocab_size = len(word2idx)  # 词汇表大小
    args.vocab_size = vocab_size
    token_list = []
    for sentence in sentences:  # sentences tokens
        arr = []
        for s in sentence.split():
            arr.append(word2idx[s])
        token_list.append(arr)

    batch = make_data(args, sentences, word2idx, token_list)

    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)  # unzip
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)

    loader = DataLoader(MyDataset(input_ids, segment_ids, masked_tokens, masked_pos, isNext),
                        args.batch_size, shuffle=True)

    model = BERT(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)  # [6, 5, 40], [6, 2]
            loss_lm = criterion(logits_lm.view(-1, args.vocab_size), masked_tokens.view(-1))  # [30, 40]  [30]
            loss_lm = (loss_lm.float())
            # print(loss_lm)
            loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification
            loss = loss_lm + loss_clsf
            if (epoch + 1) % 10 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss = ', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, batch


def test(model, batch, args):
    # Predict mask tokens and isNext
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]

    logits_lm, logits_clsf = model(torch.LongTensor([input_ids]),
                                   torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))

    logits_lm = logits_lm.data.max(2)[1][0].tolist()
    print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
    print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1][0].tolist()
    print('isNext : ', True if isNext else False)
    print('predict isNext : ', True if logits_clsf else False)


if __name__ == '__main__':
    '''
    code by hao_shuai
    '''
    parser = argparse.ArgumentParser(description="BERT")

    parser.add_argument('--path', type=str, default='./data.txt')
    parser.add_argument('--maxlength', type=int, default=30, help='The max length of sentences.')
    parser.add_argument('--batch_size', type=int, default=6, help='')
    parser.add_argument('--max_pred', type=int, default=5, help='The max prediction words.')
    parser.add_argument('--n_layers', type=int, default=6, help='Encoder layers.')
    parser.add_argument('--n_heads', type=int, default=12, help='The number of heads of Encoder layer.')
    parser.add_argument('--d_model', type=int, default=768, help='(Token, Segment, Position)Embeddings dimension.')
    parser.add_argument('--d_ff', type=int, default=768 * 4, help='FeedForward dimension.')
    parser.add_argument('--d_k', type=int, default=64, help='K, Q dimension.')
    parser.add_argument('--d_v', type=int, default=64, help='V dimension.')
    parser.add_argument('--n_segments', type=int, default=2, help='')
    parser.add_argument('--vocab_size', type=int, default=None, help='The total numbers of words.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--epoch', type=int, default=50, help='')

    args = parser.parse_args()

    model, batch = train(args)

    test(model, batch, args)

