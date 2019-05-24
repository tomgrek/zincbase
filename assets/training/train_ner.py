import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
from nn.ner import BertNER

def train(model, it, optimizer, criterion):
    model.train()
    for i, batch in enumerate(it):
        words, x, is_heads, tags, y, seqlens = batch
        optimizer.zero_grad()
        logits, y, _ = model(x, y)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"{i}: Loss: {loss.item()}")

def eval(model, it, f):
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(it):
            words, x, is_heads, tags, y, seqlens = batch
            _, _, y_hat = model(x, y)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall==0:
            f1 = 1.0
        else:
            f1 = 0

    os.remove("temp")

    print("precision = %.3f" % precision)
    print("recall = %.3f" % recall)
    print("f1 = %.3f" % f1)
    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertNER(len(VOCAB), device, args.finetuning).cuda()
    train_dataset = NerDataset(args.trainset)
    eval_dataset = NerDataset(args.validset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=pad)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.n_epochs+1):
        train(model, train_iter, optimizer, criterion)
        
        print("Evaluating at epoch: %d" % epoch)
        if not os.path.exists(args.logdir): os.makedirs(args.logdir)
        fname = os.path.join(args.logdir, str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname)
        torch.save(model.state_dict(), "%s.pt" % fname)
        print("Saved weights at %s" % (fname + '.pt')")

