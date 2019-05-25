import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from zincbase.nn.ner import BertNER, TOKEN_TYPES, tag2idx, idx2tag
from dataloader import NERDataset, pad

def train(model, it, optimizer, criterion):
    model.train()
    for i, batch in enumerate(it):
        words, x, is_heads, tags, y, seqlens = batch
        y = y.to(model.device)
        optimizer.zero_grad()
        logits, _ = model(x)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i}: loss={loss.item()}")

def eval(model, it, f):
    model.eval()
    Tokens, Is_heads, Tags, Y, Y_pred = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(it):
            tokens, x, is_heads, tags, y, seqlens = batch
            _, y_pred = model(x)
            Tokens.extend(tokens)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_pred.extend(y_pred.cpu().numpy().tolist())

    # with open("temp", 'w') as fout:
    record = []
    for tokens, is_heads, tags, y_pred in zip(Tokens, Is_heads, Tags, Y_pred):
        y_pred = [pred for head, pred in zip(is_heads, y_pred) if head == 1]
        preds = [idx2tag[pred] for pred in y_pred]
        for w, t, p in zip(tokens.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
            record.append((w, t, p))

    y_true = np.array([tag2idx[item[1]] for item in record])
    y_pred = np.array([tag2idx[item[2]] for item in record])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"Proposed: {num_proposed}")
    print(f"Correct: {num_correct}")
    print(f"Gold: {num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0
    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1: %.3f" % f1)
    return precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertNER(len(TOKEN_TYPES), device, True).cuda()
    train_dataset = NERDataset(args.trainset)
    eval_dataset = NERDataset(args.validset)

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.n_epochs+1):
        train(model, train_iter, optimizer, criterion)
        print("Evaluating at epoch: %d" % epoch)
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        filename = os.path.join(args.logdir, str(epoch))
        precision, recall, f1 = eval(model, eval_iter, filename)
        torch.save(model.state_dict(), "%s.bin" % filename)
        print("Saved weights at %s" % (filename + '.bin'))

