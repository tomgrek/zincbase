import numpy as np
import torch
from torch.utils import data

from nn.ner import TOKEN_TYPES, tag2idx, idx2tag
from nn.tokenizer import get_tokenizer

class NERDataset(data.Dataset):
    def __init__(self, filepath):
        entries = open(filepath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], []
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []

        for w, t in zip(words, tags):
            tokens = get_tokenizer().tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = get_tokenizer().convert_tokens_to_ids(tokens)
            is_head = [1] + [0]*(len(tokens) - 1)
            t = [t] + ["<PAD>"] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]
            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        seqlen = len(y)
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    y = f(-2, maxlen)

    return words, torch.LongTensor(x), is_heads, tags, torch.LongTensor(y), seqlens


