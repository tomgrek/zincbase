"""Runs evaluation on the Countries S3 dataset
to reproduce the results from the RotatE paper.

Follows the paper's authors' methodology for calculating the AUC PR.
"""

import csv

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from utils.calc_auc_roc import calc_auc_roc
from zincbase import KB

kb = KB()

Xs = []
Ys = []
csvfile = csv.reader(open('./assets/countries_s3_test.csv', 'r'), delimiter='\t')
for row in csvfile:
    Xs.append([row[0], row[1]])
    Ys.append(row[2])

kb.from_csv('./assets/countries_s3_train.csv', delimiter='\t')

kb.build_kg_model(cuda=True, embedding_size=1000, gamma=0.1)
kb.train_kg_model(steps=40000, batch_size=512, lr=0.000002, neg_to_pos=64)

y_true = []
sample = []
for ((head, relation), tail) in zip(Xs, Ys):
    for candidate_region in ['oceania', 'asia', 'europe', 'africa', 'americas']:
        y_true.append(1 if candidate_region == tail else 0)
        sample.append((kb._entity2id[head], kb._relation2id[relation], kb._entity2id[candidate_region]))
sample = torch.LongTensor(sample).cuda()
with torch.no_grad():
    y_score, _ = kb._kg_model(sample)
    y_score = y_score.squeeze(1).cpu().numpy()
y_true = np.array(y_true)
auc_pr = average_precision_score(y_true, y_score)
print('AUC PR: %2.2f' % auc_pr)
print('^^ Should be ~0.95 to match the paper.')

# I somewhat prefer this method of evaluation, which does 2 distinct hops
# to demonstrate multi hop reasoning. But it tends
# to come up a bit lower than what the paper has.

# Ys_pred = []
# for x in Xs:
#     subregion = kb.get_most_likely(x[0], x[1], '?', k=1)[0]['triple'][2]
#     region = kb.get_most_likely(subregion, x[1], '?', k=1)[0]['triple'][2]
#     Ys_pred.append(region)

# aucroc = calc_auc_roc(Ys, Ys_pred)
# print(aucroc) 