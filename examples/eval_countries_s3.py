"""Runs evaluation on the Countries S3 dataset
to reproduce the results from the RotatE paper"""

import csv

from utils.calc_auc_roc import calc_auc_roc
from zincbase import KB

kb = KB()

Xs = []
Ys = []
csvfile = csv.reader(open('./assets/countries_s3_test.csv', 'r'), delimiter='\t')
for row in csvfile:
    Xs.append((row[0], row[1]))
    Ys.append(row[2])

kb.from_csv('./assets/countries_s3_train.csv', delimiter='\t')

kb.build_kg_model(cuda=True, embedding_size=1000, gamma=0.1)
kb.train_kg_model(steps=40000, batch_size=512, lr=0.000002, neg_to_pos=64)

Ys_pred = []
for x in Xs:
    subregion = kb.get_most_likely(x[0], x[1], '?', k=1)[0]['triple'][2]
    region = kb.get_most_likely(subregion, x[1], '?', k=1)[0]['triple'][2]
    Ys_pred.append(region)

aucroc = calc_auc_roc(Ys, Ys_pred)
print(aucroc) # should be ~ 0.95 to match the paper