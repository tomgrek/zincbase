import csv
import re

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from utils.string_utils import cleanse

def calc_mrr(kb, test_file, delimiter=',', header=None, size=None):
    """Calculate the mean reciprocal rank using a test set."""
    with open(test_file) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader, None)
        i = 0
        test_triples = []
        for row in reader:
            pred = cleanse(pred)
            sub = cleanse(sub)
            ob = cleanse(ob)
            if not (sub.replace('_','').isalnum() and ob.replace('_','').isalnum()):
                continue
            test_triples.append((sub, pred, ob))
    mrr = 0
    print('Test set size: {}'.format(len(test_triples)))
    if not size:
        size = len(test_triples)
    for t in tqdm(test_triples[:size]):
        # TODO, this is inefficient, not batched.
        try:
            ob = t[2]
            ranks = kb.get_most_likely(t[0], t[1], '?', k=100)
            ranked_ents = [x['triple'][2] for x in ranks]
            if ob not in ranked_ents:
                continue
            mrr += 1 / (ranked_ents.index(ob) + 1)
        except:
            continue
    return mrr / size

def calc_auc_roc(truth, pred, average="macro"):
    """Calculate the Area-Under-the-Curve Receiver Operating Characteristic

    A funny measure that combines precision and recall. Sklearn can't agree how
    to implement it for multiclass; this version is from fbrundu on
    https://github.com/scikit-learn/scikit-learn/issues/3298
    """
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)
