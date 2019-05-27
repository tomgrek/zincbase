import csv
import re

from tqdm import tqdm

from zincbase.utils.string_utils import cleanse

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
