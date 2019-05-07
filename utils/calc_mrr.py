import csv
from tqdm import tqdm

def calc_mrr(kb, test_file, delimiter=',', header=None, size=None):
    """Calculate the mean reciprocal rank using a test set."""
    with open(test_file) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader, None)
        i = 0
        test_triples = []
        for row in reader:
            pred = row[1].replace('.', '').replace('(', '').replace(')','').replace('/','_')
            sub = row[0].replace(' ','').replace('.', '').replace('(', '').replace(')','').replace('/','_')
            sub = sub[0].lower() + sub[1:]
            ob = row[2].replace(' ','').replace('.', '').replace('(', '').replace(')','').replace('/','_')
            ob = ob[0].lower() + ob[1:]
            if not (sub.replace('_','').isalnum() and ob.replace('_','').isalnum()):
                continue
            test_triples.append((sub, pred, ob))
    mrr = 0
    print('Test set size: {}'.format(len(test_triples)))
    if not size:
        size = len(test_triples)
    for t in tqdm(test_triples[:size]):
        try:
            try:
                ranks = kb.get_most_likely(sub, pred, '?', k=20)
            except:
                ranks = kb.get_most_likely(sub, pred, '?', k=2)
            ranked_ents = [x['triple'][2] for x in ranks]
            if ob not in ranked_ents:
                continue
            mrr += 1 / (ranked_ents.index(ob) + 1)
        except:
            continue
    return mrr / len(test_triples)
