"""Calculate the Area-Under-the-Curve Receiver Operating Characteristic

A funny measure that combines precision and recall. Sklearn can't agree how
to implement it for multiclass; this version is from fbrundu on
https://github.com/scikit-learn/scikit-learn/issues/3298
"""

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def calc_auc_roc(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)
