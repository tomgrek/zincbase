from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def calc_auc_roc(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)
