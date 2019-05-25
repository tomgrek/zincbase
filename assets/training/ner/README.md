# Script for training an NER model

This requires having the CONLL2003 dataset, which can be found at:

https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en

# Example invocation

(From the root dir of Zincbase)

```
python assets/training/ner/train_ner.py --logdir /tmp --batch_size 12 --lr 1e-4 --n_epochs 5 --trainset conll2003/train.txt --validset conll2003/valid.txt
```