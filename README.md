[![CircleCI](https://circleci.com/gh/tomgrek/zincbase.svg?style=svg)](https://circleci.com/gh/tomgrek/zincbase)
[![DOI](https://zenodo.org/badge/183831265.svg)](https://zenodo.org/badge/latestdoi/183831265)
[![Documentation Status](https://readthedocs.org/projects/zincbase/badge/?version=latest)](https://zincbase.readthedocs.io/en/latest/?badge=latest)

<img src="https://user-images.githubusercontent.com/2245347/57199440-c45daf00-6f33-11e9-91df-1a6a9cae6fb7.png" width="140" alt="Zincbase logo">

ZincBase is a state of the art knowledge base. It does the following:

* Extract facts (aka triples and rules) from unstructured data/text
* Store and retrieve those facts efficiently
* Build them into a graph
* Provide ways to query the graph, including via bleeding-edge graph neural networks.

Zincbase exists to answer questions like "what is the probability that Tom likes LARPing", or "who likes LARPing", or "classify people into LARPers vs normies":

<img src="https://user-images.githubusercontent.com/2245347/57595488-2dc45b80-74fa-11e9-80f4-dc5c7a5b22de.png" width="320" alt="Example graph for reasoning">

It combines the latest in neural networks with symbolic logic (think expert systems and prolog) and graph search.

View full documentation [here](https://zincbase.readthedocs.io).

## Quickstart

```
from zincbase import KB
kb = KB()
kb.store('eats(tom, rice)')
for ans in kb.query('eats(tom, Food)'):
    print(ans['Food']) # prints 'rice'

...
# The included assets/countries_s1_train.csv contains triples like:
# (namibia, locatedin, africa)
# (lithuania, neighbor, poland)

kb = KB()
kb.from_csv('./assets/countries.csv')
kb.build_kg_model(cuda=False, embedding_size=40)
kb.train_kg_model(steps=2000, batch_size=1, verbose=False)
kb.estimate_triple_prob('fiji', 'locatedin', 'melanesia')
0.8467
```

# Requirements

* Python 3
* Libraries from requirements.txt
* GPU preferable for large graphs but not required

# Installation

`pip install -r requirements.txt`

_Note:_ Requirements might differ for PyTorch depending on your system.

# Testing

```
python test/test_main.py
python test/test_graph.py
python test/test_lists.py
python test/test_nn_basic.py
python test/test_nn.py
python test/test_neg_examples.py
python test/test_truthiness.py
python -m doctest zincbase/zincbase.py
```

# Validation

There is a script to evaluate that ZincBase gets at least as good
performance on the Countries dataset as the original (2019) RotatE paper. From the repo's
root directory:

```
python examples/eval_countries_s3.py
```

It tests the hardest Countries task and prints out the AUC ROC, which should be
~ 0.95 to match the paper. It takes about 30 minutes to run on a modern GPU.

## Building documentation

From docs/ dir: `make html`. If something changed a lot: `sphinx-apidoc -o . ..`

# TODO

* Add documentation
* "solidify" method that takes bindings output from a rule and adds them to graph as concrete atoms so NN can work on them.
* to_csv method
* utilize postgres as backend triple store
* The to_csv/from_csv methods do not yet support node attributes.
* Add relation extraction from arbitrary unstructured text
* Address inconsistencies with Prolog syntax vs basic triples
* Add relation attributes (e.g. 'formerly') without having to add the relation as 100% fact first.
* Add documentation regarding adding a truthiness attribute
* Better interface to negative examples
* Add 'real name' attribute to nodes when adding them to the KB. It should track the original name before we strip out spaces and special chars.

# References & Acknowledgements

[Theo Trouillon. Complex-Valued Embedding Models for Knowledge Graphs. Machine Learning[cs.LG]. Université Grenoble Alpes, 2017. English. ffNNT : 2017GREAM048](https://tel.archives-ouvertes.fr/tel-01692327/file/TROUILLON_2017_archivage.pdf)

[L334: Computational Syntax and Semantics -- Introduction to Prolog, Steve Harlow](http://www-users.york.ac.uk/~sjh1/courses/L334css/complete/complete2li1.html)

[Open Book Project: Prolog in Python, Chris Meyers](http://www.openbookproject.net/py4fun/prolog/intro.html)

[Prolog Interpreter in Javascript](https://curiosity-driven.org/prolog-interpreter)

[RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space, Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang, International Conference on Learning Representations, 2019](https://openreview.net/forum?id=HkgEQnRqYQ)

# Citing

If you use this software, please consider citing:

```
@software{zincbase,
  author = {{Tom Grek}},
  title = {ZincBase: A state of the art knowledge base},
  url = {https://github.com/tomgrek/zincbase},
  version = {0.1.1},
  date = {2019-05-12}
}

```