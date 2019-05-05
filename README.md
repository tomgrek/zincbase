[![CircleCI](https://circleci.com/gh/tomgrek/zincbase.svg?style=svg)](https://circleci.com/gh/tomgrek/zincbase)
[![DOI](https://zenodo.org/badge/183831265.svg)](https://zenodo.org/badge/latestdoi/183831265)
[![Documentation Status](https://readthedocs.org/projects/zincbase/badge/?version=latest)](https://zincbase.readthedocs.io/en/latest/?badge=latest)

<img src="https://user-images.githubusercontent.com/2245347/57199440-c45daf00-6f33-11e9-91df-1a6a9cae6fb7.png" width="140" alt="Zincbase logo">

ZincBase is a state of the art knowledge base.

It combines symbolic logic (think expert systems), graph search, and the latest in neural networks.

View full documentation [here](https://zincbase.readthedocs.io).

## Quickstart

```
from zincbase import KB
kb = KB()
kb.store('eats(tom, rice)')
for ans in kb.query('eats(tom, Food)'):
    print(ans['Food']) # prints 'rice'
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
python -m doctest zincbase/zincbase.py
```

## Building documentation

From docs/ dir: `make html`. If something changed a lot: `sphinx-apidoc -o . ..`

# TODO

* Add documentation
* "solidify" method that takes bindings output from a rule and adds them to graph as concrete atoms so NN can work on them.
* refactor the .attr method to be prolog style ie owns_a_raincoat(tom)
* to_csv method
* utilize postgres as backend triple store
* attributes for nodes / relations
* The to_csv/from_csv methods do not yet support node attributes.
* Add relation extraction from arbitrary unstructured text

# References & Acknowledgements

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
  version = {0.0.1},
  date = {2019-04-27},
}

```