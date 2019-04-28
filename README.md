[![CircleCI](https://circleci.com/gh/tomgrek/xinkbase.svg?style=svg)](https://circleci.com/gh/tomgrek/xinkbase)
[![DOI](https://zenodo.org/badge/183831265.svg)](https://zenodo.org/badge/latestdoi/183831265)

# XinKBase

A state of the art knowledge base.

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
python test/test_lists.py
python -m doctest xinkbase/xinkbase.py
```

# TODO

* Reimplement search as a generator

# References & Acknowledgements

[L334: Computational Syntax and Semantics -- Introduction to Prolog, Steve Harlow](http://www-users.york.ac.uk/~sjh1/courses/L334css/complete/complete2li1.html)

[Open Book Project: Prolog in Python, Chris Meyers](http://www.openbookproject.net/py4fun/prolog/intro.html)

[Prolog Interpreter in Javascript](https://curiosity-driven.org/prolog-interpreter)

[RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space, Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang, International Conference on Learning Representations, 2019](https://openreview.net/forum?id=HkgEQnRqYQ)

# Citing

If you use this software, please consider citing:

```
@software{xinkbase,
  author = {{Tom Grek}},
  title = {XinKBase: A state of the art knowledge base},
  url = {https://github.com/tomgrek/xinkbase},
  version = {0.0.1},
  date = {2019-04-27},
}
```