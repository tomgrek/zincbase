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

# References

[http://www-users.york.ac.uk/~sjh1/courses/L334css/complete/complete2li1.html](L334: Computational Syntax and Semantics -- Introduction to Prolog, Steve Harlow)
[http://www.openbookproject.net/py4fun/prolog/intro.html](Open Book Project: Prolog in Python, Chris Meyers)
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
[https://curiosity-driven.org/prolog-interpreter](Prolog in JS with nice Einstein puzzle example)

# Citing

