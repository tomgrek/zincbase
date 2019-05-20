zincbase package
================

The main Zincbase package.

See README.md for some simple docs.

zincbase.zincbase module
------------------------

.. automodule:: zincbase.zincbase
    :members:
    :undoc-members:
    :show-inheritance:


Negative Examples
=================

Negative examples can be added to a Zincbase in two ways. Either:

* Prefix a rule with ~, such as `~likes(tom, sprouts)`
* Give it a truthiness attribute that's less than zero.

Concretely, this looks like:

.. code-block:: python

    kb.store('~likes(tom, sprouts)')
    kb.store('likes(tom, sprouts)', edge_attributes={'truthiness': -1})

Negative examples are fed in to the KG model as part of the usual training regime; you may
control the frequency that this happens with the `neg_ratio` kwarg of `KB.train_kg_model`.

Note that you can specify truthiness as something you want the model to learn to predict
(i.e. specify `pred_attributes=['truthiness']` when you call `build_kg_model`). But, negative
truthiness takes the example out of the normal flow of this: only examples with 0 <= truthiness <= 1
are part of 'proper' training where the predicate prediction is taken into account.

Anecdotally, negative examples do not help much, or only help with small datasets.
