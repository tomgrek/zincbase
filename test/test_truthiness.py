"""Test the combination of edge attributes and negative examples."""

import context

from zincbase import KB

kb = KB()
kb.seed(555)

kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')

# specifying both a ~ and a truthiness < 0 is probably unnecessary.
kb.store('~locatedin(canada, africa)', {'truthiness': -1.})

kb.build_kg_model(cuda=False, embedding_size=100, pred_attributes=['truthiness'])

kb.train_kg_model(steps=1000, batch_size=4, neg_ratio=0.01)

canada_in_africa = kb.estimate_triple_prob('canada', 'locatedin', 'africa')
canada_in_asia = kb.estimate_triple_prob('canada', 'locatedin', 'asia')
canada_in_america = kb.estimate_triple_prob('canada', 'locatedin', 'northern_america')
assert 2 * canada_in_africa < canada_in_asia
assert canada_in_america > 2 * canada_in_asia

clafrica_truthiness = kb.estimate_triple_prob_with_attrs('canada', 'locatedin', 'africa', 'truthiness')
clasia_truthiness = kb.estimate_triple_prob_with_attrs('canada', 'locatedin', 'asia', 'truthiness')

assert clafrica_truthiness < clasia_truthiness

print('All truthiness tests passed.')