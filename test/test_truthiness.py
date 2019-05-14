"""Test the combination of edge attributes and negative examples."""

import context

from zincbase import KB

kb = KB()
kb.seed(555)

kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')

kb.store('locatedin(canada, africa)')
kb.edge_attr('canada', 'locatedin', 'africa', {'truthiness':-0.1}) # long time ago
#kb.store('~locatedin(canada, africa)')

kb.build_kg_model(cuda=True, embedding_size=100, pred_attributes=['truthiness'])

kb.train_kg_model(steps=500, batch_size=512, neg_ratio=0.01)

canada_in_africa = kb.estimate_triple_prob('canada', 'locatedin', 'africa')
canada_in_asia = kb.estimate_triple_prob('canada', 'locatedin', 'asia')

print(kb.estimate_triple_prob_with_attrs('canada', 'locatedin', 'africa', 'truthiness'))
print(canada_in_africa, canada_in_asia)