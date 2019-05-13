"""Test negative examples using Countries.

The main idea here is that if we explicitly enter some false facts (signalling
to the KB that they are false, it should make less-wrong predictions
for them, versus just going by its own synthetic negative examples.)

It may have the side effect of pushing UP the probability of other wrong
triples, see e.g. "canada in asia" below.
"""
import context

from zincbase import KB

kb = KB()
kb.seed(555)

kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')

rule_num = kb.store('~locatedin(canada, africa)')

b = list(kb.query('locatedin(canada, X)'))
assert len(b) == 1; assert b[0]['X'] == 'northern_america'
assert kb.delete_rule(rule_num)

kb.build_kg_model(cuda=False, embedding_size=100)

kb.train_kg_model(steps=500, batch_size=512, neg_ratio=0.01)

canada_in_africa_naive = kb.estimate_triple_prob('canada', 'locatedin', 'africa')
canada_in_asia_naive = kb.estimate_triple_prob('canada', 'locatedin', 'asia')

austria_neighbors_spain_naive = kb.estimate_triple_prob('austria', 'neighbor', 'spain')
austria_neighbors_france_naive = kb.estimate_triple_prob('austria', 'neighbor', 'france')

kb = KB()
kb.seed(555)
kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')
kb.store('~locatedin(canada, africa)')
kb.store('~neighbor(austria, spain)')

kb.build_kg_model(cuda=False, embedding_size=100)
kb.train_kg_model(steps=500, batch_size=512, neg_ratio=0.1)

canada_in_africa_explicit = kb.estimate_triple_prob('canada', 'locatedin', 'africa')
canada_in_asia_explicit = kb.estimate_triple_prob('canada', 'locatedin', 'asia')
austria_neighbors_spain_explicit = kb.estimate_triple_prob('austria', 'neighbor', 'spain')
austria_neighbors_france_explicit = kb.estimate_triple_prob('austria', 'neighbor', 'france')

assert canada_in_africa_naive > canada_in_africa_explicit
assert austria_neighbors_spain_naive > austria_neighbors_spain_explicit

print('All negative example tests passed.')