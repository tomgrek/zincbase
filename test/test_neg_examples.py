"""Test negative examples using Countries.
"""

from zincbase import KB

kb = KB()
kb.seed(555)

kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')

rule_num = kb.store('~locatedin(canada, africa)')

b = list(kb.query('locatedin(canada, X)'))
assert len(b) == 1; assert b[0]['X'] == 'northern_america'
assert kb.delete_rule(rule_num)

#kb.store('~locatedin(canada, africa)')

kb.build_kg_model(cuda=True, embedding_size=100)

kb.train_kg_model(steps=500, batch_size=512, neg_ratio=0.01)

print(kb.estimate_triple_prob('canada', 'locatedin', 'africa'))
print(kb.estimate_triple_prob('canada', 'locatedin', 'asia'))