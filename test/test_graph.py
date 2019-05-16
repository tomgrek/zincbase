import context
from zincbase import KB

kb = KB()
kb.store('a(b,c)')
kb.attr('b', {'is_letter': 1.0})
assert kb.node('b') == {'is_letter': 1.0}
kb.edge_attr('b', 'a', 'c', {'both_alpha': 1.0})
assert kb.edge('b', 'a', 'c') == {'both_alpha': 1.0}
assert kb.to_triples() == [('b', 'a', 'c')]
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {}, False)]
kb.attr('c', {'is_letter': 0.9})
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {'is_letter': 0.9}, False)]
neg_rule_idx = kb.store('~a(b,c)')
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {'is_letter': 0.9}, True)]
kb.delete_rule(neg_rule_idx)
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {'is_letter': 0.9}, False)]
kb.edge_attr('b', 'a', 'c', {'truthiness':-1})
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0, 'truthiness': -1}, {'is_letter': 0.9}, True)]
kb.delete_edge_attr('b', 'a', 'c', ['truthiness'])
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {'is_letter': 0.9}, False)]
print('All graph tests passed.')