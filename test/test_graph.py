import context
from xinkbase import KB

kb = KB()
kb.store('a(b,c)')
kb.attr('b', {'is_letter': 1.0})
assert kb.node('b') == {'is_letter': 1.0}
kb.edge_attr('b', 'a', 'c', {'both_alpha': 1.0})
assert kb.edge('b', 'a', 'c') == {'both_alpha': 1.0}
assert kb.to_triples() == [('b', 'a', 'c')]
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {})]
kb.attr('c', {'is_letter': 0.9})
triples = kb.to_triples(data=True)
assert triples == [('b', 'a', 'c', {'is_letter': 1.0}, {'both_alpha': 1.0}, {'is_letter': 0.9})]

print('All graph tests passed.')