import types

import context
from zincbase import KB

kb = KB()
b = kb.store('c(x)'); assert b == 0
b = kb.query('c(X)'); assert isinstance(b, types.GeneratorType); b = list(b); 
assert len(b) == 1; assert b[0]['X'] == 'x'; assert kb.node(b[0]['X']) == {}
b = kb.store('c(y)'); assert b == 1
b = kb.delete_rule(1); assert b; assert not list(kb.query('c(y)'))
b = kb.store('c(y)'); assert b == 1
b = kb.query('c(X)'); b = list(b); assert len(b) == 2; assert b[0]['X'] in ('x', 'y');
assert b[1]['X'] in ('x', 'y'); assert b[0]['X'] != b[1]['X']; assert kb.node(b[0]['X']) == {}
assert kb.node(b[1]['X']) == {}
b = kb.store('loves(tom, shamala)'); assert 'tom' in kb.G; assert 'shamala' in kb.G
assert kb.G['tom']['shamala'][0]['pred'] == 'loves'
b = kb.query('loves(X, Y)'); b = list(b); assert len(b) == 1; assert b[0]['X'] == 'tom'; assert b[0]['Y'] == 'shamala'
b = kb.query('loves( X,Y)'); b = list(b); assert len(b) == 1; assert b[0]['X'] == 'tom'; assert b[0]['Y'] == 'shamala'
b = kb.query('loves(X, foo)'); b = list(b); assert not b
b = kb.query('loves(tom, shamala)'); b = list(b); assert b
b = kb.query('loves(tom, X)'); b = list(b); assert b; assert len(b) == 1; assert b[0]['X'] == 'shamala'
kb.store('loves(kitty, shamala)')
b = kb.query('loves(Who, shamala)'); b = list(b); assert b; assert len(b) == 2;
assert b[0]['Who'] in ('tom', 'kitty'); assert b[1]['Who'] in ('tom', 'kitty'); assert b[0]['Who'] != b[1]['Who']
kb.store('love_rel(X, Y) :- loves(X, Y)')
b = kb.query('love_rel(X, Y)'); b = list(b); assert len(b) == 2; assert b[0]['Y'] == b[1]['Y'] == 'shamala'
assert b[0]['X'] in ('tom', 'kitty'); assert b[1]['X'] in ('tom', 'kitty'); assert b[0]['X'] != b[1]['X']
kb.store('endless(X):-endless(X)')
b = kb.query('endless(zig)'); b = list(b); assert not b
kb.store('eats(kitty, tuna)')
b = kb.query('eats(K, What_Food)'); b = list(b); assert b; assert len(b) == 1; assert b[0]['What_Food'] == 'tuna'
b = kb.store('is_cat(C) :- eats(C, tuna), love_rel(Y, C)')
b = kb.query('is_cat(C)'); b = list(b); assert not b
kb.store('loves(tom, kitty)')
b = kb.query('is_cat(C)'); b = list(b); assert b; assert len(b) == 1; assert b[0]['C'] == 'kitty'
b = kb.query('is_cat(kitty)'); b = list(b); assert b
b = kb.query('is_cat(tom)'); b = list(b); assert not b
b = kb.query('eats(kitty, _)'); b = next(b); assert b; assert isinstance(b, bool)
b = kb.query('eats(_, tuna)'); b = next(b); assert b; assert isinstance(b, bool)
b = kb.query('eats(_, X)'); b = list(b); assert len(b) == 1; assert len(b[0]) == 1; assert b[0]['X'] == 'tuna'
b = kb.query('eats(Y, X)'); b = list(b); assert len(b) == 1; assert b[0]['X'] == 'tuna'; assert b[0]['Y'] == 'kitty'

kb.store('employed_by(tom, primer)')
kb.store('works_as(tom, engineer)')
kb.store('engineers_at(X, Y) :- employed_by(X, Y), works_as(X, engineer)')
b = kb.query('employed_by(X, Y)'); b = list(b); assert b[0]['X'] == 'tom'; assert b[0]['Y'] == 'primer'
b = kb.query('works_as(tom, X)'); b = list(b); assert b; assert b[0]['X'] == 'engineer'
b = kb.query('engineers_at(ZZ, primer)'); b = list(b); assert len(b) == 1; assert b[0]['ZZ'] == 'tom'

kb.store('employed_by(oleg,primer)')
b = kb.query('engineers_at(ZZZ, primer)'); b = list(b); assert len(b) == 1; assert b[0]['ZZZ'] == 'tom'
kb.store('works_as(oleg, engineer)')
b = kb.query('engineers_at(Xss, primer)'); b = list(b); assert len(b) == 2; assert b[0]['Xss'] in ['oleg', 'tom']; assert b[1]['Xss'] in ['oleg', 'tom']; assert b[0]['Xss'] != b[1]['Xss']

kb.store('engineers_at_primer(You) :- engineers_at(You, primer)')
b = kb.query('engineers_at_primer(Engineers)'); b = list(b); assert b[0]['Engineers'] in ['oleg', 'tom'];
assert b[1]['Engineers'] in ['oleg', 'tom']; assert b[1]['Engineers'] != b[0]['Engineers']

kb.store('X_is_engineer_at_primer_doing_neural_networks(X) :- engineers_at_primer(X), works_on(X, neural_networks)')
b = kb.query('X_is_engineer_at_primer_doing_neural_networks(tom)'); b = list(b); assert not b

kb.store('works_on(tom, neural_networks)')
b = kb.query('X_is_engineer_at_primer_doing_neural_networks(tom)'); b = next(b); assert b
b = kb.query('X_is_engineer_at_primer_doing_neural_networks(_)'); b = next(b); assert b; assert isinstance(b, bool)
b = kb.query('X_is_engineer_at_primer_doing_neural_networks(Brainiac)'); b = list(b); assert b; assert len(b) == 1; assert b[0]['Brainiac'] == 'tom'
kb.store('works_on(oleg, neural_networks)')
b = kb.query('X_is_engineer_at_primer_doing_neural_networks(Brainiac)'); b = list(b); assert b; assert len(b) == 2; assert b[0]['Brainiac'] in ['oleg', 'tom']
assert b[1]['Brainiac'] in ['oleg', 'tom']; assert b[1]['Brainiac'] != b[0]['Brainiac'];

kb = KB()
kb.store('gave_mono_to(john, jane)')
kb.store('gave_mono_to(jane, phil)')
b = kb.bfs('john', 'phil'); b = list(b); assert len(b) == 1 
assert len(b[0]) == 2; assert b[0][0] == ('gave_mono_to', 'jane'); assert b[0][1] == ('gave_mono_to', 'phil')

b = kb.bfs('phil', 'john'); b = list(b); assert not b

b = kb.bfs('phil', 'john', reverse=True); b = list(b); assert len(b) == 1
assert b[0][0] == ('gave_mono_to', 'jane'); assert b[0][1] == ('gave_mono_to', 'john')

kb.store('gave_mono_to(jane, janice)')
kb.store('gave_mono_to(janice, phil)')
b = kb.bfs('john', 'phil'); b = list(b); assert len(b) == 2
assert len(b[0]) == 2; assert b[0][0] == ('gave_mono_to', 'jane')
assert b[0][1] == ('gave_mono_to', 'phil'); assert b[1][0] == ('gave_mono_to', 'jane');
assert b[1][1] == ('gave_mono_to', 'janice'); assert b[1][2] == ('gave_mono_to', 'phil')

print('All main KB tests passed.')