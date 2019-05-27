import context

from zincbase import KB

kb = KB()

kb.store('append([ ], List, List)')
kb.store('append([Head | Tail], List, [Head | Result]) :- append(Tail, List, Result)')
b = kb.query('append([a, b], [c, d], X)'); b = list(b); assert len(b) == 1; assert b[0]['X'] == '[a,b,c,d]'
b = kb.query('append([a, b], X, [a, b, c, d])'); b = list(b); assert len(b) == 1; assert b[0]['X'] == '[c,d]'
b = kb.query('append(X, Y, [a, b, c, d])'); b = list(b); assert len(b) == 5; assert b[0]['X'] == '[]'; assert b[0]['Y'] == '[a,b,c,d]'
assert b[1]['X'] == '[a]'; assert b[1]['Y'] == '[b,c,d]'; assert b[2]['X'] == '[a,b]'; assert b[2]['Y'] == '[c,d]'
assert b[3]['X'] == '[a,b,c]'; assert b[3]['Y'] == '[d]'; assert b[4]['X'] == '[a,b,c,d]'; assert b[4]['Y'] == '[]'

print('All list tests passed.')