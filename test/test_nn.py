import context
from xinkbase import KB

kb = KB()
people = ['john', 'oleg', 'tom', 'vedant', 'akshay', 'todd']
for person in people:
    kb.store('works_at({}, primer)'.format(person))
other_people = ['other1', 'other2', 'other3', 'other4', 'other5', 'other6']
for person in other_people:
    kb.store('works_at({}, zillow)'.format(person))
kb.store('based_in(primer, bay_area)')
kb.store('based_in(zillow, seattle)')

for person in people:
    kb.store('lives_in({}, bay_area)'.format(person))

for person in other_people:
    kb.store('lives_in({}, seattle)'.format(person))

kb.store('knows(tom, john)'); kb.store('knows(tom, oleg)'); kb.store('knows(akshay, john)'); kb.store('knows(tom, todd)'); kb.store('knows(vedant, akshay)')
kb.store('knows(other2, other1)'); kb.store('knows(other6, other5)'); kb.store('knows(other1, other2)'); kb.store('knows(other4, other3)'); kb.store('knows(other3, other2)')
kb.store('knows(tom, other4)')
kb.store('lives_in(shamala, bay_area)'); kb.store('lives_in(mary, seattle)')
kb.store('associated_with(zillow, amazon)'); kb.store('associated_with(primer, google)')
kb.store('associated_with(amazon, zillow)'); kb.store('associated_with(google, primer)')
kb.store('associated_with(amazon, microsoft)'); kb.store('associated_with(google, facebook)')
kb.store('based_in(microsoft, seattle)'); kb.store('based_in(facebook, bay_area)')

kb.attr('tom', {'owns_a_raincoat': 0.0})
kb.attr('todd', {'owns_a_raincoat': 0.0})
kb.attr('oleg', {'owns_a_raincoat': 0.0})
kb.attr('john', {'owns_a_raincoat': 0.0})
kb.attr('akshay', {'owns_a_raincoat': 0.0})
kb.attr('vedant', {'owns_a_raincoat': 0.0})
kb.attr('other1', {'owns_a_raincoat': 1.0})
kb.attr('other2', {'owns_a_raincoat': 1.0})
kb.attr('other3', {'owns_a_raincoat': 1.0})
kb.attr('other4', {'owns_a_raincoat': 1.0})
kb.attr('other5', {'owns_a_raincoat': 1.0})
kb.attr('other6', {'owns_a_raincoat': 1.0})
kb.seed(555)
kb.build_kg_model(cuda=True, embedding_size=50)
kb.train_kg_model(steps=20001, batch_size=1) #batch_size=1, 

x = kb._kg_model.run_embedding(kb.get_embedding('other1')) # should = 1
print('want 1', x)
x = kb._kg_model.run_embedding(kb.get_embedding('other2')) # should = 1
print('want 1', x)
x = kb._kg_model.run_embedding(kb.get_embedding('tom')) # should = 0
print('want 0', x)
x = kb._kg_model.run_embedding(kb.get_embedding('todd')) # should = 0
print('want 0', x)
x = kb._kg_model.run_embedding(kb.get_embedding('shamala')) # should = 0
print('want 0', x)
x = kb._kg_model.run_embedding(kb.get_embedding('mary')) # should = 1
print('want 1', x)

x = kb.estimate_triple_prob('tom', 'lives_in', 'bay_area') #should be high
print('want high', x)
x = kb.estimate_triple_prob('tom', 'lives_in', 'seattle') #should be low
print('want low', x)
x = kb.estimate_triple_prob('shamala', 'lives_in', 'bay_area') #should be high
print('want high', x)
x = kb.estimate_triple_prob('shamala', 'lives_in', 'seattle')
print('want low', x)
x = kb.estimate_triple_prob('other1', 'lives_in', 'seattle')
print('want high', x)
x = kb.estimate_triple_prob('other1', 'lives_in', 'bay_area')
print('want low', x)
x = kb.estimate_triple_prob('other2', 'lives_in', 'seattle')
print('want high', x)
x = kb.estimate_triple_prob('other2', 'lives_in', 'bay_area')
print('want low', x)