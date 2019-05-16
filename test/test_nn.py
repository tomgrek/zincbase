# Neural Network tests

# Possible that tests could flake given stochasticity of NN
# but they are fairly relaxed.

import context
from zincbase import KB

kb = KB()
kb.seed(555)

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

# # # # # # # # # # # # # # # # # # # # # # # #
# Test using node attributes in NN
# # # # # # # # # # # # # # # # # # # # # # # #
kb.attr('tom', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('todd', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('oleg', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('john', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('akshay', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('vedant', {'owns_a_raincoat': 0.0, 'doesnt_own_raincoat': 1.0})
kb.attr('other1', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})
kb.attr('other2', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})
kb.attr('other3', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})
kb.attr('other4', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})
kb.attr('other5', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})
kb.attr('other6', {'owns_a_raincoat': 1.0, 'doesnt_own_raincoat': 0.0})

kb.build_kg_model(cuda=False, embedding_size=30, node_attributes=['owns_a_raincoat', 'doesnt_own_raincoat'],
               attr_loss_to_graph_loss=0.9)
# Ideally use bs=1 to overfit on this small dataset
# bs=2 at least checks that it works with > 1 bs
kb.train_kg_model(steps=12001, batch_size=2, neg_to_pos=4)

# # # # # # # # # # # # # # # # # # # # # # # #
# People from Seattle should be more likely to
# own an umbrella (attribute prediction test)
# # # # # # # # # # # # # # # # # # # # # # # #
x = kb._kg_model.run_embedding(kb.get_embedding('other1'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('other1'), 'doesnt_own_raincoat')
assert round(x) == 1; assert round(y) == 0
x = kb._kg_model.run_embedding(kb.get_embedding('other2'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('other2'), 'doesnt_own_raincoat')
assert round(x) == 1; assert round(y) == 0
x = kb._kg_model.run_embedding(kb.get_embedding('mary'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('mary'), 'doesnt_own_raincoat')
assert x > 1.5 * y # Not a known fact
x = kb._kg_model.run_embedding(kb.get_embedding('tom'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('tom'), 'doesnt_own_raincoat')
assert round(x) == 0; assert round(y) == 1
x = kb._kg_model.run_embedding(kb.get_embedding('todd'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('todd'), 'doesnt_own_raincoat')
assert round(x) == 0; assert round(y) == 1
x = kb._kg_model.run_embedding(kb.get_embedding('shamala'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('shamala'), 'doesnt_own_raincoat')
assert y > 1.5 * x # Not a known fact

# # # # # # # # # # # # # # # # # # # # # # # #
# These relations should still work (link
# prediction test)
# # # # # # # # # # # # # # # # # # # # # # # #
bay_prob = kb.estimate_triple_prob('tom', 'lives_in', 'bay_area')
sea_prob = kb.estimate_triple_prob('tom', 'lives_in', 'seattle')
assert bay_prob > 2 * sea_prob

bay_prob = kb.estimate_triple_prob('shamala', 'lives_in', 'bay_area')
sea_prob = kb.estimate_triple_prob('shamala', 'lives_in', 'seattle')
assert bay_prob > 2 * sea_prob

sea_prob = kb.estimate_triple_prob('other1', 'lives_in', 'seattle')
bay_prob = kb.estimate_triple_prob('other1', 'lives_in', 'bay_area')
assert sea_prob > 2 * bay_prob
sea_prob = kb.estimate_triple_prob('other2', 'lives_in', 'seattle')
bay_prob = kb.estimate_triple_prob('other2', 'lives_in', 'bay_area')
assert sea_prob > 2 * bay_prob

print('First suite of neural network tests passed.')
# # # # # # # # # # # # # # # # # # # # # # # #
# Test predicate attributes
# # # # # # # # # # # # # # # # # # # # # # # #
kb.store('lives_in(tom, seattle)') ## TODO seems weird to have to add this fact then stick 'formerly' on it
# on the next line.
kb.edge_attr('tom', 'lives_in', 'seattle', {'formerly': 1.0})
kb.seed(555)
kb.build_kg_model(cuda=False, embedding_size=50, node_attributes=['owns_a_raincoat', 'doesnt_own_raincoat'],
                pred_attributes=['formerly'], attr_loss_to_graph_loss=0.9, pred_loss_to_graph_loss=5.0)
kb.train_kg_model(steps=8001, batch_size=2, neg_to_pos=4)

# # # # # # # # # # # # # # # # # # # # # # # #
# People from Seattle should be more likely to
# own an umbrella (attribute prediction test)
# # # # # # # # # # # # # # # # # # # # # # # #
x = kb._kg_model.run_embedding(kb.get_embedding('other1'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('other1'), 'doesnt_own_raincoat')
assert round(x) == 1; assert round(y) == 0
x = kb._kg_model.run_embedding(kb.get_embedding('other2'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('other2'), 'doesnt_own_raincoat')
assert round(x) == 1; assert round(y) == 0
x = kb._kg_model.run_embedding(kb.get_embedding('mary'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('mary'), 'doesnt_own_raincoat')
assert x > 1.5 * y # Not a known fact
x = kb._kg_model.run_embedding(kb.get_embedding('tom'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('tom'), 'doesnt_own_raincoat')
assert round(x) == 0; assert round(y) == 1
x = kb._kg_model.run_embedding(kb.get_embedding('todd'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('todd'), 'doesnt_own_raincoat')
assert round(x) == 0; assert round(y) == 1
x = kb._kg_model.run_embedding(kb.get_embedding('shamala'), 'owns_a_raincoat')
y = kb._kg_model.run_embedding(kb.get_embedding('shamala'), 'doesnt_own_raincoat')
assert y > 1.5 * x # Not a known fact

# # # # # # # # # # # # # # # # # # # # # # # #
# These relations should still work (link
# prediction test)
# # # # # # # # # # # # # # # # # # # # # # # #
bay_prob = kb.estimate_triple_prob('tom', 'lives_in', 'bay_area')
sea_prob = kb.estimate_triple_prob('tom', 'lives_in', 'seattle')
assert bay_prob > sea_prob

bay_prob = kb.estimate_triple_prob('shamala', 'lives_in', 'bay_area')
sea_prob = kb.estimate_triple_prob('shamala', 'lives_in', 'seattle')
assert bay_prob > 2 * sea_prob

sea_prob = kb.estimate_triple_prob('other1', 'lives_in', 'seattle')
bay_prob = kb.estimate_triple_prob('other1', 'lives_in', 'bay_area')
assert sea_prob > 2 * bay_prob
sea_prob = kb.estimate_triple_prob('other2', 'lives_in', 'seattle')
bay_prob = kb.estimate_triple_prob('other2', 'lives_in', 'bay_area')
assert sea_prob > 2 * bay_prob

x = kb.estimate_triple_prob_with_attrs('tom', 'lives_in', 'bay_area', 'formerly')
y = kb.estimate_triple_prob_with_attrs('john', 'lives_in', 'bay_area', 'formerly')
z = kb.estimate_triple_prob_with_attrs('mary', 'lives_in', 'bay_area', 'formerly')
assert x > y; assert x > z

x = kb.estimate_triple_prob_with_attrs('tom', 'knows', 'john', 'formerly')
y = kb.estimate_triple_prob('tom', 'knows', 'john')
assert y > x

print('Neural network tests passed.')