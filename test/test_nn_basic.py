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

kb.build_kg_model(cuda=False, embedding_size=30)
# Ideally use bs=1 to overfit on this small dataset
# bs=2 at least checks that it works with > 1 bs
kb.train_kg_model(steps=8001, batch_size=1)

# # # # # # # # # # # # # # # # # # # # # # # #
# Link prediction test
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

print('All basic NN tests passed.')