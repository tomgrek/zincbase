"""Demo operations on the `countries` dataset

TODO: It would be interesting to put lat, lng as attributes
on the entities

"""

from zincbase import KB

kb = KB()

kb.from_csv('./assets/countries_s1_train.csv', delimiter='\t')

print(list(kb.query('locatedin(X, northern_europe)')))
# prints [{'X': 'norway'}, {'X': 'iceland'}, {'X': 'faroe_islands'}, ...]

print(list(kb.query('neighbor(austria, X)')))
# prints [{'X': 'italy'}, {'X': 'czechia'}, {'X': 'slovenia'}, ...]

kb.build_kg_model(cuda=True, embedding_size=100)

kb.train_kg_model(steps=1000, batch_size=512) # takes < 1 minute

print(kb.estimate_triple_prob('mali', 'locatedin', 'africa'))
# prints a number close to 1

print(kb.get_most_likely('singapore', 'locatedin', '?', k=2))
# prints [{'prob': 0.9672, 'triple': ('singapore', 'locatedin', 'south_eastern_asia')}, ...]

print(kb.get_most_likely('austria', 'neighbor', '?', k=8))
# prints [{'prob': 0.9749, 'triple': ('austria', 'neighbor', 'liechtenstein')} ...]

kb.fit_knn()

print(kb.get_nearest_neighbors('uganda', k=4))
# prints [{'distance': 0.0, 'entity': 'uganda'}, {'distance': 2.5374, 'entity': 'ethiopia'} ...]

print(kb.get_nearest_neighbors('united_kingdom', k=4))
# prints [{'distance': 0.0, 'entity': 'united_kingdom'}, {'distance': 2.7378, 'entity': 'finland'} ...]

kb.fit_knn(entities=set([x['X'] for x in kb.query('neighbor(X, Y)')]))

print(kb.get_nearest_neighbors('jordan', k=10))
# prints [{'distance': 0.0, 'entity': 'jordan'}, {'distance': 2.7171, 'entity': 'lebanon'} ...]

kb.create_multi_classifier('locatedin')

print(kb.multi_classify('australia', 'locatedin'))
# prints 'australia_and_new_zealand'