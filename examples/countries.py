"""Demo operations on the `countries` dataset

TODO: It would be interesting to put lat, lng as attributes
on the entities

"""

from zincbase import KB

kb = KB()

kb.from_csv('../assets/countries.csv')

print(list(kb.query('locatedin(X, caribbean')))

print(list(kb.query('neighbor(austria, X)')))

kb.build_kg_model(cuda=True, embedding_size=100)

kb.train_kg_model(steps=5001, batch_size=512)

print(kb.estimate_triple_prob('mali', 'locatedin', 'africa'))

print(kb.get_most_likely('singapore', 'locatedin', '?', k=5))

print(kb.get_most_likely('austria', 'located', '?', k=8))

kb.fit_knn()

print(kb.get_nearest_neighbors('uganda', k=4))

print(kb.get_nearest_neighbors('sri_lanka', k=4))

kb.fit_knn(entities=[x['X'] for x in kb.query('capital_of(X, Y)')])

print(kb.get_nearest_neighbors('paris', k=20))

kb.create_multi_classifier('capital_of')

print(kb.multi_classify('oslo', 'capital_of'))