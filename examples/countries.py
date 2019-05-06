"""Demo operations on the `countries` dataset

TODO: It would be interesting to put lat, lng as attributes
on the entities

"""

from zincbase import KB

kb = KB()

kb.from_csv('../assets/countries.csv')

print(list(kb.query('in_subregion(X, caribbean')))

print(list(kb.query('borders(austria, X)')))

kb.build_kg_model(cuda=True, embedding_size=40)

kb.train_kg_model(steps=30001, batch_size=32)

print(kb.estimate_triple_prob('oranjestad', 'capital_of', 'aruba'))

print(kb.get_most_likely('ottawa', 'capital_of', '?', k=5))

print(kb.get_most_likely('austria', 'borders', '?', k=8))

kb.fit_knn()

print(kb.get_nearest_neighbors('aruba', k=4))

print(kb.get_nearest_neighbors('kampala', k=4))

kb.fit_knn(entities=[x['X'] for x in kb.query('capital_of(X, Y)')])

print(kb.get_nearest_neighbors('paris', k=20))

kb.create_multi_classifier('capital_of')

print(kb.multi_classify('oslo', 'capital_of'))