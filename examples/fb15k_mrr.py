from zincbase import KB
from zincbase.utils.data_science import calc_mrr

kb = KB()

kb.from_csv('./assets/fb15k_train_mod.txt', delimiter='\t')

kb.build_kg_model(cuda=True, embedding_size=1000, gamma=24)

kb.train_kg_model(steps=20000, batch_size=2048, neg_to_pos=128)

mrr = calc_mrr(kb, './assets/fb15k_test_mod.txt', delimiter='\t') # add optional `size` kwarg since eval is currently slow.

print(mrr) # should be ~0.797 to match the paper.
