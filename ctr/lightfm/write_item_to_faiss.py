# -*- coding: utf-8 -*-

import numpy as np
import faiss
import pickle
import joblib
import time
from contextlib import contextmanager

path = '../data/'
d = 64
nlist = 2
k = 1

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

item_embedding = np.loadtxt(open(path + "item_embedding.csv", "rb"), delimiter=",", skiprows=0)
item_embedding = np.array(item_embedding, dtype=np.float32)
print(item_embedding.shape)
print(type(item_embedding))

i_label_model_dict = joblib.load(path + 'i_label_model_dict.csv')
#print(list(u_label_model_dict.keys()))
#ids = np.array(list(u_label_model_dict.keys()), dtype=np.int)
#print(ids.shape)
#print(type(ids))
# quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexFlatL2(d)
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# index = faiss.index_factory(d, "IVF100,PQ8")
# assert not index.is_trained
# index = faiss.IndexIDMap(index_b)
with timer("index train..."):
    index.train(item_embedding)
# assert index.is_trained
    index.add(item_embedding)
# index.add_with_ids(user_embedding, ids)
# D, I = index.search(item_embedding, k)
# print(I[-5:])
# index.nprobe = 10
# D, I = index.search(item_embedding, k)
# print(I[-5:])

faiss.write_index(index, path + "index")


ids_item_embedding = dict(zip(i_label_model_dict.keys(), item_embedding))
#print("-------------------------------------------------")
#print(ids_user_embedding)
pickle.dump(ids_item_embedding, open(path + 'ids_vectors.p', 'wb'))

ids_to_ids = dict(zip(i_label_model_dict.values(), i_label_model_dict.keys()))
pickle.dump(ids_to_ids, open(path + 'ids_to_ids.p', 'wb'))
