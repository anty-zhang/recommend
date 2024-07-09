# -*- coding: utf-8 -*-

import numpy as np
import faiss
import pickle
import joblib
import time
from contextlib import contextmanager

path = '../data/'
d = 256
nlist = 100
k = 5

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))

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
index = faiss.IndexIVFPQ(index, d, nlist, 4, 8)
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# index = faiss.index_factory(d, "IVF100,PQ8")
# assert not index.is_trained
# index = faiss.IndexIDMap(index_b)
with timer("index train..."):
    index.train(item_embedding)
# assert index.is_trained
    index.add(item_embedding)
"""
with timer("index serach..."):
    faiss.omp_set_num_threads(4)
    for i in range(10000):
        D, I = index.search(np.array([[-6.49936000e+07, -2.45711648e+08,  1.46173250e+07,  2.97774320e+07,
       -1.03326800e+08,  5.68221184e+08,  2.81079680e+08, -2.29284896e+08,
       -2.63225280e+08,  2.03918560e+07,  1.12748560e+08, -6.42673200e+07,
       -3.24373888e+08,  3.20941696e+08,  1.09333340e+07, -1.43882432e+08,
       -1.65979480e+07, -1.18655096e+08, -1.07578230e+07, -1.98220528e+08,
       -3.75385880e+07,  3.95239025e+06,  8.25284000e+07, -1.29852144e+08,
       -1.61960400e+07, -1.17552200e+08,  1.91828640e+08,  3.85060192e+08,
        8.25041200e+07, -1.20430720e+08,  3.82383160e+07,  1.10925920e+07,
        2.67949820e+07, -4.65500880e+07, -1.53644870e+07,  4.43892608e+08,
       -3.02943775e+06, -1.08786752e+08,  4.49425254e+09, -4.60233560e+07,
        1.06557800e+08,  2.68729897e+10,  1.03875048e+08, -2.63901648e+08,
        7.36304480e+07,  2.08291136e+08, -3.36122528e+08,  6.79777760e+07,
       -4.51777000e+06, -1.07248806e+09, -1.00856712e+08, -3.39701600e+07,
       -6.29560704e+08, -2.47683120e+07,  9.50208384e+08, -1.05623936e+08,
        5.63434816e+08, -1.48625392e+08,  4.32990208e+08,  2.96867296e+08,
       -2.21747632e+08, -1.99662560e+08, -1.75744420e+07, -1.07636520e+08]], dtype=np.float32), k)   
"""
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
