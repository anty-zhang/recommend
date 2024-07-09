# -*- coding: utf-8 -*-

import numpy as np
import faiss
import pickle
import joblib

path = '../data/'
d = 64
nlist = 2
k = 1

user_embedding = np.loadtxt(open(path + "user_embedding.csv", "rb"), delimiter=",", skiprows=0)
user_embedding = np.array(user_embedding, dtype=np.float32)
print(user_embedding.shape)
print(type(user_embedding))

u_label_model_dict = joblib.load(path + 'u_label_model_dict.csv')
#print(list(u_label_model_dict.keys()))
#ids = np.array(list(u_label_model_dict.keys()), dtype=np.int)
#print(ids.shape)
#print(type(ids))
# quantizer = faiss.IndexFlatL2(d)
# index = faiss.IndexFlatL2(d)
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# index = faiss.index_factory(d, "IVF100,PQ8")
# assert not index.is_trained
# index = faiss.IndexIDMap(index_b)
# index.train(user_embedding)
# assert index.is_trained

# index.add(user_embedding)
# index.add_with_ids(user_embedding, ids)
# D, I = index.search(user_embedding, k)
# print(I[-5:])
# index.nprobe = 10
# D, I = index.search(user_embedding, k)
# print(I[-5:])

#faiss.write_index(index, path + "index")


ids_user_embedding = dict(zip(u_label_model_dict.keys(), user_embedding))
#print("-------------------------------------------------")
#print(ids_user_embedding)
pickle.dump(ids_user_embedding, open(path + 'ids_user_vactor.csv', 'wb'))

#ids_to_ids = dict(zip(u_label_model_dict.values(), u_label_model_dict.keys()))
#pickle.dump(ids_to_ids, open(path + 'ids_to_ids.p', 'wb'))
