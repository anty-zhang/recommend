# -*- coding: utf-8 -*-

import numpy as np
import faiss
import pickle
import joblib

path = '../data/'
d = 64
nlist = 2
k = 1
user_embedding_list = []
with open(path + "user_embedding.csv", "r") as f:
    index = 0 
    for line in f.readlines():
        user_embedding_list.append(line.strip())
# print(user_embedding_list[:1])
## user_embedding = np.loadtxt(open(path + "user_embedding.csv", "r"), delimiter=",", skiprows=0)
## # print(user_embedding[:1,:])
## i = 0
## # np.set_printoptions(suppress=True)
## for factor in user_embedding:
##     factor_list = factor.tolist()
##     # str_list = [str("{:e}".format(i)) for i in factor_list]
##     str_list = [str(i) for i in factor_list]
##     user_embedding_list.append(",".join(str_list))

print("size=", len(user_embedding_list))
print("top user embedding 2=", user_embedding_list[:2])
u_label_model_dict = joblib.load(path + 'u_label_model_dict.csv')
u_label_unionid_list = list(u_label_model_dict.keys())
print("unionid size=", len(u_label_unionid_list))
print("top unionid 2", u_label_unionid_list[:2])

unionid_embedding_list = zip(u_label_unionid_list, user_embedding_list)

with open(path + "user_embedding_text_1.csv", "w") as f:
    for unionid,embedding in unionid_embedding_list:
        f.write("fmue_"+unionid+"\001"+embedding+"\n")
