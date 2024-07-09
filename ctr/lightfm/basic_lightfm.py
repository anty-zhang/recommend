# -*- coding: utf-8 -*-
import lightfm
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from lightfm.data import Dataset

path = '../data/'

u_i_df = pd.read_csv(path + 'u_i.csv')
print(u_i_df)

u_i_row_ind = u_i_df['userid'].tolist()
print(u_i_row_ind)
u_i_col_ind = u_i_df['itemid'].tolist()
print(u_i_col_ind)
u_i_data = u_i_df['label'].tolist()
print(u_i_data)
u_i_csr = csr_matrix((u_i_data, (u_i_row_ind, u_i_col_ind)))
print(u_i_csr)


model = LightFM()
model.fit(u_i_csr)

print(model.get_user_representations())
#获得user_biases和user_features
print(model.get_item_representations())
#同上
print(model.predict_rank(u_i_csr))
#(test_interactions, train_interactions=None, item_features=None, user_features=None, num_threads=1, check_intersections=True)
#test_interactions:要预测的矩阵
#获得data1中每个user对每个item的预测得分,返回一个[users,items]大小的压缩矩阵,toarray()可以展示
print(model.predict(1, [1, 3]))
#(user_ids, item_ids, item_features=None, user_features=None, num_threads=1)
#对单个用户对某个商品(列表形式传入一个或多个商品index)的得分预测
print(model.user_embedding_gradients)
#得到一个shape为[users,no_components]大小的矩阵，代表的是()
print(model.item_embedding_gradients)
#同上,shape is [items,no_components]


from lightfm.evaluation import auc_score
print(auc_score(model, u_i_csr).mean())

