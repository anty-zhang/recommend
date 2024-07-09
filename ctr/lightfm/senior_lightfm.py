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


u_f_df = pd.read_csv(path + 'u_f.csv')
print(u_f_df)

user_features_row_ind = u_f_df['userid'].tolist()
print(user_features_row_ind)
user_features_col_ind = [0] * len(user_features_row_ind)
print(user_features_col_ind)
user_features_data = u_f_df['u_f'].tolist()
print(user_features_data)
user_features = csr_matrix((user_features_data, (user_features_row_ind, user_features_col_ind)))
print(user_features)


i_f_df = pd.read_csv(path + 'i_f.csv')
print(i_f_df)
item_features_row_ind = i_f_df['itemid'].tolist()
print(item_features_row_ind)
item_features_col_ind = [0] * len(item_features_row_ind)
print(item_features_col_ind)
item_features_data = i_f_df['i_f'].tolist()
print(item_features_data)
item_features = csr_matrix((item_features_data, (item_features_row_ind, item_features_col_ind)))
print(item_features)

model = LightFM()
model.fit(u_i_csr, user_features=user_features, item_features=item_features)

print(model.get_user_representations(features=user_features))
print(model.get_item_representations(features=item_features))
print(model.predict_rank(u_i_csr, user_features=user_features, item_features=item_features))
print(model.predict(1, [1, 2], user_features=user_features, item_features=item_features))
print(model.user_embedding_gradients)
print(model.item_embedding_gradients)


from lightfm.evaluation import auc_score
print(auc_score(model, u_i_csr, user_features=user_features, item_features=item_features).mean())

