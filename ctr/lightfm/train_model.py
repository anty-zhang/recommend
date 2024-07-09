# -*- coding: utf-8 -*-
import os
import lightfm
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

home_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

u_label_model = LabelEncoder()
i_label_model = LabelEncoder()
u_f_name_model = LabelEncoder()
i_f_name_model = LabelEncoder()
u_f_value_model = LabelEncoder()
i_f_value_model = LabelEncoder()


def get_u_i():
    u_i_df = pd.read_csv(home_path + 'u_i.csv', sep='\001', names=['userid', 'itemid', 'label'])
    u_i_row_ind = u_i_df['userid'].tolist()
    # u_label_model.fit(u_i_row_ind)
    # joblib.dump(u_label_model, path + 'u_label_model.csv')
    # u_label_model_dict = dict(zip(u_label_model.classes_, [j for j in range(len(u_label_model.classes_))]))
    # joblib.dump(u_label_model_dict, path + 'u_label_model_dict.csv')
    u_i_row_ind = u_label_model.transform(u_i_row_ind)
    # print(u_i_row_ind)

    u_i_col_ind = u_i_df['itemid'].tolist()
    # i_label_model.fit(u_i_col_ind)
    # joblib.dump(i_label_model, path + 'i_label_model.csv')
    # i_label_model_dict = dict(zip(i_label_model.classes_, [j for j in range(len(i_label_model.classes_))]))
    # joblib.dump(i_label_model_dict, path + 'i_label_model_dict.csv')
    u_i_col_ind = i_label_model.transform(u_i_col_ind)
    # print(u_i_col_ind)

    u_i_data = u_i_df['label'].tolist()
    # print(u_i_data)

    u_i_csr = csr_matrix((u_i_data, (u_i_row_ind, u_i_col_ind)))
    # print(u_i_csr)

    return u_i_csr


def get_u_f():
    u_f_df = pd.read_csv(home_path + 'u_f.csv', sep='\001', names=['userid', 'u_f_name', 'u_f_value'])
    user_features_row_ind = u_f_df['userid'].tolist()
    # user_features_row_ind = u_label_model.transform(user_features_row_ind)
    u_label_model.fit(user_features_row_ind)
    # joblib.dump(u_label_model, home_path + 'u_label_model.csv')
    u_label_model_dict = dict(zip(u_label_model.classes_, [j for j in range(len(u_label_model.classes_))]))
    # joblib.dump(u_label_model_dict, home_path + 'u_label_model_dict.csv')
    user_features_row_ind = u_label_model.transform(user_features_row_ind)
    # print(user_features_row_ind)

    user_features_col_ind = u_f_df['u_f_name'].tolist()
    u_f_name_model.fit(user_features_col_ind)
    user_features_col_ind = u_f_name_model.transform(user_features_col_ind)
    # print(user_features_col_ind)

    user_features_data = u_f_df['u_f_value'].tolist()
    u_f_value_model.fit(user_features_data)
    user_features_data = u_f_value_model.transform(user_features_data)
    # print(user_features_data)

    user_features = csr_matrix((user_features_data, (user_features_row_ind, user_features_col_ind)))
    # print(user_features)

    return user_features


def get_i_f():
    i_f_df = pd.read_csv(home_path + 'i_f.csv', sep='\001', names=['itemid', 'i_f_name', 'i_f_value'])
    item_features_row_ind = i_f_df['itemid'].tolist()
    # item_features_row_ind = i_label_model.transform(item_features_row_ind)
    i_label_model.fit(item_features_row_ind)
    # joblib.dump(i_label_model, home_path + 'i_label_model.csv')
    # i_label_model_dict = dict(zip(i_label_model.classes_, [j for j in range(len(i_label_model.classes_))]))
    # joblib.dump(i_label_model_dict, home_path + 'i_label_model_dict.csv')
    item_features_row_ind = i_label_model.transform(item_features_row_ind)
    # print(item_features_row_ind)

    item_features_col_ind = i_f_df['i_f_name'].tolist()
    i_f_name_model.fit(item_features_col_ind)
    item_features_col_ind = i_f_name_model.transform(item_features_col_ind)
    # print(item_features_col_ind)

    item_features_data = i_f_df['i_f_value'].tolist()
    i_f_value_model.fit(item_features_data)
    item_features_data = i_f_value_model.transform(item_features_data) 
    # print(item_features_data)

    item_features = csr_matrix((item_features_data, (item_features_row_ind, item_features_col_ind)))
    # print(item_features)

    return item_features


def get_data():
    # u_i_csr = get_u_i()
    user_features = get_u_f()
    item_features = get_i_f()
    u_i_csr = get_u_i()
    
    return u_i_csr, user_features, item_features


def train():
    u_i_csr, user_features, item_features = get_data()
    print("start train")
    model = LightFM(no_components=32)
    model.fit(u_i_csr, user_features=user_features, item_features=item_features, epochs=3, verbose=True, num_threads=1)
    # print(model.get_user_representations(features=user_features))
    # print(model.get_item_representations(features=item_features))

    save_embeddings(model.get_user_representations(features=user_features)[1], 'user_embedding')
    save_embeddings(model.get_item_representations(features=item_features)[1], 'item_embedding')

    # print(model.predict_rank(u_i_csr, user_features=user_features, item_features=item_features))
    # print(model.predict(1, [2, 3], user_features=user_features, item_features=item_features))

    # print(model.user_embedding_gradients)
    # print(model.item_embedding_gradients)

    # print(model.user_embeddings)
    # print(model.item_embeddings)

    print(auc_score(model, u_i_csr, user_features=user_features, item_features=item_features).mean())


def save_embeddings(matrix, name):
    np.savetxt(home_path + name + ".csv", matrix, delimiter=',')
    # matrix = np.loadtxt(open(path + "matrix.csv", "rb"), delimiter=",", skiprows=0)
    # print(matrix)


if __name__ == "__main__":
    train()
