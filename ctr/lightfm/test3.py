# -*- coding: utf-8 -*-
import lightfm
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import csr_matrix
import numpy as np

ids = list([1, 2, 3])
print(ids)
ids = np.array(ids).astype('int')
print(ids.shape)


