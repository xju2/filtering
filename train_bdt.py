#!/usr/bin/env python 

# %%
import numpy as np
import xgboost as xgb
from utils import plot_metrics
import time

def read(fname):
    array = np.load(fname)
    return array['arr_0'], array['arr_1']
# %%

# %% 
fname = "/global/homes/x/xju/work/exatrkx/data/train/{}"
train_in, train_label = read(fname.format("1003.npz"))
test_in, test_label = read(fname.format("1019.npz"))

# %%
print(train_in.shape, train_label.shape)
# %%
dtrain = xgb.DMatrix(train_in, label=train_label)
dtest  = xgb.DMatrix(test_in, label=test_label)
# %%
param = {
    'max_depth': 20,
    'eta': 0.001,
    "objective": "binary:logistic",
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
}
num_round = 1
# %%
bst = xgb.train(param, dtrain, num_round)
# %%
start_time = time.time()
preds = bst.predict(dtest)
print("total time for predicting one event {:.2f} s".format(time.time()-start_time))

# %%
plot_metrics(preds, test_label, outname='test.png')