#!/usr/bin/env python 

# %%
import xgboost as xgb
from utils import plot_metrics
import utils_data
# %% 
fname = "large/train/{}"
train_in, train_label = utils_data.read(fname.format("1000"))
test_in, test_label = utils_data.read(fname.format("1001"))

# %%
print(train_in.shape, train_label.shape)
# %%
dtrain = xgb.DMatrix(train_in, label=train_label)
dtest  = xgb.DMatrix(test_in, label=test_label)
# %%
param = {'max_depth': 20, 'eta': 0.1, "objective": "binary:logistic"}
num_round = 2000
# %%
bst = xgb.train(param, dtrain, num_round)
# %%
preds = bst.predict(dtest)
# %%
preds
# %%
plot_metrics(preds, test_label, outname='test.png')