from __future__ import division

# %matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle

## DATA IMPORT 
df = pd.read_csv('datasets/ctrainset.csv', sep = ',')
df_score = pd.read_csv('datasets/cscoreset.csv', sep = ',')


df.dropna() 

print df.shape
print df.describe()
Data = df.values
Score = df_score.values

# Clean data from entries without output 
# [NOT NECESSARY]
# Data = Data[np.isfinite(Data[:,0])]

#N = number of input samples
N = 30000
Y = Data[:N,0]
X = Data[:N,1:]
Y_score = Data[:N,0]
X_score = Data[:N,1:]

## PREPROCESSING

# # Undersampling training_set
# zero_indices = np.where(Y == 0)[0]
# ones_indices = np.where(Y == 1)[0]
# random_indices = np.random.choice(zero_indices, 2, replace=False)
# healthy_sample = X[random_indices]
# sample_size = sum(Y == 1)  # Equivalent to len(data[data.Healthy == 0])
# random_indices = np.random.choice(zero_indices, sample_size, replace=False)
# print 'sample size: '
# print sample_size
# undersampling_indexes =  np.concatenate((random_indices,ones_indices))
# X = X[undersampling_indexes,:]
# Y = Y[undersampling_indexes]

from sklearn import preprocessing

# Add rows for Nan values
def create_nanmask(col):
	if np.isnan(col).any():
	    return np.isnan(col)

nanmask = np.apply_along_axis( create_nanmask, axis=1, arr=X )
print X.shape
print nanmask.shape
X = np.hstack((X, nanmask))
print X.shape
nanmask = np.apply_along_axis( create_nanmask, axis=1, arr=X_score )
X_score = np.hstack((X_score, nanmask))


# Deleting least complete features
nan_floor = np.floor(len(X[:,0])*0.5)
ft_nan = np.array([np.count_nonzero(~np.isnan(X[:,i])) for i in range(len(X[0]))])
print ft_nan.shape
ft_nanmask = np.where(ft_nan < nan_floor)
X =  np.delete(X, ft_nanmask, axis=1)
X_score =  np.delete(X_score, ft_nanmask, axis=1)


# Fill empty (NaN) cells with mean
imputer = preprocessing.Imputer().fit(X)
X = imputer.transform(X)
X_score = imputer.transform(X_score)


# L2 Normalization
scaler = preprocessing.Normalizer().fit(X)
X = scaler.transform(X)
X_score = scaler.transform(X_score)


# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold

print X.shape
var_selector = VarianceThreshold()
var_selector.fit(X)
X = var_selector.transform(X)
X_score = var_selector.transform(X_score)
print X.shape


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

univ_selector = SelectKBest(f_classif, k=10)
univ_selector.fit(X, Y)
X = univ_selector.transform(X)
X_score = univ_selector.transform(X_score)
print X.shape



# Save data
trainfile = 'resources/preprocessed_train.sav'
scorefile = 'resources/preprocessed_score.sav'
train_data = {}
train_data['X'] = X
train_data['Y'] = Y

score_data = {}
score_data['X'] = X_score
score_data['Y'] = Y_score

pickle.dump(train_data, open(trainfile, 'wb'))
pickle.dump(score_data, open(scorefile, 'wb'))