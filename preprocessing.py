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
Xsc = Score[:,0:]
print X.shape
print Xsc.shape
print Xsc

# PREPROCESSING

from sklearn import preprocessing

# Add rows for Nan values
def create_nanmask(col):
	if np.isnan(col).any():
	    return np.isnan(col)

from sklearn.cross_validation import train_test_split 	

# Split data into train (50 samples) and test data (the rest)
Ntr = np.floor(len(X)*0.8)
print Ntr 	
Nte = N - Ntr

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=Nte/N)

def add_nanmask(X_set):
	nanmask = np.apply_along_axis( create_nanmask, axis=1, arr=X_set )
	return np.hstack((X_set, nanmask))

Xtr = add_nanmask(Xtr)
Xte = add_nanmask(Xte)
Xsc = add_nanmask(Xsc)


# Deleting least complete features
nan_floor = np.floor(len(Xtr[:,0])*0.5)
ft_nan = np.array([np.count_nonzero(~np.isnan(Xtr[:,i])) for i in range(len(Xtr[0]))])
print ft_nan.shape
ft_nanmask = np.where(ft_nan < nan_floor)

Xtr = np.delete(Xtr, ft_nanmask, axis=1)
Xte = np.delete(Xte, ft_nanmask, axis=1)
Xsc = np.delete(Xsc, ft_nanmask, axis=1)


# Fill empty (NaN) cells with mean
imputer = preprocessing.Imputer().fit(Xtr)
Xtr = imputer.transform(Xtr)
Xte = imputer.transform(Xte)
Xsc = imputer.transform(Xsc)


# L2 Normalization
scaler = preprocessing.Normalizer().fit(Xtr)
Xtr = scaler.transform(Xtr)
Xte = scaler.transform(Xte)
Xsc = scaler.transform(Xsc)


# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold

var_selector = VarianceThreshold()
var_selector.fit(Xtr)
Xtr = var_selector.transform(Xtr)
Xte = var_selector.transform(Xte)
Xsc = var_selector.transform(Xsc)


# PCA regularization
from sklearn import decomposition

# pca = decomposition.PCA(n_components=80)

# pca.fit(Xtr)
# Xtr = pca.transform(Xtr)
# Xte = pca.transform(Xte)
# Xsc = pca.transform(Xsc)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

univ_selector = SelectPercentile(f_classif, percentile=40)
univ_selector.fit(Xtr, Ytr)
Xtr = univ_selector.transform(Xtr)
Xte = univ_selector.transform(Xte)
Xsc = univ_selector.transform(Xsc)
print Xtr.shape

# Undersampling training_set
def undersampling(X_set, Y_set):	
	zero_indices = np.where(Y_set == 0)[0]
	ones_indices = np.where(Y_set == 1)[0]
	random_indices = np.random.choice(zero_indices, 2, replace=False)
	healthy_sample = X_set[random_indices]
	sample_size = sum(Y_set == 1)  # Equivalent to len(data[data.Healthy == 0])
	random_indices = np.random.choice(zero_indices, sample_size, replace=True)
	print 'sample size: '
	print sample_size
	undersampling_indexes =  np.concatenate((random_indices,ones_indices))
	X_set = X_set[undersampling_indexes,:]
	Y_set = Y_set[undersampling_indexes]
	return X_set, Y_set

# Oversampling 
def oversampling(X_set, Y_set):

	zero_indices = np.where(Y_set == 0)[0]
	ones_indices = np.where(Y_set == 1)[0]

	bootstrap_ones = np.random.choice(ones_indices, len(zero_indices)-len(ones_indices))
	oversampling_indices = np.concatenate((zero_indices, ones_indices, bootstrap_ones))
	np.random.shuffle(oversampling_indices)
	X_set = X_set[oversampling_indices, :]
	Y_set = Y_set[oversampling_indices]

	print X_set.shape
	print Y_set.shape


	return X_set, Y_set

# Oversampling or Undersampling
Xtr, Ytr = oversampling(Xtr, Ytr)
# Xtr, Ytr = undersampling(Xtr, Ytr)

# Save data
trainfile = 'resources/preprocessed_train.sav'
scorefile = 'resources/preprocessed_score.sav'
train_data = {}
train_data['X'] = Xtr
train_data['Y'] = Ytr

score_data = {}
score_data['X'] = Xte
score_data['Y'] = Yte
score_data['X_score'] = Xsc

pickle.dump(train_data, open(trainfile, 'wb'))
pickle.dump(score_data, open(scorefile, 'wb'))