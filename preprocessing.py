from __future__ import division

# %matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle

## DATA IMPORT 
df = pd.read_table('datasets/TRAIN.txt')
df_score = pd.read_table('datasets/TEST0.txt')

# df = np.array(df)
# df.dropna() 

print df.shape
print df.describe()
Data = df.values
Score = df_score.values

# Clean data from entries without output 
# [NOT NECESSARY]
# Data = Data[np.isfinite(Data[:,0])]

#N = number of input samples
N = 30000
Y = Data[:N,1]
X = Data[:N,2:]
Xsc = Score[:,2:]
print X.shape
print Xsc.shape
print X[0:50,:] 


# PREPROCESSING

from sklearn import preprocessing


# Encoding Labels
def label_encoding(entry_set):

	labeler = preprocessing.LabelEncoder()

	# Encoding labels
	for i in range(len(entry_set[0,:])):
		if isinstance(entry_set[0,i], (str, unicode)):
			ft = entry_set[:,i]
			labeler.fit(ft)
			entry_set[:,i] = labeler.transform(ft)

	print "Still string?"
	for i in range(len(entry_set[0,:])):
		if isinstance(entry_set[0,i], (str, unicode)):
			print entry_set[0,i]

	return entry_set


def binarized_encoding(entry_set):
	labeler = preprocessing.MultiLabelBinarizer()

	new_ft = np.array([])
	substituted = np.array([])	

	# Encoding labels
	for i in range(len(entry_set[0,:])):
		if isinstance(entry_set[0,i], (str, unicode)):
			ft = entry_set[:,i]
			print ft
			print ft.shape
			print new_ft.shape
			ft = ft.astype(str)
			labeler.fit(ft)
			if len(new_ft)==0:
				new_ft = labeler.transform(ft)
			else:
				new_ft = np.hstack((labeler.transform(ft), new_ft))
			substituted = np.append(substituted, i)

	entry_set = np.delete(entry_set, substituted, axis=1)
	entry_set = np.hstack((entry_set, new_ft))

	print entry_set.shape

	print "Still string?"
	for i in range(len(entry_set[0,:])):
		if isinstance(entry_set[0,i], (str, unicode)):
			print entry_set[0,i]

	return entry_set


X = label_encoding(X)
Xsc = label_encoding(Xsc)

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

# def add_nanmask(X_set):
# 	nanmask = np.apply_along_axis( create_nanmask, axis=1, arr=X_set )
# 	return np.hstack((X_set, nanmask))

# Xtr = add_nanmask(Xtr)
# Xte = add_nanmask(Xte)
# Xsc = add_nanmask(Xsc)


# # Deleting least complete features
# nan_floor = np.floor(len(Xtr[:,0])*0.5)
# ft_nan = np.array([np.count_nonzero(~np.isnan(Xtr[:,i])) for i in range(len(Xtr[0]))])
# print ft_nan.shape
# ft_nanmask = np.where(ft_nan < nan_floor)

# Xtr = np.delete(Xtr, ft_nanmask, axis=1)
# Xte = np.delete(Xte, ft_nanmask, axis=1)
# Xsc = np.delete(Xsc, ft_nanmask, axis=1)


# Fill empty (NaN) cells with mean
imputer = preprocessing.Imputer().fit(Xtr)
Xtr = imputer.transform(Xtr)
Xte = imputer.transform(Xte)
Xsc = imputer.transform(Xsc)


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