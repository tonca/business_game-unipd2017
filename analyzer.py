from __future__ import division

# %matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

## DATA IMPORT 
df = pd.read_csv('datasets/ctrainset.csv', sep = ',')

df.dropna() 

print df.shape
print df.describe()
Data = df.values


# Clean data from entries without output 
# [NOT NECESSARY]
# Data = Data[np.isfinite(Data[:,0])]

#N = number of input samples
N = 30000
Y = Data[:N,0]
X = Data[:N,1:]


## PREPROCESSING

# Undersampling
zero_indices = np.where(Y == 0)[0]
ones_indices = np.where(Y == 1)[0]
random_indices = np.random.choice(zero_indices, 2, replace=False)
healthy_sample = X[random_indices]


sample_size = sum(Y == 1)  # Equivalent to len(data[data.Healthy == 0])
random_indices = np.random.choice(zero_indices, sample_size, replace=False)
print sample_size

undersampling_indexes =  np.concatenate((random_indices,ones_indices))
print undersampling_indexes
X = X[undersampling_indexes,:]
Y = Y[undersampling_indexes]
print X
print Y

from sklearn.cross_validation import train_test_split 	
from sklearn import preprocessing

# Deleting all incomplete rows
# [ALL ROWS ARE INCOMPLETE SOMEHOW]
# X_nonan = X[~np.isnan(X).any(axis=1)]
# print "X_nonan : " 
# print X_nonan.shape

# Deleting the least complete features
nan_floor = np.floor(len(X[:,0])*0.5)
ft_nan = np.array([np.count_nonzero(~np.isnan(X[:,i])) for i in range(len(X[0]))])
print ft_nan.shape
X =  np.delete(X, np.where(ft_nan < nan_floor), axis=1)
print X.shape

# Fill empty (NaN) cells
imputer = preprocessing.Imputer().fit(X)
X = imputer.transform(X)


# Split data into train (50 samples) and test data (the rest)
Ntr = np.floor(N*0.80)
print Ntr 	
Nte = N - Ntr


Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=Nte/N)


scaler = preprocessing.Normalizer().fit(Xtr)
Xtr = scaler.transform(Xtr)
Xte = scaler.transform(Xte)


# Comparing feature variance
ft_var = np.var(Xtr, axis=0)
print ft_var.shape
ft_sort = np.argsort(ft_var)
print ft_var[ft_sort]

var_floor = ft_var[ft_sort[-50]]
print var_floor
Xtr =  np.delete(Xtr, np.where(ft_var < var_floor), axis=1)
Xte =  np.delete(Xte, np.where(ft_var < var_floor), axis=1)
print Xtr.shape


# # DATA VISUALISATION
# from sklearn import decomposition

# pca = decomposition.PCA(n_components=2)

# X_pca = pca.fit_transform(X)

# plt.scatter(X_pca[Y==0, 0], X_pca[Y==0, 1], c=plt.cm.RdBu_r(0), s=80)
# plt.scatter(X_pca[Y==1, 0], X_pca[Y==1, 1], c=plt.cm.RdBu_r(256), s=80)

# plt.show()

## TRAINING

def choose_C_gamma(X, Y, n_folds=5, kernel='linear', C_range=[1e-2,1,1e2], gamma_range=[1], degree=3, coef0=0.0):

    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC

    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(Y, n_folds=n_folds, random_state=None)
    grid = GridSearchCV(SVC(kernel=kernel, degree=degree, coef0=coef0), param_grid=param_grid, cv=cv, verbose=10, n_jobs=3)
    grid.fit(X, Y)

    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    plt.figure(figsize=(8, 6))
    if len(gamma_range) > 1:
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
    else:
        plt.plot(scores)
        plt.xlabel('C')
        plt.ylabel('Validation accuracy')
        plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
    plt.show()
    
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    
    if len(gamma_range) > 1:
        return grid.best_params_['C'], grid.best_params_['gamma']
    else:
        return grid.best_params_['C']


# C_range_rbf_CV = [10**i for i in np.arange(-2,8,2)]
# gamma_range_rbf_CV = [10**i for i in np.arange(-9,0,2)]
# best_C_rbf, best_gamma_rbf = choose_C_gamma(Xtr, Ytr, kernel='rbf', C_range=C_range_rbf_CV, gamma_range=gamma_range_rbf_CV)


# from sklearn.svm import SVC

# model = SVC(kernel="rbf", C=best_C_rbf, gamma=best_gamma_rbf, degree=3, coef0=0.0)
# model.fit(Xtr, Ytr)


# from sklearn import linear_model

# # define a logistic regression model with very high C parameter -> low impact from regularization
# model = linear_model.LogisticRegressionCV(solver='newton-cg', class_weight='balanced')

# #fit the model on training data
# model.fit(Xtr, Ytr)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(Xtr, Ytr)


# SCORING

from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

#prediction on training data
Yhat_tr_LR = model.predict(Xtr)
#prediction on test data
Yhat_test_LR = model.predict(Xte)


# compute accuracy as suggested above using metrics.accuracy_score from scikit-learn for training dataset
print "Training Accuracy:", metrics.accuracy_score(Ytr,Yhat_tr_LR)
# compute accuracy as suggested above using metrics.accuracy_score from scikit-learn for test dataset
print "Test Accuracy:", metrics.accuracy_score(Yte, Yhat_test_LR)

pred_te = model.predict(Xte)

Yte = np.reshape(Yte, (Yte.shape[0],1))
pred_te = np.reshape(pred_te, (pred_te.shape[0],1))
print Yte.shape
print pred_te.shape

print Yte.shape
fpr, tpr, thresholds = roc_curve(Yte, pred_te)

#print "AUC score:" + 
print "The AUC score is "+auc(fpr, tpr)
