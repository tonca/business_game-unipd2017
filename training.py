import pickle
import numpy as np
import sys


# Load preprocessed data
savedfile = 'resources/preprocessed_train.sav'
file = open(savedfile,'r')
dataset = pickle.load(file)

X = dataset['X']
Y = dataset['Y']


mod_i = '1'
if len(sys.argv) > 1 :
	mod_i = sys.argv[1]


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor



models = {
    '1' : LinearRegression(),
    '2' : RidgeCV(alphas=(0.1, 0.5, 50.0)),
    '3' : RandomForestRegressor(n_estimators=100, min_samples_leaf=40),
    '4' : LassoCV(),
    '5' : ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50)
}

#fit the model on training data
models[mod_i].fit(X, Y)

# Save data
scorefile = 'resources/model.sav'

pickle.dump(models[mod_i], open(scorefile, 'wb'))