import pickle
import numpy as np
import sys

# Load preprocessed data
savedfile = 'resources/preprocessed_train.sav'
file = open(savedfile,'r')
dataset = pickle.load(file)

X = dataset['X']
Y = dataset['Y']


mod_i = 0
if len(sys.argv) > 1 :
	mod_i = sys.argv[1]


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


models = {
    '0' : LogisticRegressionCV(solver='newton-cg', class_weight='balanced'),
    '1' : AdaBoostClassifier(),
    '2' : GaussianNB(),
    '3' : RandomForestClassifier(),
    '4' : BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
    '5' : ExtraTreesClassifier(),
    '6' : GradientBoostingClassifier()
}

#fit the model on training data
models[mod_i].fit(X, Y)


# Save data
scorefile = 'resources/model.sav'

pickle.dump(models[mod_i], open(scorefile, 'wb'))