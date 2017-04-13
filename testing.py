import pickle
import numpy as np

# Load model
savedmodel = 'resources/model.sav'
file = open(savedmodel,'r')
model = pickle.load(file)

# Load preprocessed data
savedscore = 'resources/preprocessed_score.sav'
file = open(savedscore,'r')
score_data = pickle.load(file)

Xte = score_data['X']
Yte = score_data['Y']


# TESTING
from sklearn import metrics

#prediction on test data
pred_te = model.predict(Xte)


# compute accuracy as suggested above using metrics.accuracy_score from scikit-learn for test dataset
print "MAE:", metrics.mean_absolute_error(Yte, pred_te)
print "MSE:", metrics.mean_squared_error(Yte, pred_te)
print "r2: ", metrics.r2_score(Yte, pred_te)


import matplotlib.pyplot as plt
from sklearn import decomposition

pca = decomposition.PCA(n_components=1)

X_pca = pca.fit_transform(Xte)
plt.scatter(Yte, X_pca, c=plt.cm.RdBu_r(0), s=80)

plt.show()