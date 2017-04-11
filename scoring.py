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


# SCORING
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score


#prediction on test data
Yhat_test_LR = model.predict(Xte)


# compute accuracy as suggested above using metrics.accuracy_score from scikit-learn for test dataset
print "Test Accuracy:", metrics.accuracy_score(Yte, Yhat_test_LR)


pred_te = model.predict(Xte)


fpr, tpr, thresholds = roc_curve(Yte, pred_te)
print fpr
print tpr
print "The AUC score is "
print auc(fpr, tpr)

