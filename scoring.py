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

Xte = score_data['X_score']

# SCORING
#prediction on test data
pred_te = model.predict(Xte)

score_path = 'resources/score.csv'
np.savetxt(score_path, pred_te, delimiter=",")
