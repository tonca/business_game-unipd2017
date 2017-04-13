# %matplotlib inline
import matplotlib.pyplot as plt
import pickle

# DATA VISUALISATION

from sklearn import decomposition

savedfile = 'resources/preprocessed_train.sav'
file = open(savedfile,'r')
dataset = pickle.load(file)

X = dataset['X']
Y = dataset['Y']

print X[0,0:5]


pca = decomposition.PCA(n_components=1)

X_pca = pca.fit_transform(X)

plt.scatter(Y, X[:,0], c=plt.cm.RdBu_r(0), s=80)

plt.show()

