# %matplotlib inline
import matplotlib.pyplot as plt
import pickle

# DATA VISUALISATION

from sklearn import decomposition

savedfile = 'resources/preprocessed.sav'
file = open(savedfile,'r')
dataset = pickle.load(file)

X = dataset['X']
Y = dataset['Y']


pca = decomposition.PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.scatter(X_pca[Y==0, 0], X_pca[Y==0, 1], c=plt.cm.RdBu_r(0), s=80)
plt.scatter(X_pca[Y==1, 0], X_pca[Y==1, 1], c=plt.cm.RdBu_r(256), s=80)

plt.show()