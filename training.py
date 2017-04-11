import pickle
import numpy as np


# Load preprocessed data
savedfile = 'resources/preprocessed_train.sav'
file = open(savedfile,'r')
dataset = pickle.load(file)

X = dataset['X']
Y = dataset['Y']



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
# model.fit(X, Y)


# from sklearn.ensemble import AdaBoostClassifier

# model = AdaBoostClassifier()
# model.fit(X,Y)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X,Y)


# # Naive Bayes Classifier
# from sklearn.naive_bayes import GaussianNB

# model = GaussianNB()
# model.fit(X, Y)


# Save data
scorefile = 'resources/model.sav'

pickle.dump(model, open(scorefile, 'wb'))