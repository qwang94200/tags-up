from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np



def score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def _metamodelApply(X_train,y_train):
    _nround=5
    names = [ "Decision Tree", "Neural Net", "Naive Bayes" ]
    classifiers=[ DecisionTreeClassifier(),
                  MLPClassifier(alpha=1),
                  GaussianNB(),
                ]
    score=np.zeros(len(classifiers))
    
    for train_index, test_index in KFold(n_splits=_nround).split(X_train):
        X_train2, X_test2 = X_train[train_index,:], X_train[test_index,:]
        y_train2, y_test2 = y_train[train_index], y_train[test_index]
        
        for i in range(len(classifiers)):
            clf=classifiers[i]
            clf.fit(X_train2, y_train2)#fitting Random forest
            y_pred=clf.predict(X_test2)
            score[i]+=score_function(y_test2, y_pred)
            
    score/=_nround
    print("ROC AUC SCORE :", score)
    i=np.argmax(score)
    print(names[i], score[i])
    return score[i]

def _estimate_scores(X_train, y_train):
    _nround=5
    _score=0
    clf =  DecisionTreeClassifier()
    
    for train_index, test_index in KFold(n_splits=_nround).split(X_train):
        X_train2, X_test2 = X_train[train_index,:], X_train[test_index,:]
        y_train2, y_test2 = y_train[train_index], y_train[test_index]
        clf.fit(X_train2, y_train2)#fitting Random forest
        y_pred=clf.predict(X_test2)
        _score +=score_function(y_test2, y_pred)
    return _score/_nround
