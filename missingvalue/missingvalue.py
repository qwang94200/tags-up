import numpy as np
import pandas as pd

import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from dateutil.parser import parse as dateparse
def is_date(string):
    try:
        dateparse(string)
        return True
    except ValueError:
        return False

def score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def _preprocessing_date_features(X_train, datefeatures, minDate='2000-01-01'):
    import time
    dateOutfeatures=[]
    for feature in datefeatures:
        X_train[feature]=X_train[feature].fillna(minDate)
        X_train[feature]=pd.to_datetime(X_train[feature])
        timefeatures=['timestamp', 'current_year', 'month_of_year', 'week_of_year', 'weekday', 'day_of_year', 'day_of_month']
        timefunctions=[
            lambda x: time.mktime(x.timetuple()),
            lambda x: x.strftime("%Y"),
            lambda x: x.strftime("%m"),
            lambda x: x.weekday(),
            lambda x: x.strftime("%W"),
            lambda x: x.strftime("%j"),
            lambda x: x.strftime("%d"),
        ]
        for _i, _timefeature in enumerate(timefeatures):
            _featurename=feature+'_'+_timefeature
            dateOutfeatures.append(_featurename)
            X_train[_featurename]=X_train[feature].apply(timefunctions[_i])
    return X_train[dateOutfeatures]

def _preprocessing_bool_features(X_train, boolfeatures):
    func_bool=lambda x: 1 if x==True else 0
    
    dateOutfeatures=[]
    for feature in boolfeatures:
        _featurename=feature+'_int'
        dateOutfeatures.append(_featurename)
        X_train[_featurename]=X_train[feature].dropna().apply(func_bool)
    return X_train[dateOutfeatures]

def _preprocessing_string_features(X_train, stringfeatures):
    IDfeature='ID'
    def _label_strings(stringList):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        return le.fit_transform(stringList)
    
    dateOutfeatures=[]
    for feature in stringfeatures:
        _featurename=feature+'_int'
        dateOutfeatures.append(_featurename)
        X_train[_featurename] = _label_strings(X_train[feature])
    
    for feature in stringfeatures:
        
    return X_train[dateOutfeatures]


def _preprocessing_feature_dtypes(train, trainLAB):
    import itertools
    #X_train = pd.read_csv(train, header = 0,sep=';').as_matrix()[:,1:]
    #y_train = pd.read_csv(trainLAB, header = 0,sep=';').as_matrix()[:,1]
    X_train = pd.read_csv(train, sep=';')
    y_train = pd.read_csv(trainLAB, sep=';')
    x = X_train.dtypes
    xdict={}
    for d, v in x.iteritems():
        xdict[d]=v
    fd=pd.DataFrame(xdict.items() ,columns=['name', 'dtype'])
    
    fd_types=fd.groupby(by='dtype')
    """
    print(fd_types.size())
    print(len(X_train['Q2'].index))
    print(len(X_train['Q2'].dropna().index))
    exit(0)
    """
    datefeatures=[]
    boolfeatures=[]
    stringfeatures=[]
    for feature in X_train.columns.values:
        if xdict[feature]=='object':
            x = X_train[feature].dropna().tolist()
            
            if type(x[0])==bool:
                boolfeatures.append(feature)
                continue
                
            if is_date(x[0]):
                datefeatures.append(feature)
                continue
            
            stringfeatures.append(feature)
            
    #_preprocessing_date_features(X_train,datefeatures).to_csv(train+'_datefeatures',sep=';')
    #_preprocessing_bool_features(X_train,boolfeatures).to_csv(train+'_boolfeatures',sep=';')
    _preprocessing_string_features(X_train,stringfeatures).to_csv(train+'_stringfeatures',sep=';')
    #print(pd.concat([a,b,c], axis=1).head(4))
    

                
                
        
            
    #print(X_train[datefeatures].head(10))
        
        
        
    
    
    
    
    
    
    
    
    
    #print(y_train.dtypes)
    
    
    

def _classifiers_missing_value(X_train, y_train):
    _nround=100
    names = [ "Decision Tree", "Neural Net", "Naive Bayes" ]
    classifiers=[ DecisionTreeClassifier(),
                  MLPClassifier(alpha=1),
                  GaussianNB(),]
    score=np.zeros(len(classifiers))
    
    X_train = pd.read_csv(train, header = 0,sep=';').as_matrix()[:,1:]
    y_train = pd.read_csv(trainLAB, header = 0,sep=';').as_matrix()[:,1]
    
    for train_index, test_index in KFold(n_splits=_nround):
        X_train2, X_test2 = X_train[train_index,:], X_train[test_index, :]
        y_train2, y_test2 = y_train[train_index], y_train[test_index]
        for i in range(len(classifiers)):
            clf=classifiers[i]
            clf.fit(X_train2, y_train2)#fitting Random forest
            y_pred=clf.predict(X_test2)
            score[i]+=score_function(y_test2, y_pred)
            
    score/=_nround
    fd=pd.DataFrame(data=[names, score], columns=['names', 'scores'])
    print("ROC AUC SCORE :")
    print(fd)
    i=np.argmax(score)
    clf=classifiers[i]
    
    from sklearn.externals import joblib
    joblib.dump(clf, train+'missingvalue.pkl')
    return clf

def _estimate_missing_value(fd):
    pass
    
    
