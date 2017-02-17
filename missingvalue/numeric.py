import numpy as np
import pandas as pd

import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import logging
logging.basicConfig(level=logging.DEBUG)

from dateutil.parser import parse as dateparse

def is_date(string):
    try:
        dateparse(string)
        return True
    except ValueError:
        return False

def _preprocessing_date_features(X_train, datefeatures, minDate='2000-01-01'):
    import time
    dateOutfeatures=[]
    for feature in datefeatures:
        #X_train[feature]=X_train[feature].fillna(minDate)
        X_train[feature]=pd.to_datetime(X_train[feature], errors='coerce',)
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
            X_train[_featurename]=X_train[feature].dropna().apply(timefunctions[_i])
        
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
            
    def _label_strings(stringList):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        return le.fit_transform(stringList)
    
    stringOutfeatures=[]
    for feature in stringfeatures:
        _featurename=feature+'_int'
        stringOutfeatures.append(_featurename)
        X_train[_featurename] = _label_strings(X_train[feature])
        _features=[feature, _featurename]
        X_train[_featurename] = X_train[_features].apply(lambda row: row[1] if row[0] else None, axis=1)

        
    return X_train[stringOutfeatures]


def _preprocessing_feature_dtypes(train):
    import itertools
    #X_train = pd.read_csv(train, header = 0,sep=';').as_matrix()[:,1:]
    #y_train = pd.read_csv(trainLAB, header = 0,sep=';').as_matrix()[:,1]
    X_train = pd.read_csv(train, sep=';')
    x = X_train.dtypes
    xdict={}
    for d, v in x.iteritems():
        xdict[d]=v
    """
    fd=pd.DataFrame(xdict.items() ,columns=['name', 'dtype'])
    fd_types=fd.groupby(by='dtype')
    print(fd_types.size())
    print(len(X_train['Q2'].index))
    print(len(X_train['Q2'].dropna().index))
    exit(0)
    """
    datefeatures=[]
    boolfeatures=[]
    stringfeatures=[]
    numericfeatures=[]
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
        else:
            numericfeatures.append(feature)
            
            
    logging.warning("transform features")
    ofilenames =[train+'_features_numeric', train+'_features_bool', train+'_features_string', train+'_features_date']
    
    """
    if len(numericfeatures)>0:
        X_train[numericfeatures].to_csv(train+'_features_numeric', sep=';')
    if len(boolfeatures)>0:
        _preprocessing_bool_features(X_train,boolfeatures).to_csv(train+'_features_bool',sep=';')
    if len(stringfeatures)>0:
        _preprocessing_string_features(X_train, stringfeatures).to_csv(train+'_features_string',sep=';')
    if len(datefeatures)>0:
        _preprocessing_date_features(X_train,datefeatures).to_csv(train+'_features_date',sep=';')
    
    """
    
    return ofilenames
    
    
    """
    #fd_datefeatures = _preprocessing_date_features(X_train,datefeatures)
    #fd_datefeatures.to_csv(train+'_datefeatures',sep=';')
    
    #fd_datefeatures = pd.read_csv(train+'_datefeatures',sep=';')
    l1=list(fd_datefeatures.columns.values)
    l2=datefeatures
    print(pd.concat([X_train, fd_datefeatures], axis=1)[l1+l2].head(5))
    """
    
    """
    _preprocessing_bool_features(X_train,boolfeatures).to_csv(train+'_boolfeatures',sep=';')
    fd_bool_features=pd.read_csv(train+'_boolfeatures',sep=';')
    l1 = list(fd_bool_features.columns.values)
    l2= boolfeatures
    print(pd.concat([X_train, fd_bool_features], axis=1)[l1+l2].head(5))
    """
    """
    _preprocessing_string_features(X_train, stringfeatures).to_csv(train+'_stringfeatures',sep=';')
    fd_string_features=pd.read_csv(train+'_stringfeatures',sep=';')
    l1 = list(fd_string_features.columns.values)
    l2= stringfeatures
    print(pd.concat([X_train, fd_string_features], axis=1)[l1+l2].head(5))
    """

                
        
        
    
    
    
    
    
    
