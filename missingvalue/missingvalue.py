import numpy as np
import pandas as pd
import pickle

import datetime

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import logging

logging.basicConfig(level=logging.DEBUG)

    
def score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def _estimate_features(features, scores, threshold):
    logging.warning("thresholds %f", threshold)
    out_ifeatures=[]
    for _i, _feature in enumerate(features):
        if scores[_i]>threshold:
            out_ifeatures.append(_i)
    return out_ifeatures

    
def _estimate_scores(X_train, y_train, features):
    _nround=5
    _score=0
    for feature in features:
        X_train[feature]=X_train[feature].fillna(-1)
    X_train=X_train[features].as_matrix()
    clf =  DecisionTreeClassifier()
    
    for train_index, test_index in KFold(n_splits=_nround).split(X_train):
        X_train2, X_test2 = X_train[train_index,:], X_train[test_index,:]
        y_train2, y_test2 = y_train[train_index], y_train[test_index]
        clf.fit(X_train2, y_train2)#fitting Random forest
        y_pred=clf.predict(X_test2)
        _score +=score_function(y_test2, y_pred)
    return _score/_nround
    

def _estimate_classifiers(X_train, y_train, features):
    for feature in features:
        X_train[feature]=X_train[feature].fillna(-1)
    X_train=X_train[features].as_matrix()
    return _estimate_classifiers_matrix(X_train, y_train)
        
        
def _estimate_classifiers_matrix(X_train, y_train):
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
    print("ROC AUC SCORE :")
    print(score)
    i=np.argmax(score)
    print(names[i], score[i])
    return score[i]

def _estimate_missing_value(X_train, y_train, ofilename):
    features= [_feature for _feature in list(X_train.columns.values) if _feature!='Unnamed: 0']
    #_estimate_threshold_missing_value(X_train, y_train, features, ofilename)
    features = pickle.load( open( ofilename, "rb" ) )
    _estimate_variance_values(X_train, y_train, features, ofilename)
    

def _estimate_threshold_missing_value(X_train, y_train, features, ofilename):
    logging.warning('_estimate_threshold')
    missingcounts={feature: X_train[feature].count() for feature in features}
    _missingcountvalues=np.unique(np.array(missingcounts.values()))
    logging.info(_missingcountvalues.size)
    _counts=len(X_train.index)
    _thresholds=xrange(22, 30)
    _scores=np.zeros(len(_thresholds))
    
    _feature_length=len(features)
    for i in range(len(_thresholds)):
        _count =_thresholds[i]*_counts/100
        _countfeatures=[]
        for feature in features:
            if missingcounts[feature]> _count:
                _countfeatures.append(feature)
        if len(_countfeatures)!=_feature_length:
            logging.info('estimate_threshold %d', _thresholds[i])
            _feature_length=len(_countfeatures)
            _scores[i]=_estimate_classifiers(X_train, y_train, _countfeatures)
        
    i=np.argmax(_scores)
    with open(ofilename, 'wb') as fp:
        pickle.dump( np.array([feature for feature in features if missingcounts[feature]>20*_count/100]), fp)
    return
    
        

def _estimate_threshold(X_train, y_train, df_thresholds):
    _max_threshold=np.max(df_thresholds)
    _min_threshold=np.min(df_thresholds)
    if isinstance(_max_threshold, float) or isinstance(_min_threshold, float):
        funcInt=np.vectorize(pyfunc=lambda x: int(100*x))
        df_thresholds=funcInt(df_thresholds)
        _max_threshold=int(100*_max_threshold)
        _min_threshold=int(100*_min_threshold)
        
    thresholds=xrange(_min_threshold, _max_threshold, (_max_threshold-_min_threshold)/10)
    scores=np.zeros(len(thresholds))
    _nb_features=len(df_thresholds)
    
    i_features=np.array([])
    for _i_threshold, _threshold in enumerate(thresholds):
        _ifeatures=np.array([_i for _i, _feature in enumerate(df_thresholds) if df_thresholds[_i]> _threshold])
        if len(_ifeatures) == _nb_features:
            continue
        else:
            _nb_features=_ifeatures
            _i_scores = _estimate_classifiers_matrix(X_train[:, _ifeatures], y_train)
            if _i_scores>np.max(scores):
                i_features=_ifeatures
            scores[_i_threshold]=_i_scores
                
    return
        
        
        
        
    
    pass
    
def _estimate_variance_values(X_train, y_train, features, ofilename):
    _max_scores=_estimate_scores(X_train, y_train, features)
    
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X_train= imputer.fit_transform(X_train[features])
    std_scale = preprocessing.StandardScaler().fit(X_train)
    df_std = std_scale.transform(X_train)
    
    minmax_scale = preprocessing.MinMaxScaler().fit(X_train)
    df_minmax = minmax_scale.transform(X_train)
    
    df_std_thresholds=[]
    df_minmax_thresholds=[]
    for _id, feature in enumerate(features):
        df_std_thresholds.append(df_std[:, _id].std())
        df_minmax_thresholds.append(df_minmax[:, _id].std())
    
    _score_std, _ifeature_std = _estimate_threshold(X_train, y_train, np.array(df_std_thresholds))
    _score_minmax, _ifeature_minmax = _estimate_threshold(X_train, y_train, np.array(df_minmax_thresholds))

    

    
    logging.info("standardscales")
    _thresholds=np.sort(np.unique(np.array([int(v*100) for v in df_std_thresholds])))[:-1]
    scores=[]
    if _thresholds.size>0:
        for _threshold in _thresholds:
            _out_ifeatures = _estimate_features(features, df_std_thresholds, _threshold*0.01)
            scores.append( _estimate_classifiers_matrix(X_train[:,_out_ifeatures], y_train))
        scores=np.array(scores)
        if scores>_estimate_classifiers_matrix(X_train, y_train):
            _i=np.argmax(scores)
            with open(ofilename, 'wb') as fp:
                pickle.dump(features[_estimate_features(features, df_std_thresholds, _thresholds[_i]*0.01)], fp)
            
    return
            
    if len(set(df_minmax_thresholds)) > 1:
        logging.info("minmaxscales")
        _currentfeatureCount=len(features)
        _thresholds=np.sort(np.unique(np.array([int(v*100) for v in df_minmax_thresholds])))[:-1]
        scores=[]
        for _threshold in _thresholds:
            _features = _estimate_features(features, df_minmax_thresholds, _threshold*0.01)
            scores.append( _estimate_classifiers(X_train, y_train, _features))
        scores=np.array(scores)
        if scores>_estimate_scores(X_train, y_train, features):
            _i=np.argmax(scores)
            features = _estimate_features(features, df_std_thresholds, _thresholds[_i]*0.01)
            with open(ofilename, 'wb') as fp:
                pickle.dump(features, fp)
        
        
            
        
            
        
    if len(set(df_minmax_thresholds)) > 0:
        logging.info("minmax")
        for _threshold in set(df_minmax_thresholds)-set(max(df_minmax_thresholds)):
            pass
            
        
        
        
    
    pass
        
    
    
    
    
    
