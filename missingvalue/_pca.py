import numpy as np
import pandas as pd
import pickle
import random

from sklearn import preprocessing
import _metamodels
import standaring

import collections
import logging

logging.basicConfig(level=logging.DEBUG)

def _remove_pairnodes(values, threshold):
    logging.info(threshold)
    _removenodes=[]
    _select_removevalues=[value for value in values if value[2]>=threshold]
    
    _select_nodes=np.array([value[0] for value in _select_removevalues])
    _select_nodes=np.append(_select_nodes, np.array([value[1] for value in _select_removevalues]))
    _nodecount =collections.Counter(_select_nodes)
    
    
    while len(_nodecount)>0 and len(_select_removevalues)>0:
        for _node in sorted(_nodecount, key=_nodecount.get, reverse=True):
            _removenodes.append(_node)
            _select_removevalues = [value for value in _select_removevalues if value[1]!=_node and value[0]!=_node]
            if len(_select_removevalues)==0:
                break
            _select_nodes=np.array([value[0] for value in _select_removevalues])
            _select_nodes=np.append(_select_nodes, np.array([value[1] for value in _select_removevalues]))
            _nodecount =collections.Counter(_select_nodes)
    return values, _removenodes
            
            
        
        

def _pca_apply(X, y, features, ofname):
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X= imputer.fit_transform(X[features])
    X= standaring._standardizing(X)
    pass


def _choose_features(X,y, features, values):
    scores=[]
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X= imputer.fit_transform(X[features])
    for i in xrange(47, 51):
        logging.info('start choosen features %d', i)
        select_iifeatures=np.array([value[0] for value in values])[:i]
        scores.append((i,_metamodels._metamodelApply(X[:,select_iifeatures],y)))
    print("scores", scores)
    
def _select_feature_seuil(features, values, threshold):
    select_iifeatures=np.array([value[0] for value in values])[:threshold]
    return features[select_iifeatures]
    
    
def _choose_features_correlation(X,y, features, values):
    scores=[]
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X= imputer.fit_transform(X[features])
    thresholds=np.linspace(0.8, 1.0, num=30)[::-1]
    _il, _jl = X.shape
    print(X.shape)
    for threshold in thresholds:
        values, _removenodes = _remove_pairnodes(values, threshold)
        _ifeatures=np.full(_jl, True, dtype=bool)
        for _node in _removenodes:
             _ifeatures[_node]=False
        _iifeatures = np.reshape(np.where(_ifeatures), -1)
        scores.append((threshold,_metamodels._metamodelApply(X[:,_iifeatures],y)))
        ##scores.append((threshold, _metamodels._estimate_scores(X[:,_iifeatures],y)))
    print("scores", scores)
    
    
def _select_feature_correlation_seuil(features, values, threshold):
    values, _removenodes = _remove_pairnodes(values, threshold)
    _ifeatures=np.full(len(features), True, dtype=bool)
    for _node in _removenodes:
        _ifeatures[_node]=False
    _iifeatures = np.reshape(np.where(_ifeatures), -1)
    return features[_iifeatures]
        
        
def _fill_null(X):
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    return imputer.fit_transform(X)
        
        
        
        
        

    
        
