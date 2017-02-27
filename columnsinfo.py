import csv
import numpy as np
import os.path
from sklearn import decomposition


from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import _columnsinfo
#from missingvalue import _estimate_classifiers, _estimate_missing_value
from missingvalue import standaring, _pca, utils
from _columnsinfo import  _read_columns
from numeric import _preprocessing_feature_dtypes

import pandas as pd
PATH='/home/qwang/Downloads/challengeDataEDF/'

import logging
logging.basicConfig(level=logging.DEBUG)


print("LOADING DATA")
train = "training_inputs.csv"
trainLAB = "challenge_fichier_de_sortie_dentrainement_predire_les_clients_qui_ont_realise_des_economies_denergie.csv"
test="testing_inputs.csv"

train=PATH+train
trainLAB= PATH+trainLAB

"""
##1. read file
ifilenames = _preprocessing_feature_dtypes(train)
ofilenames = _preprocessing_feature_dtypes(trainLAB)

_trains=[]
for _filename in ifilenames:
    if (os.path.isfile(_filename)):
        _trains.append(pd.read_csv(_filename, sep=';'))
    
for _filename in ofilenames:
    if os.path.isfile(_filename):
        _trainLAB=pd.read_csv(_filename, sep=';')
        
#logging.info(len(_trains))
X_train=pd.concat(_trains, axis=1)
X_trainLAB=_trainLAB.as_matrix()[:,2:]

X_train.to_csv('train_first.csv', sep=';', index=False)
_columnsinfo._write_numpy('trainlab_first.csv', X_trainLAB)
print('finish')
"""

X_train=pd.read_csv('train_first.csv', sep=';')
X_trainLAB=_columnsinfo._read_numpy('trainlab_first.csv')
    

##2. select features based on normalized
features=X_train.columns.values
features = np.array([feature for feature in features if not feature.startswith('Unnamed')])
ifeatures= np.array([not feature.startswith('Unnamed') for feature in features])
print(features.size)
X_train=X_train[features]

"""
X=standaring._standardizing(X_train)
a,b,c=standaring._covarianceMatrix(X)
values=standaring._sortEngenpairs(b,c)
#_pca._choose_features(X_train, X_trainLAB, features, values)
_select_features=_pca._select_feature_seuil(features,values, 95)

utils._pickle_write(_select_features, 'select_feature_first.csv')


features = utils._pickle_read('select_feature_first.csv')
X_train = X_train[features]
X=standaring._standardizing(X_train)
values=standaring._calculatecorrelation(X)
#_pca._choose_features_correlation(X_train, X_trainLAB, features, values)
_select_features=_pca._select_feature_correlation_seuil(features,values,0.896551)
utils._pickle_write(_select_features, 'select_feature_second.csv')
"""

##3. select features based on missing value
features = utils._pickle_read('select_feature_second.csv')
X_train = X_train[features]
"""
print(len(features))
values=standaring._estimate_missing_value(X_train, features)
print (values)
_select_features=_pca._select_feature_correlation_seuil(features,values,49)
X_train = X_train[features]
"""
from sklearn import linear_model, decomposition, datasets
X=_pca._fill_null(X_train)
##4. select features based on importances and PCA
logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
import matplotlib.pyplot as plt
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')



n_components = [5, 20, 40]
Cs = np.logspace(-4, 4, 3)



estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X, X_trainLAB)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.savefig('res.pdf')



clf = ExtraTreesClassifier()
clf = clf.fit(X, X_trainLAB)
print(clf.feature_importances_)
exit(0)


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)


##5. estimated models


exit(0)






#columnfile=_columnsinfo._read_columns(train)
columnfile=train+'_basic'
#_columnsinfo._read_missing_columns(columnfile)
#_classifiers(PATH+train, PATH+trainLAB)


#logging.warning(ifilenames)
#logging.info(ofilenames)



###estimiate classifiers###
#_estimate_classifiers(X_train, X_trainLAB)
ofilename=train+'_features_estimates'
_estimate_missing_value(X_train, X_trainLAB, ofilename)
###

        
    
    




