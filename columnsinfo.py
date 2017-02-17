import numpy as np
import os.path
import _columnsinfo
from missingvalue import _estimate_classifiers, _estimate_missing_value
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

#columnfile=_columnsinfo._read_columns(train)
columnfile=train+'_basic'
#_columnsinfo._read_missing_columns(columnfile)
#_classifiers(PATH+train, PATH+trainLAB)
ifilenames = _preprocessing_feature_dtypes(train)
ofilenames = _preprocessing_feature_dtypes(trainLAB)

#logging.warning(ifilenames)
#logging.info(ofilenames)

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

###estimiate classifiers###
#_estimate_classifiers(X_train, X_trainLAB)
ofilename=train+'_features_estimates'
_estimate_missing_value(X_train, X_trainLAB, ofilename)
###

        
    
    




