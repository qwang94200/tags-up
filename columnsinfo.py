import numpy as np
import _columnsinfo
from missingvalue import _classifiers_missing_value, _preprocessing_feature_dtypes

PATH='~/Downloads/challengeDataEDF/'


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
_preprocessing_feature_dtypes(train, trainLAB)
