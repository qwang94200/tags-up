from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC,NuSVC
import numpy as np
import pandas as pd

def score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

print "LOADING DATA"
train = "training_inputs.csv"
trainLAB = "challenge_fichier_de_sortie_dentrainement_predire_les_clients_qui_ont_realise_des_economies_denergie.csv"
test="testing_inputs.csv"

"""
X_train = pd.read_csv(train, header = 0,sep=';').as_matrix()[:,1:]
y_train = pd.read_csv(trainLAB, header = 0,sep=';').as_matrix()[:,1]
X_test = pd.read_csv(test, header = 0,sep=';').as_matrix()[:,1:]
"""
X_train = pd.read_csv(train, sep=';')

x=X_train.iloc[:, 63:63]
for feature in x.columns.values:
    x[feature] = x[feature].fillna(-1)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(x)
print(pca.explained_variance_ratio_)
exit(0)



x=X_train.iloc[:, 56:61]
for feature in x.columns.values:
    x[feature] = x[feature].fillna(-1)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(x)
print(pca.explained_variance_ratio_)
exit(0)

y_train = pd.read_csv(trainLAB, sep=';')
X_test = pd.read_csv(test, sep=';')
"""
lens=[]
for i in range(108):
    x=X_train.iloc[:,i+1:i+2]
    lens.append(np.unique(x))
    
fd=pd.DataFrame(data=np.array(lens))
print(fd.index)
fd['columns']=X_train.columns.values[1:]
print(len(X_train.columns.values[1:]))
fd.to_csv(train+'_basic', sep=';')
"""






valides=[0,3,4,5,6,7,8,9,10,11, 12, 13,15, 16,17,18,19,20,21,22, 23,24,]
column_date=[25, 26, 27]
column_chiffre=[28, 29, 30, 31, 33, 36, 37, 38 ]
column_string=[33,34,35]
column_bool=[39]

column_date_nan=[25, 26, 27]
column_float_nan=[40,41, 45,46, 47, 52, 54]
column_entier_nan=[49,50, 51]
column_string_nan=[42, 43, 44, 60, 61]
column_bool_nan=[39, 48]
"""
##reduce dimension
exit(0)
(49,50)



print "CONVERTING NON-NUMERICAL FEATURES TO NUMERIC"
def isNaN(num):
    return num != num
def isanumber(a):
    bool_a = True
    try:
        a=float(a)
        if isNaN(a):
            bool_a = False
    except:
        bool_a = False
    return bool_a

for i in range(len(X_train[0,:])):
    le = preprocessing.LabelEncoder()
    x=np.concatenate((map(str,X_train[:,i]),map(str,X_test[:,i])))
    for a in x:
        if not(isanumber(a)):
            le.fit(x)
            X_train[:,i] = le.transform(map(str,X_train[:,i]))
            X_test[:,i] = le.transform(map(str,X_test[:,i]))
            break

print "FITTING RANDOM FOREST WITH VARIOUS PARAMETERS DOING K-FOLD CROSS VALIDATION TO AVOID OVERFITTING"

classifiers=[
#RandomForestClassifier(n_jobs=-1,min_samples_split=10),
#RandomForestClassifier(n_jobs=-1,max_features=None,min_samples_split=10),
GaussianNB(),
DecisionTreeClassifier(min_samples_split=40),
RandomForestClassifier(n_jobs=-1,n_estimators=10,min_samples_split=10)
#,oob_score=True)
#RandomForestClassifier(n_jobs=-1,n_estimators=30,max_features="log2",min_samples_split=10),
#,class_weight="balanced")
#RandomForestClassifier(n_jobs=-1,max_features=0.9,min_samples_split=10),
#RandomForestClassifier(n_jobs=-1,max_features=0.7,min_samples_split=10),
#RandomForestClassifier(n_jobs=-1,max_features=0.5,min_samples_split=5)
]

score=np.zeros(len(classifiers))

NROUND=5#number of cross validation sets
for train_index, test_index in KFold(X_train.shape[0], n_folds=NROUND):
    X_train2, X_test2 = X_train[train_index,:], X_train[test_index, :]
    y_train2, y_test2 = y_train[train_index], y_train[test_index]
    for i in range(len(classifiers)):
        clf=classifiers[i]
        clf.fit(X_train2, y_train2)#fitting Random forest
        y_pred=clf.predict(X_test2)
        score[i]+=score_function(y_test2, y_pred)

score/=NROUND
print "ROC AUC SCORE :",score
i=np.argmax(score)
clf=classifiers[i]

print("FITTING CLASSIFIERS ON THE WHOLE TRAINING SET")
clf.fit(X_train, y_train)

print "PREDICTING THE LABELS OF THE TEST"
y_test=clf.predict(X_test)

print "GENERATING OUTPUT FILE TO SUBMIT"

f=open("output.csv","w")
print>>f,"ID;TARGET"
for i in range(len(y_test)):
    print>>f,str(85529+i)+';'+str(y_test[i])
f.close()
"""
