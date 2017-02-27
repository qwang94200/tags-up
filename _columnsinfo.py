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


def _read_columns(filename):
    X_train = pd.read_csv(filename, sep=';')
    lens=[]
    missingvalues=[]
    _ncolumns=len(X_train.columns.values)
    for i in range(_ncolumns):
        x=X_train[X_train.columns.values[i]]
        lens.append(np.unique(x))
        missingvalues.append(x.isnull().sum())
           
    fd=pd.DataFrame(data=np.array(lens), columns=['values'])
    fd['columns']=X_train.columns.values
    fd['missingvalue']=missingvalues
    print(fd.info())
    ofilename=filename+'_basic'
    fd.to_csv(ofilename , sep=';')
    return ofilename
    
def _read_missing_columns(filename):
    fd=pd.read_csv(filename, sep=';')
    features = ['missingvalue', 'columns']
    print(fd[features])
    
def _read_numpy(filename):
    return np.genfromtxt(filename, delimiter=',')
def _write_numpy(filename, x):
    np.savetxt(filename, x, delimiter=',')
    
def _select_basedpredictionvalue():
        
    """
    
    print(len(X_train.index))
    X_train['prediction']=_trainLAB.as_matrix()[:,2:]
    X_train=X_train.dropna(subset=['prediction'])
    print(len(X_train.index))
    print(X_train.head(5))
    print(X_trainLAB[2])
    """
    pass



    

