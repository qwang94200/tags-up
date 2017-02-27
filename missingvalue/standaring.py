
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _standardizingtot(X_train):
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X_train= imputer.fit_transform(X_train)
    X_train = preprocessing.normalize(X_train, norm='l2')
    return X_train
    

def _standardizing(X):
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
    X= imputer.fit_transform(X)
    return StandardScaler().fit_transform(X)

def _covarianceMatrix(X_std):
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    return cov_mat, eig_vals, eig_vecs

def _correlationMatrix(X_std):
    cor_mat1 = np.corrcoef(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
    return cor_mat1, eig_vals, eig_vecs
    
def _calculatecorrelation(X_std):
    _ilength, _jlength=X_std.shape
    values=[]
    for i in xrange(_ilength):
        for j in xrange(i+1, _jlength):
            x=X_std[:,i]
            y=X_std[:,j]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            _value=(i,j, abs(p_value))
            values.append(_value)
    values.sort(key=lambda x: x[2], reverse=True)
    return values
    

def _sortEngenpairs(eig_vals, eig_vecs):
    eig_pairs = [(i, np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[1], reverse=True)
    return eig_pairs

def _explainedVariance(eig_vals, eig_vecs, name): ##choose principal components
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
     
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(139), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(139), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(name+'.png')

def _explainedVariance_single(eig_vals, eig_vecs, name): ##choose principal components
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
     
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(139), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(name+'.pdf')


def _estimate_missing_value(X_train, features):
    missingcounts=[(_i, X_train[feature].count() ) for _i, feature in enumerate(features)]
    missingcounts.sort(key= lambda x: x[1], reverse=True)
    return missingcounts
    

    




