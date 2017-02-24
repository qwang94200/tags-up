
import missingvalue._missingvalue as _missingvalues
import missingvalue._metamodels as _metamodels
import pickle

def test(X,y, ofilename):
    getSupport = _missingvalues(X,y)
    _ifeatures = [_ifeature for _ifeature, v in enumerate(getSupport) if v]
    _metamodels._metamodelApply(X[_ifeatures],y)
    with open(ofilename, 'wb') as fp:
        pickle.dump(getSupport, fp)
    
def test2(X,y, getSupport):
    
    
def totTest(X,y):
    file1= '1.csv'
    test(X,y,file1)
    getSupport = pickle.load( open( file1, "rb" ))
    
    test2(X,y, getSupport)
    
