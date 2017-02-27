import pickle


def _inttoBoolfetaures(features, realfeatures):
    l=len(realfeatures)
    a=set(features)
    b=set([ _i for _i, v in enumerate(realfeatures) if v])
    c=[i for i in a&b]
    pass

def _pickle_write(x, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(x, fp)
        
def _pickle_read(filename):
    with open (filename, 'rb') as fp:
        return pickle.load(fp)
        
        
        
