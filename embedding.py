import pickle
from util.mathutil import *
from util.featureProcess import *

__author__ = 'PauKung'

class EmbeddingModel(object):
    '''
    Base model for learning embedding
    Parameters:
    Data = N x F numpy matrix (N: number of instances, F: number of features)
    '''

    def __init__(self,X=None):
        if X:
            self.data = X

    def load_data(self,X):
        assert isinstance(X, (np.ndarray, np.generic)), "input must be numpy array or matrix"
        self.data = X

    def save_model(self, fname):
        with open(fname, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, 2)

    def load_model(self, fname):
        with open(fname, 'rb') as infile:
            self.__dict__.update(pickle.load(infile))

    def transform(self):
        raise("not implemented exception")

    def fit(self):
        raise("not implemented exception")
