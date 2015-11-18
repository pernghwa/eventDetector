import pickle

class clusterTestBase(object):

    '''
    test base class
    '''

    def load_data(self, path):
        raise("not implemented exception")

    def fit_model(self):
        raise("not implemented exception")

    def learn_clusters(self):
        raise("not implemented exception")

    def draw_clusters(self):
        raise("not implemented exception")

    def map_clusters(self):
        raise("not implemented exception")

    def save_model(self, fname):
        with open(fname, 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, 2)

    def load_model(self, fname):
        with open(fname, 'rb') as infile:
            self.__dict__.update(pickle.load(infile))
