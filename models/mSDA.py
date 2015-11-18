import numpy.matlib
from scipy.sparse import vstack, csr_matrix
from embedding import *

__author__ = 'PauKung'

class mSDA(EmbeddingModel):

    def __init__(self, l=3, prob=0.3, unit="Sigmoid", multimap=False, batch_size=5000):
        funcMap = {"ReLU":relu,"Sigmoid":csr_matrix.tanh}
        super(mSDA, self).__init__(self)
        self.p = prob
        self.layer_num = l
        self.unit = np.tanh
        self.multimap = multimap
        self.batch_size = batch_size
        self.data = None
        if unit:
            self.unit = funcMap[unit]

    @staticmethod
    def mDA(X,p,unit):
        '''
        :param X: data matrix
        :param p: perturbation probability
        :return:
        '''
        X = vstack((X, np.ones((1, X.shape[1]))))
        d = X.shape[0]
        q = np.ones((d,1))
        q[:-1,0] *= (1-p)
        S = X.dot(X.T)
        Q = S * (q.dot(q.T))
        Q[np.diag_indices(Q.shape[0])] = q.reshape((q.shape[0],)) * S.diagonal()
        P = S * csr_matrix(np.matlib.repmat(q.T,d,1))
        W = P[:-1,:].dot(csr_matrix(np.linalg.inv(Q+1e-5*np.eye(d))))
        h = unit(csr_matrix(W).dot(X))
        return W, h

    def fit(self, X=None):
        if X is None:
            assert self.data is not None, "no data initialized"
        else:
            self.data = X
        d, n = self.data.shape
        self.Ws = np.zeros((d, d+1, self.layer_num))

        def internal(data,append=False):
            d, n = data.shape
            hs = [csr_matrix((d, n), dtype=np.float) for i in range(self.layer_num+1)] #np.zeros((d, n, self.layer_num+1))
            Ws = [csr_matrix((d, d+1), dtype=np.float) for i in range(self.layer_num)] #np.zeros((d, d+1, self.layer_num))
            hs[0] = data
            for t in range(self.layer_num):
                print "processing layer ", t
                Ws[t], hs[t+1] = mSDA.mDA(hs[t], self.p, self.unit)
            if append:
                try:
                    self.WsList.append(Ws)
                except Exception:
                    self.WsList = [Ws]
            else:
                self.WsList = [Ws]

        if self.multimap:
            for data, rndidx in generate_random_mapping(self.data, batch_size=self.batch_size):
                internal(data, append=True)
                self.rndidx = rndidx
        else:
            internal(self.data)

    def transform(self, test_data):
        h = test_data
        for i in range(self.layer_num):
            if self.multimap:
                new_h = self.unit(np.sum(np.array([self.WsList[j][i] * h[self.rndidx[self.batch_size*j:min(self.batch_size*(j+1),
                                        len(self.rndidx))]] for j in range(len(self.WsList))]), axis=0) / float(self.batch_size))
            else:
                new_h = self.unit(self.WsList[0][i].dot(h))
            h = new_h
        del new_h
        return h