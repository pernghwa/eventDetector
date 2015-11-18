import numpy as np
__author__ = 'PauKung'

def generate_random_mapping(x, batch_size=5000):
    #assert isinstance(x, (np.ndarray, np.generic)), "input has to be numpy array-like object"
    tmp = x.copy()
    rndind = np.arange(x.shape[1])
    np.random.shuffle(rndind)
    segnum = int(x.shape[1]/float(batch_size)) + 1
    for i in range(segnum):
        yield tmp[:, rndind[batch_size*i:min(batch_size*(i+1), x.shape[1])].tolist()], rndind
    del tmp