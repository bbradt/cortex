import numpy as np
import matlab.engine

class MKMeans():
    def __init__(self, eng=None, n_clusters=8, init='plus', n_init=10, max_iter=300, distance='sqeuclidean'):
        self.eng = eng
        self.n_clusters = n_clusters
        if type(init) is np.ndarray:
            self.init = matlab.double(init.tolist())
        else:
            self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.distance = distance

    def fit(self, X):   
        out = self.eng.kmeans(matlab.double(X.tolist()), self.n_clusters, 'Distance', self.distance, 'Replicates', self.n_init, 'Start', self.init, nargout=2)
        self.labels_ = np.array(out[0]).flatten().reshape(len(out[0])) - 1
        self.cluster_centers_ = np.array(out[1])
        return self
