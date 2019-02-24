import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()

class MKMeans():
    def __init__(self, n_clusters=8, init='plus', n_init=10, max_iter=300, distance='sqeuclidean'):
        self.n_clusters = n_clusters
        if type(init) is np.ndarray:
            self.init = matlab.double(init)
        else:
            self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.distance = distance

    def fit(self, X):    
        out = eng.kmeans(matlab.double(X), self.n_clusters, 'Distance', self.distance, 'Replicates', self.init, 'MaxIter', self.max_iter, 'Start', self.start, nargout=2)
        self.labels_ = out[0].flatten().reshape(len(out[0])).tolist()
        self.cluster_centers_ = np.array(out[1])
        return self
