__author__ = 'nicococo'
import numpy as np

class ClusterSvdd:
    """ Implementation of the cluster support vector data description (ClusterSVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """

    PRECISION = 1e-4 # important: effects the threshold, support vectors and speed!
    clusters = 0    # (scalar) number of clusters
    svdds = None    # (list) list of dual qp svdds

    def __init__(self, svdds):
        self.clusters = len(svdds)
        self.svdds = svdds
        print('Creating new ClusterSVDD with {0} clusters.'.format(self.clusters))

    def fit(self, X, min_chg=0, max_iter=100):
        (dims, samples) = X.shape

        # init majorization step
        cinds_old = np.zeros(samples)
        cinds = np.random.randint(0, self.clusters, samples)

        # init maximization step
        for c in range(self.clusters):
            inds = np.where(cinds == c)[0]
            self.svdds[c].fit(X[:, inds])

        iter = 0
        scores = np.zeros((self.clusters, samples))
        while np.sum(np.abs(cinds_old-cinds))>min_chg and iter < max_iter:
            print('Iter={0}'.format(iter))
            # 1. majorization step
            for c in range(self.clusters):
                scores[c, :] = self.svdds[c].predict(X)
            cinds_old = cinds
            cinds = np.argmin(scores, axis=0)
            # 2. maximization step
            for c in range(self.clusters):
                inds = np.where(cinds == c)[0]
                if inds.size > 0:
                    self.svdds[c].fit(X[:, inds])
            iter += 1

        print('Dual QP cluster SVDD training finished after {0} iterations.'.format(iter))
        return cinds

    def predict(self, Y):
        scores = np.zeros((self.clusters, Y.shape[1]))
        for c in range(self.clusters):
            scores[c, :] = self.svdds[c].predict(Y)
        cinds = np.argmin(scores, axis=0)
        return np.min(scores, axis=0), cinds
