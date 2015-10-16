__author__ = 'nicococo'
import numpy as np

from svdd_dual_qp import SvddDualQP

class ClusterSvddDualQP:
    """ Dual QP implementation of the cluster support vector data description (ClusterSVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """

    PRECISION = 1e-4 # important: effects the threshold, support vectors and speed!

    kernel = None 	# (string) kernel name
    kparam = None   # (-) kernel parameter
    samples = -1 	# (scalar) amount of training data in X

    nu = 0.95	    # (scalar) the regularization constant > 0

    clusters = 0    # (scalar) number of clusters
    svdds = None    # (list) list of dual qp svdds

    def __init__(self, clusters, kernel, kparam, nu):
        self.kernel = kernel
        self.kparam = kparam
        self.clusters = clusters
        self.nu = nu
        self.svdds = []
        for c in range(self.clusters):
            self.svdds.append(SvddDualQP(kernel, kparam, nu))
        print('Creating new dual QP cluster SVDD ({2}) with {0} clusters and nu={1}.'.format(self.clusters, nu, kernel))

    def fit(self, X):
        (dims, self.samples) = X.shape

        # init majorization step
        cinds = np.random.randint(0, self.clusters, self.samples)

        # init maximization step
        for c in range(self.clusters):
            inds = np.where(cinds == c)[0]
            self.svdds[c].fit(X[:, inds])

        scores = np.zeros((self.clusters, self.samples))

        for iter in range(10):
            print('Iter={0}'.format(iter))
            # 1. majorization step
            for c in range(self.clusters):
                scores[c, :] = self.svdds[c].predict(X)
            cinds = np.argmin(scores, axis=0)
            # 2. maximization step
            for c in range(self.clusters):
                inds = np.where(cinds == c)[0]
                if inds.size > 0:
                    self.svdds[c].fit(X[:, inds])

        print('Dual QP cluster SVDD training finished.')
        return cinds

    def predict(self, k, norms):
        # number of training examples
        Pc = self.kernel[self.svs, :][:, self.svs]
        aKa = self.get_support().T.dot(Pc.dot(self.get_support()))
        res = aKa - 2. * k.dot(self.get_support()).T + norms
        return res - self.radius2
