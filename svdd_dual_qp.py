from cvxopt import matrix,spmatrix,sparse
from cvxopt.solvers import qp
import numpy as np

class SvddDualQP:
    """ Dual QP implementation of the support vector data description (SVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """

    PRECISION = 1e-4 # important: effects the threshold, support vectors and speed!

    kernel = None 	# (matrix) our training
    norms = None    # (vector)
    samples = -1 	# (scalar) amount of training data in X

    nu = 0.95	    # (scalar) the regularization constant > 0

    alphas = None	# (vector) dual solution vector
    svs = None      # (vector) support vector indices
    radius2 = 0.0	# (scalar) the optimized threshold (rho)

    pobj = 0.0      # (scalar) primal objective value after training

    def __init__(self, kernel, nu):
        self.kernel = kernel
        self.nu = nu
        (self.samples, foo) = kernel.shape
        self.norms = np.zeros((self.samples))
        for i in range(self.samples):
            self.norms[i] = kernel[i, i]
        print('Creating new SVDD with {0} samples and nu={1}.'.format(self.samples, nu))



    def fit(self):
        """Trains an one-class svm in dual with kernel."""
        if (self.samples<1):
            print('Invalid training data.')
            return -1

        # number of training examples
        N = self.samples
        C = 1./(self.samples*self.nu)
        if self.nu >= 1.0:
            print("Center-of-mass solution.")
            self.alphas = np.ones(self.samples)/float(self.samples)
            self.radius2 = 0.0
            self.svs = np.array(range(self.samples))
            self.pobj = 0.0  # TODO: calculate real primal objective
            return self.alphas, self.radius2

        # generate a kernel matrix
        P = 2.0*matrix(self.kernel)

        # this is the diagonal of the kernel matrix
        q = matrix(self.norms)

        # sum_i alpha_i = A alpha = b = 1.0
        A = matrix(1.0, (1,N))
        b = matrix(1.0, (1,1))

        # 0 <= alpha_i <= h = C
        G1 = spmatrix(1.0, range(N), range(N))
        G = sparse([G1, -G1])
        h1 = matrix(C, (N,1))
        h2 = matrix(0.0, (N,1))
        h = matrix([h1,h2])

        sol = qp(P, -q, G, h, A, b)

        # store solution
        self.alphas = np.array(sol['x'])
        self.pobj = -sol['primal objective']

        # find support vectors
        self.svs = np.where(self.alphas >= self.PRECISION)[0]
        print np.sum(self.alphas)
        print self.svs.shape
        # find support vectors with alpha < C for threshold calculation
        thres = self.predict(self.kernel[self.svs, :][:, self.svs], self.norms[self.svs])
        self.radius2 = np.min(thres)
        print('Threshold is {0}'.format(self.radius2))
        return self.alphas, thres

    def get_radius(self):
        return self.radius2

    def get_alphas(self):
        return self.alphas

    def get_support_inds(self):
        return self.svs

    def get_support(self):
        return self.alphas[self.svs]

    def predict(self, k, norms):
        # number of training examples
        Pc = self.kernel[self.svs, :][:, self.svs]
        aKa = self.get_support().T.dot(Pc.dot(self.get_support()))
        res = aKa - 2. * k.dot(self.get_support()).T + norms
        return res - self.radius2
