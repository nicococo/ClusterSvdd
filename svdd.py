from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
from kernel import Kernel  

class SVDD:
    """Dual QP implementation of the support vector data description (SVDD)."""

    PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

    kernel = None 	# (matrix) our training kernel
    norms = None    # (vector)
    samples = -1 	# (scalar) amount of training data in X

    nu = 0.95	    # (scalar) the regularization constant > 0

    alphas = None	# (vector) dual solution vector
    svs = None      # (vector) support vector indices
    threshold = 0.0	# (scalar) the optimized threshold (rho)

    def __init__(self, kernel, nu):
        self.kernel = kernel
        self.nu = nu
        (self.samples,foo) = kernel.size
        self.norms = matrix(0.0, (self.samples, 1))
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

        # generate a kernel matrix
        P = self.kernel

        # this is the diagonal of the kernel matrix
        q = matrix([0.5*P[i,i] for i in range(N)], (N,1))

        # sum_i alpha_i = A alpha = b = 1.0
        A = matrix(1.0, (1,N))
        b = matrix(1.0, (1,1))

        # 0 <= alpha_i <= h = C
        G1 = spmatrix(1.0, range(N), range(N))
        G = sparse([G1,-G1])
        h1 = matrix(C, (N,1))
        h2 = matrix(0.0, (N,1))
        h = matrix([h1,h2])

        sol = qp(P, -q, G, h, A, b)

        # store solution
        self.alphas = sol['x']

        # find support vectors
        self.svs = []
        for i in range(N):
            if self.alphas[i]>SVDD.PRECISION:
                self.svs.append(i)
        self.svs = matrix(self.svs)

        # find support vectors with alpha < C for threshold calculation
        thres = self.predict(self.kernel[self.svs, self.svs], self.norms[self.svs])
        self.threshold = np.min(thres)
        print('Threshold is {0}'.format(self.threshold))
        return self.alphas, thres

    def get_threshold(self):
        return self.threshold

    def get_alphas(self):
        return self.alphas

    def get_support_inds(self):
        return self.svs

    def get_support(self):
        return self.alphas[self.svs]

    def predict(self, k, norms):
        # number of training examples
        N = len(self.svs)
        (tN, foo) = k.size

        Pc = self.kernel[self.svs, self.svs]
        resc = matrix([dotu(Pc[i,:], self.alphas[self.svs]) for i in range(N)])
        resc = dotu(resc,self.alphas[self.svs])
        res = resc - 2. * matrix([dotu(k[i,:], self.alphas[self.svs]) for i in range(tN)]) + norms
        return res
