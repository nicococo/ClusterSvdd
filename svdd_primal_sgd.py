__author__ = 'nicococo'
from cvxopt import matrix,spmatrix,sparse
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
import numpy as np
from kernel import Kernel

class SvddPrimalSGD:
    """ Primal subgradient descent solver for the support vector data description (SVDD).
        Author: Nico Goernitz, TU Berlin, 2015
    """

    PRECISION = 10**-3 # important: effects the threshold, support vectors and speed!

    nu = 0.95	    # (scalar) the regularization constant > 0

    c = None        # (vecor) center of the hypersphere
    radius2 = 0.0	# (scalar) the optimized threshold (rho)

    pobj = 0.0      # (scalar) primal objective after training

    def __init__(self, nu):
        self.nu = nu
        print('Creating new primal SVDD with nu={0}.'.format(nu))



    def fit(self, X, max_iter=20000, prec=1e-4, rate=0.001):
        """ Trains a svdd in primal.
            Subgradient descent solver.
        """
        (dims, samples) = X.shape

        if (samples<1):
            print('Invalid training data.')
            return -1

        # number of training examples
        N = samples
        C = 1./(samples*self.nu)

        c = np.mean(X, axis=1)  # this is actually a very good starting position
        dist = c.T.dot(c) - 2.*c.T.dot(X) + np.sum(X*X, axis=0)
        T = 0.4 * np.max(dist) * (1.0-self.nu)  # starting heuristic T
        # if nu exceeds 1.0, then T^* is always 0 and c can
        # be computed analytically (as center-of-mass, mean)
        if self.nu >= 1.0:
            self.c = c
            self.radius2 = T
            self.pobj = 0.0  # TODO: calculate real primal objective
            return c, T

        is_converged = False
        obj_best = 1e20
        obj_bak = -100.
        iter = 0
        while not is_converged and iter < max_iter:
            # calculate the distances of the center to each datapoint
            dist = c.T.dot(c) - 2.*c.T.dot(X) + np.sum(X*X, axis=0)
            inds = np.where(dist - T >= 1e-12)[0]
            # we need at least 1 entry, hence lower T to the maximum entry
            if inds.size == 0:
                inds = np.argmax(dist)
                T = dist[inds]

            # real objective value given the current center c and threshold T
            obj = T + C*np.sum(dist[inds] - T)

            # this is subgradient, hence need to store the best solution so far
            if obj_best >= obj:
                self.c = c
                self.radius2 = T
                obj_best = obj

            # stop, if progress is too slow
            if np.abs((obj-obj_bak)/obj) < prec:
                print('Iter={2}: obj={0}  T={1}  #nnz={4} rel_change={3}'.format(obj, T, iter+1, np.abs((obj-obj_bak)/obj), inds.size ))
                is_converged = True
                continue
            obj_bak = obj

            # stepsize should be not more than 0.1 % of the maximum value encountered in dist
            max_change = rate * np.max(dist)
            # gradient step for threshold
            dT = 1 - C*np.float(inds.size)
            T -= np.sign(dT) * max_change
            # gradient step for center
            dc = 2*C*np.sum(c.reshape((dims, 1)).dot(np.ones((1, inds.size))) - X[:, inds], axis=1)
            c -= dc/np.linalg.norm(dc) * max_change

            iter += 1

        dist = self.c.T.dot(self.c) - 2.*self.c.T.dot(X) + np.sum(X*X, axis=0)
        inds = np.where(dist - self.radius2 > 0.0)[0]
        obj = self.radius2 + C*np.sum(dist[inds] - self.radius2)
        self.pobj = obj
        print('Iter={2}: obj={0}  T={1}'.format(obj, self.radius2, iter+1))

        return self.c, self.radius2

    def get_radius(self):
        return self.radius2

    def predict(self, X):
        # X : (dims x samples)
        dist = self.c.T.dot(self.c) - 2.*self.c.T.dot(X) + np.sum(X*X, axis=0)
        return dist - self.radius2
