import numpy as np
import matplotlib.pyplot as plt

from svdd_primal_sgd import SvddPrimalSGD
from svdd_dual_qp import SvddDualQP
from cluster_svdd import ClusterSvdd

if __name__ == '__main__':
    # outlier fraction
    nu = 0.3
    cluster = 3

    # generate raw training data
    Dtrain = 0.5*np.random.randn(2, 1200)
    Dtrain[:, :400] += 1.
    Dtrain[:, 400:800] -= 1.
    Dtrain[0, 800:] -= 2.
    Dtrain[1, 800:] += 2.

    # use dual svdd
    svdds = []
    for c in range(cluster):
        # svdds.append(SvddDualQP('rbf', 2.0, nu))
        svdds.append(SvddPrimalSGD(nu))
    svdd = ClusterSvdd(svdds)
    cinds = svdd.fit(Dtrain)
    print cinds

    # generate test data grid
    delta = 0.1
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    (sx, sy) = X.shape
    Xf = np.reshape(X,(1, sx*sy))
    Yf = np.reshape(Y,(1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    if Dtrain.shape[0] > 2:
        Dtest = np.append(Dtest, np.random.randn(Dtrain.shape[0]-2, sx*sy), axis=0)
    print(Dtest.shape)

    res, cres = svdd.predict(Dtest)
    print res.shape
    # pres = psvdd.predict(Dtest)

    # nice visualization
    plt.figure(1)
    plt.subplot(1, 2, 1)
    Z = np.reshape(res,(sx, sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.0], linewidths=3.0, colors='k')
    # plt.scatter(Dtrain[0, svdd.get_support_inds()], Dtrain[1, svdd.get_support_inds()], 40, c='k')
    for c in range(cluster):
        inds = np.where(cinds == c)[0]
        plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 40, c=np.random.rand(3))

    plt.subplot(1, 2, 2)
    Z = np.reshape(cres,(sx, sy))
    plt.contourf(X, Y, Z)
    for c in range(cluster):
        inds = np.where(cinds == c)[0]
        plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 40, c=np.random.rand(3))

    plt.show()
    print('finished')
