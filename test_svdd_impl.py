import numpy as np
import matplotlib.pyplot as plt

from svdd_dual_qp import SvddDualQP
from kernel import Kernel
from svdd_primal_sgd import SvddPrimalSGD

if __name__ == '__main__':
    # outlier fraction
    nu = 0.15

    # generate raw training data
    Dtrain = np.array([[0.4,0.1],[0.1,1.1]]).dot(np.random.randn(2, 1500))
    Dtrain = np.random.randn(2, 2500)

    # train dual svdd
    svdd = SvddDualQP('linear', nu)
    svdd.fit(Dtrain)

    # train primal svdd
    psvdd = SvddPrimalSGD(nu)
    psvdd.fit(Dtrain)

    # print solutions
    print('\n  dual-svdd: obj={0}  T={1}.'.format(svdd.pobj, svdd.radius2))
    print('primal-svdd: obj={0}  T={1}.\n'.format(psvdd.pobj, psvdd.radius2))

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

    res = svdd.predict(Dtest)
    pres = psvdd.predict(Dtest)

    # nice visualization
    plt.figure(1)
    Z = np.reshape(res,(sx, sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.0], linewidths=3.0, colors='k')
    plt.scatter(Dtrain[0, svdd.get_support_inds()], Dtrain[1, svdd.get_support_inds()], 40, c='k')
    plt.scatter(Dtrain[0, :], Dtrain[1, :],10)

    plt.figure(2)
    Z = np.reshape(pres,(sx, sy))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z, [0.0], linewidths=3.0, colors='k')
    plt.scatter(Dtrain[0, :], Dtrain[1, :], 10)
    plt.show()

    print('finished')
