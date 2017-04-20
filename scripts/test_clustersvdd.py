
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.cluster_svdd import ClusterSvdd


def generate_gaussians(datapoints, cluster, noise_frac=0.1, dims=2):
    mean_mul = 50.
    vars = [4.1, 4.1]

    num_noise = np.int(np.floor(datapoints*noise_frac))
    num_dpc = np.int(np.floor(float(datapoints-num_noise)/float(cluster)))

    X = np.zeros((dims, datapoints))
    X[:, :num_noise] = 100.*(2.*np.random.rand(dims, num_noise)-1.)

    y = np.zeros(datapoints)
    y[:num_noise] = -1
    cnt = num_noise

    for i in range(cluster):
        t = 4.
        v = np.diag( (t*vars[0] + (1.-t)*vars[1]) * np.ones(dims))

        # draw the mean
        m = mean_mul * (4.*np.random.rand(dims, 1)-2.)
        if i == cluster-1:
            num_dpc = datapoints-cnt
        m = m.dot(np.ones((1, num_dpc)))
        # generate the cluster gaussian
        X[:, cnt:cnt+num_dpc] = v.dot(4.*np.random.randn(dims, num_dpc)) + m

        y[cnt:cnt+num_dpc] = i
        cnt += num_dpc

    # # normalize each feature
    X = X / np.repeat(np.max(np.abs(X), axis=1)[:, np.newaxis]/2., datapoints, axis=1)
    return X, y


def train(cluster, data, nu, membership):
    svdds = []
    for c in range(cluster):
        svdds.append(SvddPrimalSGD(nu))
    svdd = ClusterSvdd(svdds, nu=nu)
    cinds = svdd.fit(data, init_membership=membership, max_svdd_iter=10000, max_iter=40)
    print cinds
    return svdd, cinds


if __name__ == '__main__':
    np.random.seed(1000)
    nu = 0.1  # CLUSTER - DUAL, PRIMAL
    n_cluster = 3  # 'k' number of clusters for the methods and data generation

    Dtrain, ytrain = generate_gaussians(1000, n_cluster, noise_frac=0.01)
    membership = np.random.randint(0, n_cluster, ytrain.size)

    # generate test data grid
    delta = 0.1
    x = np.arange(-2.0-delta, 2.0+delta, delta)
    y = np.arange(-2.0-delta, 2.0+delta, delta)
    (X, Y) = np.meshgrid(x, y)
    (sx, sy) = X.shape
    Xf = np.reshape(X,(1, sx*sy))
    Yf = np.reshape(Y,(1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)

    # The code below is basically only for beautiful visualizations
    plt.figure(1)

    # For each \nu in the nus list, train, predict and
    svdd, cinds = train(n_cluster, Dtrain, nu, membership)
    scores, cres = svdd.predict(Dtrain)
    res, cres = svdd.predict(Dtest)

    Z = np.reshape(res,(sx, sy))
    cs = plt.contourf(X, Y, Z, cmap=plt.cm.bone, alpha=0.2)

    cols = np.random.rand(3, n_cluster+1)
    cols[:, 0] = np.array([0.95, 0.1, 0.1])
    cols[:, 1] = np.array([0.9, 0.3, 0.7])
    cols[:, 2] = np.array([0.4, 0.9, 0.3])
    cols[:, 3] = np.array([0.4, 0.4, 0.9])
    for c in range(n_cluster):
        inds = np.where(cinds == c)[0]
        plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 30, alpha=0.7, c=cols[:, c])
        pl.gca().add_patch(pl.Circle((svdd.svdds[c].c[0],svdd.svdds[c].c[1]),
                                     np.sqrt(svdd.svdds[c].radius2), alpha=0.6,
                                     color=cols[:, c], fill=True))

    plt.xlim((-2., 2.))
    plt.ylim((-2., 2.))
    plt.yticks([], [])
    plt.xticks([], [])

    plt.show()
    pl.show()
    print('finished')
