import matplotlib.pyplot as plt
import numpy as np

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.svdd_dual_qp import SvddDualQP
from ClusterSVDD.cluster_svdd import ClusterSvdd


def generate_gaussians(datapoints, cluster, noise_frac=0.1, dims=2,
                       non_isotropic=False, distort=False, noise_dims=0, make_uniform=False):
    mean_mul = 10.
    vars = [1., 5.]

    num_noise = np.floor(datapoints*noise_frac)
    num_dpc = np.floor(float(datapoints-num_noise)/float(cluster))

    X = np.zeros((dims, datapoints))
    X[:, :num_noise] = 4.*(2.*np.random.rand(dims, num_noise)-1.)

    y = np.zeros(datapoints)
    y[:num_noise] = -1
    cnt = num_noise

    for i in range(cluster):
        # either generate isotropic clusters or non-isotropic
        v = 0.8*np.random.rand(dims, dims) + 0.2*np.diag(np.ones(dims))
        if not non_isotropic:
            t = np.random.rand()
            v = np.diag( (t*vars[0] + (1.-t)*vars[1]) * np.ones(dims))

        # draw the mean
        m = mean_mul * (2.*np.random.rand(dims, 1)-1.)
        if i == cluster-1:
            num_dpc = datapoints-cnt
        m = m.dot(np.ones((1, num_dpc)))
        # generate the cluster gaussian
        if make_uniform:
            X[:, cnt:cnt+num_dpc] = v.dot(np.random.rand(dims, num_dpc)) + m
        else:
            X[:, cnt:cnt+num_dpc] = v.dot(np.random.randn(dims, num_dpc)) + m
        # fill in some non-informative dimension
        X[dims-noise_dims:dims, cnt:cnt+num_dpc] = np.random.randn(noise_dims, num_dpc)

        # some position dependent rotation
        if distort:
            norms = np.linalg.norm(X, axis=0)
            X[0,:] += 2.0*np.sin(1.5*norms)
            X[1,:] += 1.0*np.cos(.5*norms)
        y[cnt:cnt+num_dpc] = i
        cnt += num_dpc
    return X, y


def generate_stacked_gaussians(datapoints, cluster, dims=2):
    vars = [0.1, 0.9]
    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)
    num_dpc = np.floor(datapoints/cluster)
    cnt = 0
    m = np.zeros((dims, 1))
    for i in range(cluster):
        t = np.random.rand()
        v = np.diag( (t*vars[0] + (1.-t)*vars[1]) * np.ones(dims))
        if i == cluster-1:
            num_dpc = datapoints-cnt
        ms = m.dot(np.ones((1, num_dpc)))
        X[:, cnt:cnt+num_dpc] = v.dot(np.random.randn(dims, num_dpc)) + ms
        y[cnt:cnt+num_dpc] = i
        cnt += num_dpc
    return X, y


def train(cluster, data, nu, membership, use_primal=True):
    svdds = []
    for c in range(cluster):
        if use_primal:
            svdds.append(SvddPrimalSGD(nu))
        else:
            svdds.append(SvddDualQP('rbf', 2.0, nu))
    svdd = ClusterSvdd(svdds)
    cinds = svdd.fit(data, init_membership=membership)
    print cinds
    return svdd, cinds


if __name__ == '__main__':
    nus = [1.0, 0.9, 0.5, 0.1]   # outlier fractions
    nus = [0.1]
    cluster = 3  # 'k' number of clusters for the methods
    cluster_real = 3  # number of clusters to create in the data generation process
    use_primal = False  # use primal sgd svdd or dual kernel qp
    use_nonisotropic = False  # distort the data even further if 'true'

    if use_primal:
        Dtrain, ytrain = generate_gaussians(1000, cluster_real, noise_frac=0.0, non_isotropic=use_nonisotropic)
    else:
        Dtrain, ytrain = generate_stacked_gaussians(1000, cluster_real)
    membership = np.random.randint(0, cluster, ytrain.size)

    # generate test data grid
    delta = 0.1
    x = np.arange(-4.0-delta, 4.0+delta, delta)
    y = np.arange(-4.0-delta, 4.0+delta, delta)
    (X, Y) = np.meshgrid(x, y)
    (sx, sy) = X.shape
    Xf = np.reshape(X,(1, sx*sy))
    Yf = np.reshape(Y,(1, sx*sy))
    Dtest = np.append(Xf, Yf, axis=0)
    if Dtrain.shape[0] > 2:
        Dtest = np.append(Dtest, np.random.randn(Dtrain.shape[0]-2, sx*sy), axis=0)
    print(Dtest.shape)

    # For each \nu in the nus list, train, predict and
    # plot the data
    for i in range(len(nus)):
        if nus[i]==1.0:
            (svdd, cinds) = train(cluster, Dtrain, nus[i], membership, use_primal=True)
        else:
            (svdd, cinds) = train(cluster, Dtrain, nus[i], membership, use_primal=use_primal)
        (res, cres) = svdd.predict(Dtest)

        # The code below is basically only for
        # nice visualizations
        plt.figure(1)
        plt.subplot(1, len(nus), i+1)
        Z = np.reshape(res,(sx, sy))
        cs = plt.contourf(X, Y, Z, alpha=0.5, cmap=plt.cm.bone)
        cs2 = plt.contour(X, Y, Z, [0.0], linewidths=2.0, colors='w', alpha=0.8)
        cs3 = plt.contour(X, Y, np.reshape(cres,(sx, sy)), [0.1, 1.1], colors='k', linewidths=1.0, alpha=0.8)

        cols = np.random.rand(3, cluster)
        cols[:, 0] = np.array([0.95, 0.8, 0.2])
        cols[:, 1] = np.array([0.9, 0.3, 0.7])
        cols[:, 2] = np.array([0.4, 0.4, 0.9])
        for c in range(cluster):
            inds = np.where(cinds == c)[0]
            plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 40, c=cols[:, c])
            if use_primal:
                plt.plot(svdd.svdds[c].c[0], svdd.svdds[c].c[1], 'hw', alpha=0.9, markersize=4)
        # set threshold line style
        for c in cs2.collections:
            c.set_linestyle('dashed')
        for c in cs3.collections:
            c.set_linestyle('dashed')
        # title
        if i == 0:
            plt.title(r'k-means')
        else:
            plt.title(r'ClusterSVDD with $\nu={0}$'.format(nus[i]))
        plt.xlim((-4., 4.))
        plt.ylim((-4., 4.))
        plt.yticks(range(-4, 4), [])
        plt.xticks(range(-4, 4), [])

    plt.show()
    print('finished')
