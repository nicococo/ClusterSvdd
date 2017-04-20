
import matplotlib.pyplot as plt
import numpy as np

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.svdd_dual_qp import SvddDualQP
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
        t = np.random.rand()
        v = np.diag( (t*vars[0] + (1.-t)*vars[1]) * np.ones(dims))

        # draw the mean
        m = mean_mul * (2.*np.random.rand(dims, 1)-1.)
        if i == cluster-1:
            num_dpc = datapoints-cnt
        m = m.dot(np.ones((1, num_dpc)))
        # generate the cluster gaussian
        X[:, cnt:cnt+num_dpc] = v.dot(np.random.randn(dims, num_dpc)) + m

        y[cnt:cnt+num_dpc] = i
        cnt += num_dpc

    # # normalize each feature
    X = X / np.repeat(np.max(np.abs(X), axis=1)[:, np.newaxis]/2., datapoints, axis=1)
    return X, y


def train(cluster, data, nu, membership, use_primal=True):
    svdds = []
    for c in range(cluster):
        if use_primal:
            svdds.append(SvddPrimalSGD(nu))
        else:
            svdds.append(SvddDualQP('rbf', 0.4, nu))
    svdd = ClusterSvdd(svdds, nu=nu)
    cinds = svdd.fit(data, init_membership=membership, max_svdd_iter=10000, max_iter=40)
    print cinds
    return svdd, cinds


if __name__ == '__main__':
    np.random.seed(10)
    nus = [0.14] # ANOM - PRIMAL
    nus = [0.07]  # ANOM - DUAL

    nus = [0.8]  # CLUSTER - DUAL, PRIMAL
    n_cluster = 4  # 'k' number of clusters for the methods and data generation
    use_primal = True
    # use primal sgd svdd or dual kernel qp
    ad_setting = True  # either ad or cluster setting

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

    # For each \nu in the nus list, train, predict and
    # plot the data
    for i in range(len(nus)+2):
        if 0 < i < len(nus)+1:
            (svdd, cinds) = train(n_cluster, Dtrain, nus[i - 1], membership, use_primal=use_primal)
            (scores, cres) = svdd.predict(Dtrain)
            print 'Fraction {0}-{1}'.format(nus[i-1], np.float(np.sum(scores>=0.)) / np.float(scores.size))
            (res, cres) = svdd.predict(Dtest)
        elif i == 0:
            if ad_setting:
                (svdd, cinds) = train(1, Dtrain, nus[i], membership, use_primal=use_primal)
            else:
                (svdd, cinds) = train(n_cluster, Dtrain, 1.0, membership, use_primal=use_primal)
            (scores, cres) = svdd.predict(Dtrain)
            print 'Fraction {0}-{1}'.format(nus[i], np.float(np.sum(scores>=0.)) / np.float(scores.size))
            (res, cres) = svdd.predict(Dtest)
        else:
            scores = ytrain < 0
            cinds = ytrain

        # The code below is basically only for beautiful visualizations
        plt.figure(1)
        plt.subplot(1, len(nus)+2, (i+1) % (len(nus)+2)+1)
        if i < len(nus)+1:
            Z = np.reshape(res,(sx, sy))
            # cs = plt.contourf(X, Y, Z, alpha=0.5, cmap=plt.cm.bone)
            if ad_setting:
                cs2 = plt.contour(X, Y, Z, [0.0], linewidths=2.0, colors='w', alpha=0.8)

        if not ad_setting:
            cols = np.random.rand(3, n_cluster+1)
            cols[:, 0] = np.array([0.95, 0.1, 0.1])
            cols[:, 1] = np.array([0.9, 0.3, 0.7])
            cols[:, 2] = np.array([0.4, 0.9, 0.3])
            cols[:, 3] = np.array([0.4, 0.4, 0.9])
            cols[:, 4] = np.array([0.7, 0.8, 0.99])

            if i > len(nus):
                cols[1,:] = cols[1, np.array([1, 2, 3, 4, 0])]

            for c in range(n_cluster+1):
                inds = np.where(cinds == c-1)[0]
                plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 30, c=cols[:, c])
        else:
            # anomaly detection setting
            inds = np.where(scores > 0.)[0]
            plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 30, c='r')
            inds = np.where(scores <= 0.)[0]
            plt.scatter(Dtrain[0, inds], Dtrain[1, inds], 30, c='g')

        # title
        if i == 0:
            if use_primal:
                if ad_setting:
                    plt.title(r'SVDD', fontsize=16)
                else:
                    plt.title(r'K-Means', fontsize=16)
            else:
                if ad_setting:
                    plt.title(r'Kernel SVDD', fontsize=16)
                else:
                    plt.title(r'Kernel K-Means', fontsize=16)
        elif i < len(nus)+1:
            if use_primal:
                plt.title(r'ClusterSVDD', fontsize=16)
            else:
                plt.title(r'Kernel ClusterSVDD', fontsize=16)
        else:
            plt.title(r'Ground truth', fontsize=16)
        plt.xlim((-2., 2.))
        plt.ylim((-2., 2.))
        plt.yticks(range(-2, 2), [])
        plt.xticks(range(-2, 2), [])

    plt.show()
    print('finished')
