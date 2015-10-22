import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

from svdd_primal_sgd import SvddPrimalSGD
from cluster_svdd import ClusterSvdd


def generate_data(datapoints, outlier_frac=0.1, dims=2):
    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)

    num_noise = np.floor(datapoints*outlier_frac)
    num_dpc = np.floor(float(datapoints-num_noise)/2.0)

    X[:, :num_noise] = 0.5*np.random.randn(dims, num_noise) + 0.
    y[:num_noise] = -1

    cnt = num_noise
    X[:, cnt:cnt+num_dpc] = 1.5*np.random.randn(dims, num_dpc) - 1.
    y[cnt:cnt+num_dpc] = 1
    cnt += num_dpc

    X[:, cnt:] = 0.5*np.random.randn(dims, num_dpc) + 1.
    y[cnt:] = 2
    return X, y


if __name__ == '__main__':
    nus = (np.arange(1, 21)/20.)[::-1]
    outlier_frac = 0.2  # fraction of uniform noise in the generated data
    reps = 2  # number of repetitions for performance measures
    ntrain = 1000
    ntest = 2000
    plot = False

    if plot:
        foo = np.load('res_10_20.npz')
        maris = foo['maris']
        saris = foo['saris']
        nus = foo['nus']
        print maris

        plt.figure(1)
        plt.errorbar(nus, maris, saris, fmt='.-.b', linewidth=1.0, elinewidth=2.0, alpha=0.8)
        plt.errorbar(nus[0], maris[0], saris[0], fmt='.-.r', linewidth=1.0, elinewidth=4.0, alpha=1.0)
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0 = k-means'], fontsize=14)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=14)
        plt.grid()
        plt.xlabel(r'regularization parameter $\nu$', fontsize=14)
        plt.ylabel(r'Adjusted Rand Index (ARI)', fontsize=14)
        names = ['ClusterSVDD',r'$k$-means']
        plt.legend(names, loc=1, fontsize=14)
        plt.show()

        exit(0)

    train = np.array(range(ntrain), dtype='i')
    test = np.array(range(ntrain, ntrain+ntest), dtype='i')

    aris = np.zeros((reps, len(nus)))
    for n in range(reps):
        # generate new gaussians
        data, y = generate_data(ntrain+ntest, outlier_frac=outlier_frac)
        inds = np.random.permutation(range(ntest+ntrain))
        data = data[:, inds]
        y = y[inds]
        # fix the initialization for all methods
        membership = np.random.randint(0, 2, y.size)

        for i in range(len(nus)):
            svdds = [SvddPrimalSGD(nus[i]), SvddPrimalSGD(nus[i])]
            svdd = ClusterSvdd(svdds)
            svdd.fit(data[:, train], init_membership=membership[train])
            _, classes = svdd.predict(data[:, test])
            # evaluate clustering abilities
            inds = np.where(y[test] >= 0)[0]
            # print y[test[inds]]
            aris[n, i] = metrics.cluster.adjusted_rand_score(y[test[inds]], classes[inds])


    maris = np.mean(aris, axis=0)
    saris = np.std(aris, axis=0)
    print np.mean(aris, axis=0)
    print np.std(aris, axis=0)
    np.savez('res_robust_{0}_{1}.npz'.format(reps, len(nus)), maris=maris, saris=saris,
             outlier_frac=outlier_frac, ntrain=ntrain, ntest=ntest, reps=reps, nus=nus)

    print('finished')
