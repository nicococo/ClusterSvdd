__author__ = 'Nico Goernitz'
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

    X[:, :num_noise] = 0.15*np.random.randn(dims, num_noise) \
                       + np.array([0.7, -0.]).reshape((2, 1)).dot(np.ones((1, num_noise)))
    y[:num_noise] = -1

    cnt = num_noise
    X[:, cnt:cnt+num_dpc] = 0.5*np.random.randn(dims, num_dpc) \
                            + np.array([-1.5, -2.]).reshape((2, 1)).dot(np.ones((1, num_dpc)))
    y[cnt:cnt+num_dpc] = 1
    cnt += num_dpc

    num_dpc = datapoints-cnt
    X[:, cnt:] = 0.5*np.random.randn(dims, num_dpc) \
                 + np.array([-1.5, +2.]).reshape((2, 1)).dot(np.ones((1, num_dpc)))
    y[cnt:] = 1
    return X, y


if __name__ == '__main__':
    nus = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    reps = 5  # number of repetitions for performance measures
    ntrain = 1000
    ntest = 2000
    plot = True

    if plot:
        foo = np.load('res_anom_50_7.npz')
        maucs1 = foo['maucs1']
        maucs2 = foo['maucs2']
        saucs1 = foo['saucs1']
        saucs2 = foo['saucs2']
        nus = foo['nus']

        plt.figure(1)
        plt.errorbar(nus, maucs1, saucs1, fmt='.-.b', linewidth=1.0, elinewidth=2.0, alpha=0.8)
        plt.errorbar(nus, maucs2, saucs2, fmt='.-.r', linewidth=1.0, elinewidth=2.0, alpha=1.0)
        plt.xlim((-0.0, 0.21))
        plt.ylim((0.35, 1.05))
        # ticks = nus.astype('|S10')
        # ticks[0] = '1.0=kmeans'
        plt.xticks(nus, ['1','2.5','5','7.5','10','15','20'])
        # plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0 = k-means'], fontsize=14)
        # plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=14)
        plt.grid()
        plt.xlabel(r'Percentage of anomalies in the dataset', fontsize=14)
        plt.ylabel(r'Anomaly Detection Accuracy (in AUROC)', fontsize=14)
        names = ['ClusterSVDD',r'SVDD']
        plt.legend(names, loc=1, fontsize=14)
        plt.show()
        exit(0)

    train = np.array(range(ntrain), dtype='i')
    test = np.array(range(ntrain, ntrain+ntest), dtype='i')
    aucs1 = np.zeros((reps, len(nus)))
    aucs2 = np.zeros((reps, len(nus)))
    for n in range(reps):
        for i in range(len(nus)):
            # generate new gaussians
            data, y = generate_data(ntrain+ntest, outlier_frac=nus[i])
            inds = np.random.permutation(range(ntest+ntrain))
            data = data[:, inds]
            y = y[inds]
            # fix the initialization for all methods
            membership = np.random.randint(0, 2, y.size)

            # cluster svdd
            svdds = [SvddPrimalSGD(nus[i]), SvddPrimalSGD(nus[i])]
            svdd = ClusterSvdd(svdds)
            svdd.fit(data[:, train], init_membership=membership[train])
            scores, _ = svdd.predict(data[:, test])
            # evaluate outlier detection abilities
            fpr, tpr, thresholds = metrics.roc_curve(np.array(y[test]<0., dtype='i'), scores, pos_label=1)
            aucs1[n, i] = metrics.auc(fpr, tpr)

            # svdd
            svdd = SvddPrimalSGD(nus[i])
            svdd.fit(data[:, train])
            scores = svdd.predict(data[:, test])
            # evaluate outlier detection abilities
            fpr, tpr, thresholds = metrics.roc_curve(np.array(y[test]<0., dtype='i'), scores, pos_label=1)
            aucs2[n, i] = metrics.auc(fpr, tpr)

    # means and standard deviations
    maucs1 = np.mean(aucs1, axis=0)
    saucs1 = np.std(aucs1, axis=0)
    print np.mean(aucs1, axis=0)
    print np.std(aucs1, axis=0)

    maucs2 = np.mean(aucs2, axis=0)
    saucs2 = np.std(aucs2, axis=0)
    print np.mean(aucs2, axis=0)
    print np.std(aucs2, axis=0)

    np.savez('res_anom_{0}_{1}.npz'.format(reps, len(nus)), maucs1=maucs1, saucs1=saucs1, maucs2=maucs2, saucs2=saucs2,
            outlier_frac=nus, ntrain=ntrain, ntest=ntest, reps=reps, nus=nus)

    print('finished.')
