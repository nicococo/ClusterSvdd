__author__ = 'Nico Goernitz'
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

from svdd_dual_qp import SvddDualQP
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
    X[:, cnt:] = 0.6*np.random.randn(dims, num_dpc) \
                 + np.array([-1.5, +1.]).reshape((2, 1)).dot(np.ones((1, num_dpc)))
    y[cnt:] = 1
    return X, y


if __name__ == '__main__':
    nus = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    sigmas = [0.1, 0.25, 0.5, 1.0, 2.0]
    ks = [1, 2, 3, 4]

    reps = 20  # number of repetitions for performance measures
    ntrain = 1000  # total number of data points is ntrain+ntest
    ntest = 2000   # nval is part of ntrain
    nval = 400
    plot = True

    use_kernels = True
    if not use_kernels:
        sigmas = [1.0]

    if plot:
        if use_kernels:
            foo = np.load('res_anom_20_7_rbf.npz')
        else:
            foo = np.load('res_anom_20_7.npz')
        maucs = foo['maucs']
        saucs = foo['saucs']
        nus = foo['nus']
        ks = foo['ks']

        plt.figure(1)
        cols = np.random.rand(maucs.shape[1], 3)
        fmts = ['-x','--o','--D','--s','--H']
        for i in range(maucs.shape[1]):
            plt.errorbar(nus, maucs[:, i], saucs[:, i], fmt=fmts[i], color=cols[i, :], ecolor=cols[i, :], linewidth=2.0, elinewidth=1.0, alpha=0.8)
        plt.xlim((-0.0, 0.21))
        plt.ylim((0.35, 1.05))
        # ticks = nus.astype('|S10')
        # ticks[0] = '1.0=kmeans'
        plt.xticks(nus, ['1','2.5','5','7.5','10','15','20'])
        # plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0 = k-means'], fontsize=14)
        # plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=14)
        plt.grid()
        plt.xlabel(r'$\nu$', fontsize=14)
        plt.ylabel(r'Anomaly Detection Accuracy (in AUROC)', fontsize=14)
        names = list()
        for i in range(maucs.shape[1]):
            names.append('ClusterSVDD (k={0})'.format(ks[i]))
        plt.legend(names, loc=4, fontsize=14)
        plt.show()
        exit(0)

    train = np.array(range(ntrain-nval), dtype='i')
    val = np.array(range(ntrain-nval, ntrain), dtype='i')
    test = np.array(range(ntrain, ntrain+ntest), dtype='i')
    aucs = np.zeros((reps, len(nus), len(ks)))
    for n in range(reps):
        # generate new gaussians
        # data, y = generate_data(ntrain+ntest, outlier_frac=nus[i])
        data, y = generate_data(ntrain+ntest, outlier_frac=0.05)
        inds = np.random.permutation(range(ntest+ntrain))
        data = data[:, inds]
        y = y[inds]
        for i in range(len(nus)):
            for k in range(len(ks)):
                # fix the initialization for all methods
                membership = np.random.randint(0, ks[k], y.size)

                max_auc = -1.0
                max_val_auc = -1.0
                for sigma in sigmas:
                    # build cluster svdd
                    svdds = list()
                    for l in range(ks[k]):
                        if use_kernels:
                            svdds.append(SvddDualQP('rbf', sigma, nus[i]))
                        else:
                            svdds.append(SvddPrimalSGD(nus[i]))

                    svdd = ClusterSvdd(svdds)
                    svdd.fit(data[:, train], init_membership=membership[train])
                    scores_val, _ = svdd.predict(data[:, val])
                    # test on validation data
                    fpr, tpr, _ = metrics.roc_curve(np.array(y[val]<0., dtype='i'), scores_val, pos_label=1)
                    curr_auc = metrics.auc(fpr, tpr)
                    if curr_auc >= max_val_auc:
                        # store test data accuracy
                        scores, _ = svdd.predict(data[:, test])
                        fpr, tpr, _ = metrics.roc_curve(np.array(y[test]<0., dtype='i'), scores, pos_label=1)
                        max_auc = metrics.auc(fpr, tpr)
                        max_val_auc = curr_auc
                aucs[n, i, k] = max_auc

    # means and standard deviations
    maucs = np.mean(aucs, axis=0)
    saucs = np.std(aucs, axis=0)
    print 'AUCs'
    print np.mean(aucs, axis=0)
    print 'Stds'
    print np.std(aucs, axis=0)

    if use_kernels:
        np.savez('res_anom_{0}_{1}_rbf.npz'.format(reps, len(nus)), maucs=maucs, saucs=saucs,
                outlier_frac=nus, ntrain=ntrain, ntest=ntest, reps=reps, nus=nus, ks=ks, sigmas=sigmas)
    else:
        np.savez('res_anom_{0}_{1}.npz'.format(reps, len(nus)), maucs=maucs, saucs=saucs,
                outlier_frac=nus, ntrain=ntrain, ntest=ntest, reps=reps, nus=nus, ks=ks, sigmas=sigmas)

    print('finished.')
