import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.datasets as datasets
import numpy as np

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.svdd_dual_qp import SvddDualQP
from ClusterSVDD.cluster_svdd import ClusterSvdd


def generate_data_uniform(datapoints, cluster_dir_alphas=(10, 10, 10), outlier_frac=0.1, feats=2, noise_feats=0):
    cluster = len(cluster_dir_alphas)
    X = np.zeros((feats, datapoints))
    y = np.zeros(datapoints)

    num_noise = np.int(np.floor(datapoints*outlier_frac))

    samples = np.random.dirichlet(cluster_dir_alphas, 1)[0]
    samples = np.array(samples*(datapoints-num_noise), dtype=np.int)
    print samples, sum(samples)
    if np.sum(samples)+num_noise < datapoints:
        print('Add another sample..')
        num_noise += datapoints-(np.sum(samples)+num_noise)
        print num_noise+np.sum(samples), datapoints

    cnt = num_noise
    for i in range(cluster):
        m = np.random.randn(feats-noise_feats)*8.
        #cov = np.diag(np.random.rand(feats-noise_feats))
        cov = 2.*np.random.rand() * np.eye(feats-noise_feats)
        print cov
        X[:feats-noise_feats, cnt:cnt+samples[i]] = np.random.multivariate_normal(m, cov, samples[i]).T
        y[cnt:cnt+samples[i]] = i+1
        cnt += samples[i]

    mul = np.max(np.abs(X))*2.
    print mul
    X[:, :num_noise] = 2.*mul*(np.random.rand(feats, num_noise)-0.5)
    y[:num_noise] = -1

    X[feats-noise_feats:, :] = 2.*mul*np.random.randn(noise_feats, datapoints)

    # normalize each feature [-1,+1]
    X = X / np.repeat(np.max(np.abs(X), axis=1)[:, np.newaxis], datapoints, axis=1)

    return X, y


def generate_data_moons(datapoints, outlier_frac=0.1, noise_feats=0.05):
    X = np.zeros((datapoints, 2))
    y = np.zeros(datapoints)

    num_noise = np.int(np.floor(datapoints*outlier_frac))

    X[num_noise:, :], y[num_noise:] = datasets.make_moons(n_samples=datapoints-num_noise, noise=noise_feats)
    X = X.T
    y[num_noise:] += 1

    mul = np.max(np.abs(X))*1.5
    print mul
    X[:, :num_noise] = 2.*mul*(np.random.rand(2, num_noise)-0.5)
    y[:num_noise] = -1

    # normalize each feature [-1,+1]
    X = X / np.repeat(np.max(np.abs(X), axis=1)[:, np.newaxis], datapoints, axis=1)

    return X, y


def generate_data(datapoints, norm_dir_alpha=10., anom_dir_alpha=4., anom_cluster=[0, 0, 0, 1, 1, 1], feats=2):
    cluster = len(anom_cluster)
    X = np.zeros((feats, datapoints))
    y = np.zeros(datapoints)

    cluster_dir_alphas = np.array(anom_cluster)*anom_dir_alpha + (1-np.array(anom_cluster))*norm_dir_alpha
    samples = np.random.dirichlet(cluster_dir_alphas, 1)[0]
    samples = np.array(samples*datapoints, dtype=np.int)
    if np.sum(samples) < datapoints:
        print('Add another sample..')
        samples[-1] += 1

    cnt = 0
    anom_lbl = -1
    norm_lbl = 1
    for i in range(cluster):
        sigma = 8.
        if anom_cluster[i] == 1:
            sigma = 1.
        m = np.random.randn(feats)*sigma
        cov = np.diag(np.random.rand(feats))
        print cov
        X[:, cnt:cnt+samples[i]] = np.random.multivariate_normal(m, cov, samples[i]).T
        label = norm_lbl
        if anom_cluster[i] == 1:
            label = anom_lbl
            anom_lbl -= 1
        else:
            label = norm_lbl
            norm_lbl += 1
        y[cnt:cnt+samples[i]] = label
        cnt += samples[i]

    # normalize each feature [-1,+1]
    X = X / np.repeat(np.max(np.abs(X), axis=1)[:, np.newaxis], datapoints, axis=1)

    return X, y


def evaluate(nu, k, data, y, train, test, use_kernel=False, kparam=0.1, plot=False):

    # fix the initialization for all methods
    membership = np.random.randint(0, k, y.size)
    svdds = list()
    for l in range(k):
        if use_kernel:
            svdds.append(SvddDualQP('rbf', kparam, nu))
        else:
            svdds.append(SvddPrimalSGD(nu))

    svdd = ClusterSvdd(svdds)
    svdd.fit(data[:, train].copy(), max_iter=60, init_membership=membership[train])
    scores, classes = svdd.predict(data[:, test].copy())

    # normal classes are positive (e.g. 1,2,3,..) anomalous class is -1
    print y[test]
    true_lbl = y[test]
    true_lbl[true_lbl < 0] = -1  # convert outliers to single outlier class
    ari = metrics.cluster.adjusted_rand_score(true_lbl, classes)
    if nu < 1.0:
        classes[scores > 0.] = -1
        ari = metrics.cluster.adjusted_rand_score(true_lbl, classes)
    print 'ARI=', ari

    fpr, tpr, _ = metrics.roc_curve(y[test]<0., scores, pos_label=1)
    auc = metrics.auc(fpr, tpr, )
    print 'AUC=', auc

    if plot:
        plt.figure(1)
        anom_inds = np.where(y == -1)[0]
        plt.plot(data[0, anom_inds], data[1, anom_inds], '.g', markersize=2)
        nom_inds = np.where(y != -1)[0]
        plt.plot(data[0, nom_inds], data[1, nom_inds], '.r', markersize=6)

        an = np.linspace(0, 2*np.pi, 100)
        for l in range(k):
            r = np.sqrt(svdd.svdds[l].radius2)
            if hasattr(svdd.svdds[l],'c'):
                plt.plot(svdd.svdds[l].c[0], svdd.svdds[l].c[1],
                         'xb', markersize=6, linewidth=2, alpha=0.7)
                plt.plot(r*np.sin(an)+svdd.svdds[l].c[0], r*np.cos(an)+svdd.svdds[l].c[1],
                         '-b', linewidth=2, alpha=0.7)
        plt.show()
    return ari, auc




if __name__ == '__main__':
    num_train = 600
    num_test = 600

    train = np.array(range(num_train), dtype='i')
    test = np.array(range(num_train, num_train + num_test), dtype='i')

    reps = 1
    nus = [0.1, 0.5, 0.8, 1.0]
    ks = [3]
    aris = np.zeros((reps, len(nus),len(ks)))
    aucs = np.zeros((reps, len(nus),len(ks)))

    data, y = generate_data_uniform(num_train + num_test, cluster_dir_alphas=(10, 10, 10), outlier_frac=0.5, feats=2, noise_feats=0)
    # data, y = generate_data(num_train + num_test, norm_dir_alpha=10., anom_dir_alpha=2., anom_cluster=[0, 0, 0, 1, 1, 1, 1, 1, 1], feats=2)
    # data, y = generate_data_moons(num_train + num_test, outlier_frac=0.3, noise_feats=0.05)


    for r in range(reps):
        # data, y = generate_data_uniform(num_train + num_test, cluster_dir_alphas=(10, 10, 10), outlier_frac=0.25, feats=2, noise_feats=0)
        inds = np.random.permutation((num_test + num_train))
        data = data[:, inds]
        y = y[inds]

        # inds = np.where(y>=-1)[0]
        # rinds = np.random.permutation(inds.size)
        # train = inds[rinds[:num_train]]
        # test = np.setdiff1d(np.arange(num_train+num_test), train)

        ssseeed = np.random.randint(low=0, high=1101010)
        for nu in range(len(nus)):
            for k in range(len(ks)):
                np.random.seed(ssseeed)
                aris[r, nu, k], aucs[r, nu, k] = evaluate(nus[nu], ks[k], data, y, train, test, use_kernel=False, kparam=1., plot=False)

    print '\n'
    for nu in range(len(nus)):
        print ''
        for k in range(len(ks)):
            print('k={0} nu={1}: ARI = {2:1.2f}+/-{4:1.2f}   AUC = {3:1.2f}+/-{4:1.2f}'.format(ks[k], nus[nu],
                        np.mean(aris[:, nu, k]), np.mean(aucs[:, nu, k]), np.std(aris[:, nu, k]), np.std(aucs[:, nu, k])))

    print('\nDONE :)')
