import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.svdd_dual_qp import SvddDualQP
from ClusterSVDD.cluster_svdd import ClusterSvdd


def generate_data(datapoints, cluster_dir_alphas=(10, 10, 10), outlier_frac=0.1, dims=2):
    cluster = len(cluster_dir_alphas)
    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)

    num_noise = np.int(np.floor(datapoints*outlier_frac))
    num_dpc = np.int(np.floor(float(datapoints-num_noise) / np.float(cluster)))

    samples = np.random.dirichlet(cluster_dir_alphas, 1)[0]
    samples = np.array(samples*num_dpc, dtype=np.int)
    if np.sum(samples) < num_dpc:
        print('Add another sample..')
        samples[-1] += 1

    cnt = num_noise
    for i in range(cluster):
        m = np.random.randn(dims)*4.
        cov = np.diag(np.random.rand(dims))
        print cov
        X[:, cnt:cnt+samples[i]] = np.random.multivariate_normal(m, cov, samples[i]).T
        y[cnt:cnt+samples[i]] = i+1
        cnt += samples[i]

    mul = np.max(np.abs(X))*1.5
    print mul
    X[:, :num_noise] = 2.*mul*(np.random.rand(dims, num_noise)-0.5)
    y[:num_noise] = -1


    X /= np.max(np.abs(X))

    return X, y


def evaluate(nu, k, outlier_frac, num_train, num_test):
    train = np.array(range(num_train), dtype='i')
    test = np.array(range(num_train, num_train + num_test), dtype='i')

    # generate new gaussians
    data, y = generate_data(num_train + num_test, outlier_frac=outlier_frac)
    inds = np.random.permutation(range(num_test + num_train))
    data = data[:, inds]
    y = y[inds]

    # fix the initialization for all methods
    membership = np.random.randint(0, k, y.size)
    svdds = list()
    for l in range(k):
        svdds.append(SvddPrimalSGD(nu))
        # svdds.append(SvddDualQP('rbf', 0.5, nu))
    svdd = ClusterSvdd(svdds)
    svdd.fit(data[:, train].copy(), max_iter=100, init_membership=membership[train])
    scores, classes = svdd.predict(data[:, test].copy())

    # evaluate clustering abilities
    inds = np.where(y[test] >= 0)[0]
    ari = metrics.cluster.adjusted_rand_score(y[test[inds]], classes[inds])
    print 'ARI=', ari
    fpr, tpr, _ = metrics.roc_curve(y[test]<0., scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print 'AUC=', auc

    plt.figure(1)
    anom_inds = np.where(y == -1)[0]
    plt.plot(data[0, anom_inds], data[1, anom_inds], '.g', markersize=2)
    nom_inds = np.where(y != -1)[0]
    plt.plot(data[0, nom_inds], data[1, nom_inds], '.r', markersize=6)

    an = np.linspace(0, 2*np.pi, 100)
    for l in range(k):
        r = np.sqrt(svdd.svdds[l].radius2)
        #print l, svdd.svdds[l].c[0], svdd.svdds[l].c[1], np.sqrt(svdd.svdds[l].radius2)
        plt.plot(svdd.svdds[l].c[0], svdd.svdds[l].c[1],
                 'xb', markersize=6, linewidth=2, alpha=0.7)
        plt.plot(r*np.sin(an)+svdd.svdds[l].c[0], r*np.cos(an)+svdd.svdds[l].c[1],
                 '-b', linewidth=2, alpha=0.7)

    plt.show()



if __name__ == '__main__':
    k = 2
    nu = 0.1

    outlier_frac = 0.1  # fraction of uniform noise in the generated data
    num_train = 1000
    num_test = 1000

    ssseeed = np.random.randint(low=0, high=1101010)
    np.random.seed(ssseeed)
    evaluate(nu, 1, outlier_frac, num_train, num_test)
    np.random.seed(ssseeed)
    evaluate(nu, 2, outlier_frac, num_train, num_test)
    np.random.seed(ssseeed)
    evaluate(1.0, 2, outlier_frac, num_train, num_test)

    print('DONE :)')
