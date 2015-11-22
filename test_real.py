import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

from svdd_primal_sgd import SvddPrimalSGD
from cluster_svdd import ClusterSvdd


def load_data_set(fname, num_data, outlier_frac, train_inds):
    from sklearn.datasets import load_svmlight_file, load_iris
    X, y = load_svmlight_file(fname)
    # X, y = load_svmlight_file("/Users/nicococo/Projects/mnist") # 10
    # X, y = load_svmlight_file("/Users/nicococo/Projects/letter.scale.txt") # 26
    # X, y = load_svmlight_file("/Users/nicococo/Projects/pendigits.txt") # 10  integer features
    # X, y = load_svmlight_file("/Users/nicococo/Projects/dna.scale.txt") # 3 binary
    # X, y = load_svmlight_file("/Users/nicococo/Projects/poker.txt") # 10
    # X, y = load_svmlight_file("/Users/nicococo/Projects/satimage.scale.txt") # 6
    # X, y = load_svmlight_file("/Users/nicococo/Projects/segment.scale.txt") # 7
    # X, y = load_svmlight_file("/Users/nicococo/Projects/news20.scale.txt") # 20, 15,935exm, 62,061feats
    # X, y = load_svmlight_file("/Users/nicococo/Projects/gisette.scale.txt") # 2, 6.000exm, 5.000feats

    # data = load_iris()
    # X = data['data']
    # y = data['target']

    print X.shape
    y -= np.min(y) # classes should start from zero: 0,1,2,3,...

    inds = np.array([], dtype='i')
    for i in range(int(max(y))+1):
    # for i in range(3,7):
        inds = np.append(inds, np.where(y == i)[0])

    print inds.shape
    X = X.toarray()
    X = X[inds, :]
    y = y[inds]

    inds = np.random.permutation(range(y.size))
    X = X[inds[:num_data], :].T
    y = y[inds[:num_data]]

    # X -= np.repeat(np.mean(X[:, train_inds], axis=1)[:, np.newaxis], num_data, axis=1)
    # X /= np.repeat(np.var(X[:, train_inds], axis=1)[:, np.newaxis], num_data, axis=1)
    # X /= np.max(np.abs(X[:, train_inds]))

    #for i in range(num_data):
    #    X[:, i] /= np.linalg.norm(X[:, i], ord=2)

    anoms = int(float(num_data)*outlier_frac)
    # ainds = np.where(y == 9)[0]
    # X[:, ainds[:anoms]] *= 10.
    # y[ainds[:anoms]] = -1
    # X[0:5, :anoms] *= 10.
    X[:, :anoms] = np.random.rand(X.shape[0], anoms)*2.-1.
    y[:anoms] = -1
    # X[:, :anoms] -= np.repeat(np.mean(X[:, :anoms], axis=1)[:, np.newaxis], anoms, axis=1)
    #X[:, :anoms] /= np.max(np.abs(X[:, :anoms]))

    # plt.figure(1)
    # sinds = np.argsort(y)
    # K = X[:, sinds].T.dot(X[:, sinds])
    # plt.imshow(K)
    # plt.show()

    print np.unique(y)
    return X, y


def evaluate(res_filename, dataset, nus, ks, outlier_frac, reps, num_train, num_val, num_test):
    train = np.array(range(num_train-num_val), dtype='i')
    val = np.array(range(num_train-num_val, num_train), dtype='i')
    test = np.array(range(num_train, num_train + num_test), dtype='i')

    aris = np.zeros((reps, len(nus), len(ks)))
    aucs = np.zeros((reps, len(nus), len(ks)))

    val_aris = np.zeros((reps, len(nus), len(ks)))
    val_aucs = np.zeros((reps, len(nus), len(ks)))

    for n in range(reps):
        # generate new gaussians
        # data, y = generate_data(num_train + num_test, outlier_frac=outlier_frac)
        inds = np.random.permutation(range(num_test + num_train))
        data, y = load_data_set(dataset, num_train + num_test, outlier_frac, inds[:num_train])
        data = data[:, inds]
        y = y[inds]
        for k in range(len(ks)):
            # fix the initialization for all methods
            membership = np.random.randint(0, ks[k], y.size)
            for i in range(len(nus)):
                svdds = list()
                for l in range(ks[k]):
                    svdds.append(SvddPrimalSGD(nus[i]))
                    #svdds.append(SvddDualQP('rbf', 0.8, nus[i]))
                svdd = ClusterSvdd(svdds)
                svdd.fit(data[:, train].copy(), init_membership=membership[train])
                # test error
                scores, classes = svdd.predict(data[:, test].copy())
                # evaluate clustering abilities
                inds = np.where((y[test] >= 0))[0]
                aris[n, i, k] = metrics.cluster.adjusted_rand_score(y[test[inds]], classes[inds])
                # ...and anomaly detection accuracy
                fpr, tpr, _ = metrics.roc_curve(np.array(y[test]<0., dtype='i'), scores, pos_label=1)
                aucs[n, i, k] = metrics.auc(fpr, tpr)

                # validation error
                scores, classes = svdd.predict(data[:, val].copy())
                # evaluate clustering abilities
                inds = np.where((y[val] >= 0))[0]
                val_aris[n, i, k] = metrics.cluster.adjusted_rand_score(y[val[inds]], classes[inds])
                # ...and anomaly detection accuracy
                fpr, tpr, _ = metrics.roc_curve(np.array(y[val]<0., dtype='i'), scores, pos_label=1)
                val_aucs[n, i, k] = metrics.auc(fpr, tpr)

    print '---------------------------------------------------'
    maris = np.mean(aris, axis=0)
    saris = np.std(aris, axis=0)
    print '(Test) ARI:'
    print np.mean(aris, axis=0)
    print np.std(aris, axis=0)

    val_maris = np.mean(val_aris, axis=0)
    val_saris = np.std(val_aris, axis=0)
    print '(Val) ARI:'
    print val_maris
    print val_saris

    print '---------------------------------------------------'
    maucs = np.mean(aucs, axis=0)
    saucs = np.std(aucs, axis=0)
    print '(Test) AUC:'
    print np.mean(aucs, axis=0)
    print np.std(aucs, axis=0)

    val_maucs = np.mean(val_aucs, axis=0)
    val_saucs = np.std(val_aucs, axis=0)
    print '(Val) AUC:'
    print val_maucs
    print val_saucs
    print '---------------------------------------------------'

    res = np.zeros(4)
    res_stds = np.zeros(4)

    # best svdd result (assume col 0 is k=1)
    svdd_ind = np.argmax(val_maucs[:, 0])
    print 'SVDD best AUC={0}'.format(maucs[svdd_ind, 0])
    csvdd_ind = np.argmax(val_maucs)
    i1, i2 = np.unravel_index(csvdd_ind, maucs.shape)
    print 'ClusterSVDD best AUC={0}'.format(maucs[i1, i2])
    res[0] = maucs[svdd_ind, 0]
    res_stds[0] = saucs[svdd_ind, 0]
    res[1] = maucs[i1, i2]
    res_stds[1] = saucs[i1, i2]

    # best svdd result (assume col 0 is k=1)
    km_ind = np.argmax(val_maris[0, :])
    print 'k-means best ARI={0}'.format(maris[0, km_ind])
    csvdd_ind = np.argmax(val_maris)
    i1, i2 = np.unravel_index(csvdd_ind, maris.shape)
    print 'ClusterSVDD best ARI={0}'.format(maris[i1, i2])
    res[2] = maris[0, km_ind]
    res_stds[2] = saris[0, km_ind]
    res[3] = maris[i1, i2]
    res_stds[3] = saris[i1, i2]
    print '---------------------------------------------------'

    return res, res_stds

if __name__ == '__main__':
    nus = (np.arange(1, 21)/20.)
    ks = [2, 3, 4]

    nus = [1.0, 0.9, 0.05]
    nus = [1.0, 0.95, 0.9, 0.5, 0.1]
    nus = [1.0, 0.95, 0.9, 0.5, 0.1, 0.01]
    # ks = [1, 5, 7, 10, 14] # segment
    ks = [1, 3, 6, 9] # satimage

    outlier_fracs = [0.0, 0.02, 0.05, 0.1, 0.15]  # fraction of uniform noise in the generated data
    reps = 10  # number of repetitions for performance measures
    # num_train = 1155
    # num_test = 1155
    # num_val = 250

    num_train = 2217
    num_test = 2218
    num_val = 400

    # dataset_name = "../segment.scale.txt" # 7c
    dataset_name = "../satimage.scale.txt" # 6c
    res_filename = 'res_real_{0}_{1}.npz'.format(reps, dataset_name[3:])

    # res: 0:AUC-SVDD, 1:AUC-CSVDD, 2:ARI-KMEANS, 3:ARI-CSVDD
    res = np.zeros((len(outlier_fracs), 4))
    res_stds = np.zeros((len(outlier_fracs), 4))
    for i in range(len(outlier_fracs)):
        res[i, :], res_stds[i, :] = evaluate(res_filename, dataset_name, \
                                             nus, ks, outlier_fracs[i], reps, num_train, num_val, num_test)

    np.savez(res_filename, dataset=dataset_name, res=res, res_stds=res_stds, \
            outlier_fracs=outlier_fracs, ntrain=num_train, nval=num_val, ntest=num_test, reps=reps, nus=nus, ks=ks)

    print '=========================================='
    for i in range(len(outlier_fracs)):
        line = '{0}\\%'.format(int(outlier_fracs[i]*100.))
        for j in range(4):
            line += ' & {0:1.2f}/{1:1.2f}'.format(res[i, j], res_stds[i, j])
        line += '  \\\\'
        print line
    print '=========================================='

    print('DONE :)')
