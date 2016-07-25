import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import time as time

from numba import autojit

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.cluster_svdd import ClusterSvdd


def generate_seqs(lens, block_len, cluster=3, dims=3):
    classes = np.random.randint(0, cluster)
    seqs = 1.0*np.random.randn(dims, lens)
    states = np.zeros(lens, dtype='i')
    y = classes
    start = np.random.randint(low=0, high=lens-block_len+1)
    states[start:start+block_len] = 1
    # seqs[0, start:start+block_len] = seqs[0, start:start+block_len]+0.5*classes-2.0*float(classes==0)
    seqs[classes, start:start+block_len] = seqs[0, start:start+block_len]+1.0
    return seqs, states, y


def generate_data(datapoints, cluster=3, outlier_frac=0.1, dims=3, plot=True):
    lens = 500
    X = []
    S = []
    y = np.ones(datapoints, dtype='i')
    idx = np.zeros(cluster, dtype='i')
    idx_anom = -1
    for i in range(datapoints):
        exm, states, y[i] = generate_seqs(lens, 250, cluster=cluster, dims=dims)
        prob = np.random.uniform()
        if prob < outlier_frac:
            idx_anom = i
            exm *= np.random.uniform(low=-0.1,high=+0.1, size=(dims, lens))
            exm *= np.exp(10.0*exm)
            y[i] = -1
        else:
            idx[y[i]] = i
        X.append(exm)
        S.append(states)

    if plot:
        plt.figure(1)

        for d in range(dims):
            for i in range(cluster):
                plt.subplot(1, cluster+1, i+1)
                plt.plot(range(lens), X[idx[i]][d, :]+d*6., '-r', alpha=0.7)
                plt.ylim((-5.0, 20.))
                plt.yticks([0.0])
                xinds = np.where(S[idx[i]]==1)[0]
                plt.fill_between(xinds, -5, 20, color=[0.3, 0.3, 0.3], alpha=0.25)
                plt.title('Class {0}'.format(i), fontsize=14)
                plt.xlabel('Sequence index', fontsize=14)
                plt.ylabel('Feature 0  Feature 1  Feature 2', fontsize=14)

        plt.subplot(1, cluster+1, cluster+1)
        for d in range(dims):
            plt.plot(range(lens), X[idx_anom][d, :]+d*6., '-r', alpha=0.7)
        plt.yticks([0.0])
        plt.ylim((-5., 20.  ))
        plt.title('Anomalous Data', fontsize=14)
        plt.xlabel('Sequence index', fontsize=14)
        plt.ylabel('Feature 0  Feature 1  Feature 2', fontsize=14)

        plt.show()
    return X, S, y


def preprocess_training_data(data_seqs, state_seqs, train_inds):
    # estimate the transition and emission matrix given the training
    # data only. Number of states is 2.
    N = len(data_seqs)
    F, _ = data_seqs[0].shape
    phi = np.zeros((2*2 + F*2, N))
    for n in train_inds:
        phi[:, n] = get_joint_feature_map(data_seqs[n], state_seqs[n])
        phi[:, n] /= np.linalg.norm(phi[:, n], ord=2)
    return phi


def preprocess_test_data(csvdd, X, S, inds):
    # 1. for all i,k:  y_i,k = argmax_y <c_k, psi(x_i, y)>
    # 2. for all i: calculate membership z_i = argmin_k ||c_k - psi(x_i, y_i,k)||^2 - R_k
    # 3. for all i: hamming loss delta(y_i, y_i,z_i)
    N = inds.size
    F, _ = X[0].shape

    pred_phis = np.zeros((2*2 + F*2, N))
    true_states = []
    pred_states = []
    states = []
    for n in range(N):
        states.append(S[inds[n]])
        true_states.append(S[inds[n]])
        pred_states.append(S[inds[n]])

    min_scores = 1e12*np.ones(N, dtype='d')
    for k in range(csvdd.clusters):
        phis = np.zeros((2*2 + F*2, N))
        for n in range(N):
            sol = csvdd.svdds[k].c
            states[n] = argmax(sol, X[inds[n]])
            phis[:, n] = get_joint_feature_map(X[inds[n]], states[n])
            # states[n] = true_states[n]
            phis[:, n] /= np.linalg.norm(phis[:, n], ord=2)

        scores = csvdd.svdds[k].predict(phis)
        minds = np.where(scores <= min_scores)[0]
        pred_phis[:, minds] = phis[:, minds]
        min_scores[minds] = scores[minds]
        for i in minds:
            pred_states[i] = states[i]

    return pred_phis, true_states, pred_states

def hamming_loss(y_true, y_pred):
    N = len(y_pred)
    loss = 0.0
    for i in range(N):
        loss += float(np.sum(y_true[i] != y_pred[i])) / float(y_pred[i].size)
    return loss / float(N)


@autojit(nopython=True)
def argmax(sol, X):
    # if labels are present, then argmax will solve
    # the loss augmented programm
    T = X.shape[1]
    N = 2

    # get transition matrix from current solution
    A = np.zeros((N, N), dtype=np.double)
    for i in range(N):
        for j in range(N):
            A[i, j] = sol[i*N+j]

    # calc emission matrix from current solution, data points and
    F = X.shape[0]
    em = np.zeros((N, T))
    for t in range(T):
        for s in range(N):
            for f in xrange(F):
                em[s, t] += sol[N*N + s*F + f] * X[f, t]

    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=np.int8)
    # initialization
    for i in xrange(N):
        # use equal start probs for each state
        delta[i, 0] = 0. + em[i, 0]

    # recursion
    for t in range(1, T):
        for i in range(N):
            foo_argmax = 0
            foo_max = -1e16
            for l in range(N):
                foo = delta[l, t-1] + A[l, i] + em[i, t]
                if foo > foo_max:
                    foo_max = foo
                    foo_argmax = l
            psi[i, t] = foo_argmax
            delta[i, t] = foo_max

    states = np.zeros(T, dtype=np.int8)
    states[T-1] = np.argmax(delta[:, T-1])

    # for t in reversed(xrange(1, T)):
    for t in range(T-1, 0, -1):
        states[t-1] = psi[states[t], t]
    return states


@autojit(nopython=True)
def get_joint_feature_map(X, y):
    N = 2
    T = y.size
    F = X.shape[0]
    jfm = np.zeros(N*N + N*F)
    # transition part
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                if y[t]==i and y[t+1]==j:
                    jfm[j*N+i] += 1
    # emission parts
    for t in range(T):
        for f in range(F):
            jfm[y[t]*F + f + N*N] += X[f, t]
    return jfm


def plot_results(res_filename):
    data, states, y = generate_data(1000, cluster=3, outlier_frac=0.05, dims=3, plot=False)

    foo = np.load(res_filename)
    maris = foo['maris']
    saris = foo['saris']
    mloss = foo['mloss']
    sloss = foo['sloss']
    nus = foo['nus']
    reps = foo['reps']

    res = np.zeros((len(nus), 4))
    res_stds = np.zeros((len(nus), 4))

    # svdd
    res[:, 0] = mloss[:, 0]
    res_stds[:, 0] = sloss[:, 0]
    # csvdd
    res[:, 1] = mloss[:, 1]
    res_stds[:, 1] = sloss[:, 1]

    # kmeans
    res[0, 2] = maris[0, 1]
    res_stds[0, 2] = saris[0, 1]
    # csvdd
    res[:, 3] = maris[:, 1]
    res_stds[:, 3] = saris[:, 1]

    print '=========================================='
    for i in range(len(nus)):
        line = '{0:1.2f}\\%'.format(nus[i])
        for j in range(4):
            line += ' & {0:1.2f}/{1:1.2f}'.format(res[i, j], res_stds[i, j])
        line += '  \\\\'
        print line
    print '=========================================='


def evaluate(res_filename, nus, ks, outlier_frac, reps, num_train, num_test):
    train = np.array(range(num_train), dtype='i')
    test = np.array(range(num_train, num_train + num_test), dtype='i')

    aris = np.zeros((reps, len(nus), len(ks)))
    loss = np.zeros((reps, len(nus), len(ks)))
    for n in range(reps):
        # generate new gaussians
        X, S, y = generate_data(num_train + num_test, cluster=3, outlier_frac=outlier_frac, dims=3, plot=False)
        inds = np.random.permutation(range(num_test + num_train))
        data = preprocess_training_data(X, S, inds[:num_train])
        data = data[:, inds]
        y = y[inds]
        print data
        print y
        for k in range(len(ks)):
            # fix the initialization for all methods
            membership = np.random.randint(0, ks[k], y.size)
            for i in range(len(nus)):
                svdds = list()
                for l in range(ks[k]):
                    svdds.append(SvddPrimalSGD(nus[i]))
                svdd = ClusterSvdd(svdds)
                svdd.fit(data[:, train], init_membership=membership[train])

                stime = time.time()
                pred_phis, true_states, pred_states = preprocess_test_data(svdd, X, S, inds[num_train:])
                _, classes = svdd.predict(pred_phis)
                print '---------------- TIME'
                print time.time()-stime
                print '----------------'

                # evaluate clustering abilities
                ninds = np.where(y[test] >= 0)[0]
                aris[n, i, k] = metrics.cluster.adjusted_rand_score(y[test[ninds]], classes[ninds])
                # evaluate structured prediction accuracy
                loss[n, i, k] = hamming_loss(true_states, pred_states)
                print loss[n, i, k]

    maris = np.mean(aris, axis=0)
    saris = np.std(aris, axis=0)
    print 'ARI'
    print np.mean(aris, axis=0)
    print np.std(aris, axis=0)

    mloss = np.mean(loss, axis=0)
    sloss = np.std(loss, axis=0)
    print 'Normalized Hamming Distance'
    print np.mean(loss, axis=0)
    print np.std(loss, axis=0)

    np.savez(res_filename, maris=maris, saris=saris, mloss=mloss, sloss=sloss,
                outlier_frac=outlier_frac, ntrain=num_train, ntest=num_test, reps=reps, nus=nus)


if __name__ == '__main__':
    nus = [1.0, 0.9, 0.5, 0.1, 0.01]
    ks = [1, 3]

    outlier_frac = 0.05  # fraction of uniform noise in the generated data
    reps = 10  # number of repetitions for performance measures
    num_train = 2000
    num_test = 500

    do_plot = True
    do_evaluation = True

    res_filename = 'res_struct_{0}_{1}_{2}.npz'.format(reps, len(ks), len(nus))

    if do_evaluation:
        evaluate(res_filename, nus, ks, outlier_frac, reps, num_train, num_test)
    if do_plot:
        # data, states, y = generate_data(num_train + num_test, outlier_frac=outlier_frac, dims=2, plot=True)
        plot_results(res_filename)

    print('DONE :)')
