import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

from svdd_primal_sgd import SvddPrimalSGD
from cluster_svdd import ClusterSvdd


def generate_seqs(lens, block_len, dims=2, proportion=0.5):
    seqs = 1.0*np.random.randn(dims, lens)
    states = np.zeros(lens, dtype='i')
    y = 0
    prob = np.random.uniform()
    if prob < proportion:
        y = 1
        start = np.random.randint(low=0, high=lens-block_len+1)
        states[start:start+block_len] = 1
        seqs[:, start:start+block_len] = seqs[:, start:start+block_len]+2.5
    return seqs, states, y


def generate_data(datapoints, outlier_frac=0.1, dims=2, plot=True):
    X = []
    S = []
    y = np.ones(datapoints, dtype='i')
    idx_1 = -1
    idx_0 = -1
    idx_anom = -1
    for i in range(datapoints):
        exm, states, y[i] = generate_seqs(500, 200)
        prob = np.random.uniform()
        if prob < outlier_frac:
            idx_anom = i
            exm *= np.random.uniform(low=-0.1,high=+0.1, size=(dims, 500))
            exm *= np.exp(10.0*exm)
        if y[i] == 0:
            idx_0 = i
        if y[i] == 1:
            idx_1 = i
        X.append(exm)
        S.append(states)

    if plot:
        plt.figure(1)
        plt.subplot(1, 3, 1)
        plt.plot(range(500), X[idx_0][0,:], '-r', alpha=0.7)
        plt.plot(range(500), S[idx_0], '-b', linewidth=2.0)
        plt.ylim((-6.5, 6.5))
        plt.title('Class 0', fontsize=14)
        plt.xlabel('Sequence index', fontsize=14)
        plt.ylabel('Feature 0', fontsize=14)

        plt.subplot(1, 3, 2)
        plt.plot(range(500), X[idx_1][0,:], '-r', alpha=0.7)
        plt.plot(range(500), S[idx_1], '-b', linewidth=2.0)
        plt.ylim((-6.5, 6.5))
        plt.title('Class 1', fontsize=14)
        plt.xlabel('Sequence index', fontsize=14)
        plt.ylabel('Feature 0', fontsize=14)

        plt.subplot(1, 3, 3)
        plt.plot(range(500), X[idx_anom][0,:], '-r', alpha=0.7)
        plt.plot(range(500), S[idx_anom], '-b', linewidth=2.0)
        plt.ylim((-6.5, 6.5))
        plt.title('Anomalous Data', fontsize=14)
        plt.xlabel('Sequence index', fontsize=14)
        plt.ylabel('Feature 0', fontsize=14)

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
        # phi[:, n] = argmax(sol, data_seqs[n])
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
            phis[:, n], states[n] = argmax(sol, X[inds[n]])
            phis[:, n] /= np.linalg.norm(phis[:, n], ord=2)

        scores = csvdd.svdds[k].predict(phis)
        minds = np.where(scores <= min_scores)[0]
        pred_phis[:, minds] = phis[:, minds]
        for i in minds:
            pred_states[i] = states[i]

    return pred_phis, true_states, pred_states


def hamming_loss(y_true, y_pred):
    N = len(y_pred)
    loss = 0.0
    for i in range(N):
        loss += float(np.sum(y_true[i] != y_pred[i])) / float(y_pred[i].size)
    return loss / float(N)


def calc_emission_matrix(sol, X):
    T = X.shape[1]
    N = 2
    F = X.shape[0]
    em = np.zeros((N, T))
    for t in xrange(T):
        for s in xrange(N):
            for f in xrange(F):
                em[s, t] += sol[N*N + s*F + f] * X[f, t]
    return em


def get_transition_matrix(sol):
    N = 2
    A = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            A[i, j] = sol[i*N+j]
    return A


def argmax(sol, X):
    # if labels are present, then argmax will solve
    # the loss augmented programm
    T = X.shape[1]
    N = 2

    # get transition matrix from current solution
    A = get_transition_matrix(sol)
    # calc emission matrix from current solution, data points and
    em = calc_emission_matrix(sol, X)

    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype='i')
    # initialization
    for i in xrange(N):
        # use equal start probs for each state
        delta[i, 0] = 0. + em[i, 0]

    # recursion
    for t in xrange(1,T):
        for i in xrange(N):
            foo = delta[:, t-1] + A[:, i] + em[i, t]
            psi[i, t] = np.argmax(foo)
            delta[i, t] = foo[psi[i, t]]

    states = np.zeros(T, dtype='i')
    states[T-1] = np.argmax(delta[:, T-1])

    for t in reversed(xrange(1, T)):
        states[t-1] = psi[states[t], t]
    return get_joint_feature_map(X, states), states


def get_joint_feature_map(X, y):
    T = y.size
    N = 2
    F = X.shape[0]
    jfm = np.zeros(N*N + N*F)

    # transition part
    for i in range(N):
        _, inds = np.where([y[1:T]==i])
        for j in range(N):
            _, indsj = np.where([y[inds]==j])
            jfm[j*N+i] = float(indsj.size)
    # emission parts
    for t in range(T):
        for f in range(F):
            jfm[y[t]*F + f + N*N] += X[f, t]
    return jfm


def plot_results(res_filename):
    foo = np.load(res_filename)
    maris = foo['maris']
    saris = foo['saris']
    #maris = foo['mloss']
    #saris = foo['sloss']
    nus = foo['nus']
    reps = foo['reps']

    plt.figure(1)
    cols = np.random.rand(maris.shape[1], 3)
    fmts = ['-->', '--o', '--D', '--s', '--H']
    for i in range(maris.shape[1]):
        plt.errorbar(nus, maris[:, i], saris[:, i]/np.sqrt(reps), fmt=fmts[i], color=cols[i, :], \
                     ecolor=cols[i, :], linewidth=2.0, elinewidth=1.0, alpha=0.8)
    for i in range(maris.shape[1]):
        plt.errorbar(nus[-1], maris[-1, i], saris[-1, i]/np.sqrt(reps), \
                     color='r', ecolor='r', fmt=fmts[i][-1], markersize=6, linewidth=4.0, elinewidth=4.0, alpha=0.7)

    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0 = k-means'], fontsize=14)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=14)
    plt.grid()
    plt.xlabel(r'regularization parameter $\nu$', fontsize=14)
    plt.ylabel(r'Adjusted Rand Index (ARI)', fontsize=14)
    names = list()
    for i in range(maris.shape[1]):
        names.append('ClusterSVDD ($k$={0})'.format(ks[i]))
    # for i in range(maris.shape[1]):
    #     names.append('$k$-means ($k$={0})'.format(ks[i]))
    plt.legend(names, loc=4, fontsize=14)

    plt.show()


def evaluate(res_filename, nus, ks, outlier_frac, reps, num_train, num_test):
    train = np.array(range(num_train), dtype='i')
    test = np.array(range(num_train, num_train + num_test), dtype='i')

    aris = np.zeros((reps, len(nus), len(ks)))
    loss = np.zeros((reps, len(nus), len(ks)))
    for n in range(reps):
        # generate new gaussians
        X, S, y = generate_data(num_train + num_test, outlier_frac=outlier_frac, dims=2, plot=False)
        inds = np.random.permutation(range(num_test + num_train))
        data = preprocess_training_data(X, S, inds[:num_train])
        data = data[:, inds]
        y = y[inds]
        for k in range(len(ks)):
            # fix the initialization for all methods
            membership = np.random.randint(0, ks[k], y.size)
            for i in range(len(nus)):
                svdds = list()
                for l in range(ks[k]):
                    svdds.append(SvddPrimalSGD(nus[i]))
                svdd = ClusterSvdd(svdds)
                svdd.fit(data[:, train], init_membership=membership[train])

                pred_phis, true_states, pred_states = preprocess_test_data(svdd, X, S, inds[num_train:])
                _, classes = svdd.predict(pred_phis)

                # evaluate clustering abilities
                ninds = np.where(y[test] >= 0)[0]
                aris[n, i, k] = metrics.cluster.adjusted_rand_score(y[test[ninds]], classes[ninds])
                # evaluate structured prediction accuracy
                loss[n, i, k] = hamming_loss(true_states, pred_states)
                print loss[n, i, k]

    maris = np.mean(aris, axis=0)
    saris = np.std(aris, axis=0)
    print np.mean(aris, axis=0)
    print np.std(aris, axis=0)

    mloss = np.mean(loss, axis=0)
    sloss = np.std(loss, axis=0)
    print np.mean(loss, axis=0)
    print np.std(loss, axis=0)

    np.savez(res_filename, maris=maris, saris=saris, mloss=mloss, sloss=sloss,
                outlier_frac=outlier_frac, ntrain=num_train, ntest=num_test, reps=reps, nus=nus)


if __name__ == '__main__':
    nus = (np.arange(1, 11)/10.)
    ks = [1, 2, 3, 4]
    #nus = [1.0, 0.5]
    #ks = [1, 2]

    outlier_frac = 0.02  # fraction of uniform noise in the generated data
    reps = 10  # number of repetitions for performance measures
    num_train = 2000
    num_test = 500

    do_plot = False
    do_evaluation = True

    res_filename = 'res_struct_{0}_{1}_{2}.npz'.format(reps, len(ks), len(nus))

    if do_evaluation:
        evaluate(res_filename, nus, ks, outlier_frac, reps, num_train, num_test)
    if do_plot:
        data, states, y = generate_data(num_train + num_test, outlier_frac=outlier_frac, dims=2, plot=True)
        plot_results(res_filename)

    print('DONE :)')
