import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import cvxpy as cp
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from fastdist import fastdist
import mosek
import numba as nb
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms,stable_cumsum
from time import time
from sklearn.cluster._k_means import _labels_inertia


class KMedians(object):
    def __init__(self, k, power=1.0, itermax=100):
        self.k = k
        self.power = power
        self.itermax = itermax
        self.labels_ = None
        self.centers_ = None

    def updateCenterOfCluster(self, cluster, i):
        if self.power == 1.0:
            x = cp.Variable(cluster.shape[1])
            x = cp.Variable(shape=(cluster.shape[1],))
            loss = cp.sum([cp.norm(p - x) for p in cluster])
            cost = cp.Minimize(loss)
            prob = cp.Problem(cost)
            mosek_params = {
                'MSK_DPAR_INTPNT_TOL_PFEAS': 0.3
            }
            prob.solve(solver=cp.MOSEK, mosek_params=mosek_params)
            self.centers_[i] = x.value
        else:
            self.centers_[i] = np.mean(cluster, axis=0)

    def fit(self, P, initial_centers=None):
        if initial_centers is None:
            initial_centers = np.array(kmeans_plusplus_initializer(P, self.k).initialize())
        self.centers_ = initial_centers
        for i in range(self.itermax):
            dists = cdist(P, self.centers_, metric='euclidean')
            self.labels_ = np.argmin(dists,axis=1)
            for i in range(self.k):
                self.updateCenterOfCluster(P[np.where(self.labels_ == i)[0]], i)


@nb.njit(fastmath=True, parallel=True)
def eucl_naive(A, B):
    assert A.shape[1] == B.shape[1]
    C = np.empty((A.shape[0], B.shape[0]), A.dtype)

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            acc = init_val
            for k in range(A.shape[1]):
                acc += (A[i, k] - B[j, k]) ** 2
            C[i, j] = np.sqrt(acc)
    return C


@nb.njit(parallel=True)
def eucl_opt(A, B):
    assert A.shape[1] == B.shape[1]
    C = np.empty((A.shape[0], B.shape[0]), A.dtype)
    I_BLK = 64
    J_BLK = 64

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    # Blocking and partial unrolling
    # Beneficial if the second dimension is large -> computationally bound problem
    #
    for ii in nb.prange(A.shape[0] // I_BLK):
        for jj in range(B.shape[0] // J_BLK):
            for i in range(I_BLK // 4):
                for j in range(J_BLK // 2):
                    acc_0 = init_val
                    acc_1 = init_val
                    acc_2 = init_val
                    acc_3 = init_val
                    acc_4 = init_val
                    acc_5 = init_val
                    acc_6 = init_val
                    acc_7 = init_val
                    for k in range(A.shape[1]):
                        acc_0 += (A[ii * I_BLK + i * 4 + 0, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_1 += (A[ii * I_BLK + i * 4 + 0, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_2 += (A[ii * I_BLK + i * 4 + 1, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_3 += (A[ii * I_BLK + i * 4 + 1, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_4 += (A[ii * I_BLK + i * 4 + 2, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_5 += (A[ii * I_BLK + i * 4 + 2, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_6 += (A[ii * I_BLK + i * 4 + 3, k] - B[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_7 += (A[ii * I_BLK + i * 4 + 3, k] - B[jj * J_BLK + j * 2 + 1, k]) ** 2
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_0)
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_1)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_2)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_3)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_4)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_5)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_6)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_7)
        # Remainder j
        for i in range(I_BLK):
            for j in range((B.shape[0] // J_BLK) * J_BLK, B.shape[0]):
                acc_0 = init_val
                for k in range(A.shape[1]):
                    acc_0 += (A[ii * I_BLK + i, k] - B[j, k]) ** 2
                C[ii * I_BLK + i, j] = np.sqrt(acc_0)

    # Remainder i
    for i in range((A.shape[0] // I_BLK) * I_BLK, A.shape[0]):
        for j in range(B.shape[0]):
            acc_0 = init_val
            for k in range(A.shape[1]):
                acc_0 += (A[i, k] - B[j, k]) ** 2
            C[i, j] = np.sqrt(acc_0)

    return C


def init_kmeans(P, k, squared=False):
    C = np.empty((k, P.shape[1]))
    idx = np.random.randint(low=0, high=P.shape[0])
    C[0] = P[idx]
    indices = list(range(P.shape[0]))

    for i in range(1,k):
        if i == 1:
            # dists = cdist(P, C[:i]).flatten()
            # dists = euclidean_distances(P, C[:i]).flatten()
            # dists = fastdist.vector_to_matrix_distance(C[:i,:].flatten(), P, fastdist.euclidean, "euclidean")
            dists = eucl_opt(P, C[:i,:]).flatten()
        else:
            # dists = np.min(cdist(P, C[:i]), axis=1).flatten()
            # dists = np.min(euclidean_distances(P, C[:i]), axis=1).flatten()
            # dists = np.min(fastdist.matrix_to_matrix_distance(P,C[:i, :], fastdist.euclidean, "euclidean"), axis=1).flatten()
            dists = np.min(eucl_opt(P, C[:i, :]), axis=1).flatten()
        if squared:
            dists = dists ** 2

        prob = dists / np.sum(dists)

        C[i] = P[int(np.random.choice(indices, size=1, p=prob))]

    # labels = np.argmin(cdist(P, C), axis=1)
    # labels = np.argmin(euclidean_distances(P, C), axis=1)
    # labels = np.argmin(fastdist.matrix_to_matrix_distance(P,C, fastdist.euclidean, "euclidean"), axis=1)
    labels = np.argmin(eucl_opt(P,C), axis=1)
    return C, labels


def init_kmeans_fast(P, k, squared=False):
    x_norms = row_norms(P, squared=True)
    n_samples, n_features = P.shape

    centers = np.empty((k, n_features), dtype=P.dtype)

    # Pick first center randomly and track index of point
    center_id = np.random.randint(n_samples)
    indices = np.full(k, -1, dtype=int)
    if sp.issparse(P):
        centers[0] = P[center_id].toarray()
    else:
        centers[0] = P[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], P, Y_norm_squared=x_norms,
        squared=squared)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, k):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(2) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            P[candidate_ids], P, Y_norm_squared=x_norms, squared=squared)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(P):
            centers[c] = P[best_candidate].toarray()
        else:
            centers[c] = P[best_candidate]
        indices[c] = best_candidate

    return centers, _labels_inertia(P, np.ones((P.shape[0], )), x_norms ** (2.0 if not squared else 1.0), centers)[0]





if __name__ == '__main__':
    P = np.random.rand(1000,7)
    start = time()
    medians = init_kmeans(P, k=7, squared=False)
    print('Time is {}'.format(time() - start))
    print(medians)
    start = time()
    medians = init_kmeans_fast(P, k=7, squared=False)
    print('Time is {}'.format(time() - start))
    print(medians)

    # medians.fit(P)