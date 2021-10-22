import sklearn as skl
from sklearn import svm, linear_model, neural_network
from scipy.linalg import norm
import numpy as np
import math
import json, time, copy
from multiprocessing import Lock
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
import pickle, os
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import accuracy_score
# import PegasosModel as Pegasos
import matplotlib
# from sklearn.cluster import KMeans, MiniBatchKMeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from FKmeans import init_kmeans_fast


from sklearn import random_projection
# import KMeansPP
matplotlib.use("TkAgg")


TOL = 1e-2 # was 1e-6 originally
MAX_ITER = -1
FAST = True  # whether to prioritize fast computation time or not

MAX_ITER_KMEANS = 1


class MLProbs(object):
    mutex = Lock()

    models = {
        'linear-svm': (lambda a,b,c,d : svm.LinearSVC(C=a, \
                        fit_intercept=b, dual=c, loss=d, tol=TOL, max_iter=MAX_ITER)),

        'sgd': (lambda a, b : SGDClassifier(loss="hinge", penalty="l2", alpha=a, fit_intercept=b, \
                                            n_iter=1000)),

        'svm': (lambda a,e, g, tol=TOL: svm.SVC(C=a, \
                 kernel=e, tol=tol, gamma=g)),

        'logistic': (lambda a,b,c,d: linear_model.LogisticRegression(C=a, \
                       fit_intercept=b, dual=c, solver=d)),

        'neural': (lambda a, b, c, d: neural_network.MLPClassifier(solver=a, \
                    alpha=b, hidden_layer_sizes=c, random_state=d))
    }

    def loadModelFromJson(self, jsonOBJ):
        params = json.loads(jsonOBJ)
        self.params = params

        # MAKE SURE YOUR C VALUE IS WITH RESPECT TO THE CORRECT N (NOT JUST LENGTH OF TRAINING DATA!!!!)
        N = len(self.params['X'])

        # self = self.updateParams('type', 'linear-svm')
        self = self.updateParams('type', 'svm')
        self = self.updateParams('solver', 'liblinear')
        self = self.updateParams('kernel', 'linear')
        self = self.updateParams('fit_intercept', True)
        self = self.updateParams('dual', True)
        self = self.updateParams('loss', 'hinge')


        # Neural Network
        self = self.updateParams('random_state', 1)
        self = self.updateParams('alpha', 1e-5)
        self = self.updateParams('hidden_layer_sizes', (5, 2))

        self.train = {'X': np.array(params['X']), 'Y': np.array(params['Y'])}
        self.test = {'X': np.array(params['XTest']), 'Y': np.array(params['YTest'])}

        self.updateParams('gamma', 1.0 / np.ma.size(self.train['X'], 1))

        if 'svm' in self.params['type'] or 'sgd' in self.params['type']:
            self.evaluateVal = lambda w,b: self.evaluateSVMs(w,b)
        else:
            self.evaluateVal = lambda w, b: self.evaluateLogistic(w,b)

        model = self.models[self.params['type']]
        if self.params['type'] == 'linear-svm' :
            self.clf = model(self.params['C'], self.params['fit_intercept'], self.params['dual'], \
                self.params['loss'])
        elif self.params['type'] == 'svm':
            self.clf = model(self.params['C'], self.params['kernel'], self.params['gamma'])
        elif self.params['type'] == 'logistic':
            self.clf = model(self.params['C'], self.params['fit_intercept'], self.params['hidden_layer_sizes'], \
                self.params['solver'])
        elif self.params['type'] == 'sgd':
            alpha = 1.0/(N*self.params['C'])
            self.clf = model(alpha, self.params['fit_intercept'])
        else:
            self.clf = model(self.params['solver'], self.params['alpha'], self.params['dual'], \
                self.params['random_state'])
        self.allDataAccuracy = 0

        self.weights = np.array(self.params['weights'])
        self.sumWeights = np.sum(self.weights)
        self.params['X'] = np.array(self.params['X'])
        self.params['Y'] = np.array(self.params['Y'])

        return self

    def updateParams(self, attribute, value):
        if attribute not in self.params:
            self.params[attribute] = value

        return self

    def __init__(self, params, fileName, normalization, centering, streaming):
        # change
        self = self.loadModelFromJson(params)
        self.our_coreset_time = 0.0

        self._sense_bound_lambda = (lambda x, w, args=None: np.maximum(9 * w / args[0], 2 * w / args[1]) + 13 * w / (4 * args[0]) +
                              125 * (args[0] + args[1]) / (4 * 1) * (w * np.linalg.norm(x, ord=2, axis=1)**2 +
                                                                          w/(args[0] + args[1])))

        # self.FAST = (self.params['X'].shape[0] < 20000)
        # print("Fast: {} (n={})".format(self.FAST, self.params['X'].shape[0]))

        self.FAST = True if streaming else FAST


        PICKLES_DIR = 'pickles'
        fileName = "{}/{}".format(PICKLES_DIR, "{}-n{}-c{}-C{}.pickle".format(fileName.split('.')[0], normalization, centering, int(100.0/self.params['C'])))

        if os.path.isfile(fileName):
            # Load the cached results from here.
            with open(fileName, "rb") as input_file:
                try:
                    info = pickle.load(input_file)
                except:
                    info = pickle.load(input_file, encoding='latin1')
                print(info)
                self.optTime = info['optTime']
                self.optVal = info['optVal']
                self.allDataAccuracy = info['allAccuracy']
            return

        start = time.time()
        print("Fitting {} data points".format(self.params['X'].shape[0]))
        self.sOPT = self.clf.fit(self.params['X'], self.params['Y'])
        self.optVal = 0
        self.optVal = self.evaluateSVMs(self.clf.coef_, self.clf.intercept_)
        self.optW = np.append(self.clf.coef_, float(self.clf.intercept_))

        self.optVals = []
        self.optWs = []
        self.optAcc = []
        self.gamma = 1.0 / np.size(self.train['X'], 1)


        if self.params['classify'] == True:
            self.allDataAccuracy = self.clf.score(self.test['X'], self.test['Y']) * 100.0
            print('Model Accuracy on test data is: {:.3f}'.format(self.allDataAccuracy))

        self.optTime = time.time() - start
        tol = TOL
        params_svm = {"tol": tol}
        self.clf.set_params(**params_svm)
        print('Optimal Training Time: {:.3f}'.format(self.optTime))
        print('Optimal Value: {:.3f}'.format(self.optVal))

        with open(fileName, "wb") as output_file:
            pickle.dump({'optTime': self.optTime, 'optVal': self.optVal, 'allAccuracy': self.allDataAccuracy}, output_file)

    def evaluateSVMsOld(self, w, b=0.0, C = None):
        if C == None:
            C = self.params['C']

        reg = 0.5 * norm(w) ** 2.0
        hinge = np.maximum(0, 1 - np.multiply(self.train['Y'], self.train['X'].dot(w.T).flatten() + b))
        return reg + self.params['C'] * np.sum(hinge)

    def evaluateSVMs(self, w, b=0.0, C = None):
        if C == None:
            C = self.params['C']

        reg = 0.5 * norm(w) ** 2.0
        hinge = np.maximum(0, 1 - np.multiply(self.train['Y'], self.train['X'].dot(w.T).flatten() + b))
        return reg + self.params['C'] * np.sum(hinge)


    def evaluateSvmsPerS(self, X, Y, Weight, w, b=0.0, C = None):
        if C is None:
            C = self.params['C']
        reg = 0.5 * norm(w) ** 2
        hinge = np.maximum(0, 1 - np.multiply(Y, X.dot(w.T).flatten() + b))
        return np.sum(Weight) / self.sumWeights * reg + self.params['C'] * np.sum(np.multiply(Weight[:,np.newaxis].T, hinge))

    def gradient(self, X, Y, Weight, w, b):
        indicator = np.multiply(Y, X.dot(w.T).flatten() + b) < 1
        subGrad = np.multiply(indicator.astype(float)[:,np.newaxis],(-np.multiply(Y[:,np.newaxis],X)))

        subGrad = np.sum(Weight) / self.sumWeights * w + self.params['C'] * np.sum(np.multiply(Weight[:,np.newaxis],subGrad), axis=0)
        subGrad = np.append(subGrad, np.sum(np.multiply(Weight,np.multiply(Y,-self.params['C'] * indicator.astype(float)))))
        return subGrad

    def evaluateRelativeError(self, xTrain, yTrain, weights = None):
        USERDEFINEDOBJFUNC = False

        start = time.time()
        MLProbs.mutex.acquire()
        clf = copy.deepcopy(self.clf)
        n = self.train['X'].shape[0]

        # NEED TO DIVIDE BY SUM OF WEIGHTS FOR REG NO JUST N, SINCE SUM OF WEIGHTS MAY NOT EQUAL N
        Cprime = self.params['C'] * float(np.sum(self.weights) / (np.sum(weights)))

        # Cprime = self.params['C']

        MLProbs.mutex.release()

        if np.unique(yTrain).shape[0] < 2:
            print('No unique labels! Returning 1!')
            return 1,0, 0

        if not USERDEFINEDOBJFUNC:
            params_svm = {"C": Cprime}
            clf.set_params(**params_svm)
            clf.fit(xTrain, yTrain, sample_weight=weights)
            w = clf.coef_
            b = clf.intercept_

        else:
            f = lambda x, xTrain=xTrain, yTrain=yTrain , weights=weights: self.evaluateSvmsPerS(xTrain, yTrain, weights, x[:-1], x[-1])
            g = lambda x, xTrain=xTrain, yTrain=yTrain,weights=weights: self.gradient(xTrain, yTrain, weights, x[:-1], x[-1])
            x0 = np.random.rand(1, xTrain.shape[1] + 1)
            res = minimize(f, x0, jac=g)
            w = res.x[:-1]
            b = res.x[-1]

        timeTaken = time.time() - start
        MLProbs.mutex.acquire()

        val = self.evaluateVal(w, b)
        ret = float(val / self.optVal) - 1.0
        MLProbs.mutex.release()

        return ret, timeTaken


    def evaluatePegasosRelativeError(self, w, b = 0.0):
        val = self.evaluateVal(w, b)
        ret = float(val / self.optVal) - 1.0
        return ret, val

    def predictLabels(self, X, Y, w, b = 0.0):
        yPred = np.zeros(Y.shape)

        for i in range(0,X.shape[0]):
            if (np.dot(X[i,:], w) + b) > 0:
                yPred[i] = 1
            else:
                yPred[i] = -1

        return yPred

    def evaluateOnTest(self, w, b=0.0):
        reg = 0.5 * norm(w) ** 2
        hinge = np.maximum(0, 1 - np.multiply(self.test['Y'], self.test['X'].dot(w.T).flatten() + b))
        return reg + self.params['C'] * np.sum(hinge)


    def evaluateAccuracy(self, xTrain, yTrain, weights = None, clf = None):

        MLProbs.mutex.acquire()
        start = time.time()
        if clf == None:
            clf = copy.deepcopy(self.clf)

        MLProbs.mutex.release()

        Cprime = self.params['C'] * float(np.sum(self.weights) / (np.sum(weights)))
        params_svm = {"C": Cprime}
        clf.set_params(**params_svm)

        clf.fit(xTrain, yTrain, sample_weight=weights)

        MLProbs.mutex.acquire()

        # DO WE HAVE TO PASS IN WEIGHTS HERE OR NOT?
        accuracy = clf.score(self.test['X'], self.test['Y'])

        timeTaken = time.time() - start
        MLProbs.mutex.release()

        return accuracy*100.0, timeTaken

    def evaluateRelativeAccuracy(self, X, Y, weights = None):
        acc, timeTaken = self.evaluateAccuracy(X, Y, weights=weights)

        return 1 - acc / self.allDataAccuracy, timeTaken

    def evaluateRelativeLoss(self, X, Y, weights=None):
        MLProbs.mutex.acquire()
        start = time.time()
        clf = copy.deepcopy(self.clf)

        MLProbs.mutex.release()

        clf.fit(X, Y, sample_weight=weights)
        w = clf.coef_
        b = clf.intercept_

        MLProbs.mutex.acquire()

        val = self.evaluateOnTest(w, b)
        ret = float(val / self.optVal) - 1.0
        timeTaken = time.time() - start
        MLProbs.mutex.release()

        return ret, timeTaken


    def applyHyperParameterTuning(self, X, Y, Cs, weights=None):

        trainLoss = np.zeros((1, len(Cs)))
        accuracyScore = np.zeros((1, len(Cs)))
        timeTaken = np.zeros((1, len(Cs)))
        clf = copy.deepcopy(self.clf)

        for i,C in enumerate(Cs):
            start = time.time()
            n = self.train['X'].shape[0]

            if weights is not None:
                Cprime = C * float(np.sum(self.weights) / (np.sum(weights)))
            else:
                Cprime = C

            # MLProbs.mutex.release()

            params_svm = {"C": Cprime}
            clf.set_params(**params_svm)

            if weights is None:
                clf.fit(X, Y)
            else:
                clf.fit(X, Y, weights)

            accuracyScore[0,i] = clf.score(self.test['X'], self.test['Y']) * 100.0
            w = clf.coef_
            b = clf.intercept_
            if weights is None:
                trainLoss[0,i] = self.evaluateSVMs(w, b, C)
            else:
                trainLoss[0, i] = self.evaluateSvmsPerS(X, Y, weights, w, b, Cprime)
            timeTaken[0,i] = time.time() - start
            print('Iteration: {}, C: {:.3f}, Time Taken: {:.3f}'.format(i, C, timeTaken[0,i]))
        return trainLoss, accuracyScore, timeTaken

    def computeSensitivity(self, P=None, weights=None, use_k_median=False):
        """
        :return: bounded sensitivity vector of n entries
        """

        if P is None:
            X = self.params['X']
            Y = self.params['Y']
        else:
            X = P[:, :-1]
            Y = P[:, -1]

        if weights is None:
            weights = self.weights

        # start time and compute the optimal solution of the SVM problem
        start_time = time.time()

        w_opt, b_opt = self.computeOptimalSolverUsingScipy(X, Y, weights, self.params['C'])

        # compute the optimal value
        opt = self.evaluateSvmsPerS(X, Y, weights, w_opt, b_opt)

        print('Computing OPT in {:.3f} Seconds'.format(time.time() - start_time))

        start_time = time.time()


        # compute the indices of the positive and negative indices
        positive_indices = np.array(np.where(Y == 1)).flatten()
        negative_indices = np.setdiff1d(range(0, np.ma.size(X, 0)), positive_indices)

        positive_points = X[positive_indices, :]
        negative_points = X[negative_indices, :]

        # attain the weights of the positive and negative points by labels
        weights_positive = weights[positive_indices]
        weights_negative = weights[negative_indices]

        # compute the number of positive and negative points by label
        n_positive = np.size(positive_indices)
        n_negative = np.size(negative_indices)

        # compute the sum of the weights of the negative and positive points by labels
        sum_of_pos_weights = np.sum(weights_positive)
        sum_of_neg_weights = np.sum(weights_negative)

        # initialize the sensitvities to 0s and compute the total sum of weights
        n = n_positive + n_negative
        sensitivity = np.zeros((n, 1)).flatten()
        sum_of_weights_all = sum_of_pos_weights + sum_of_neg_weights

        if n_positive <= 1 or n_negative <= 1:
            print('Interesting!')

        # bound the sensitivity of the points with positive labels

        if n_positive > 0:
            sensitivity[positive_indices],dists_pos = \
                self.getAnalyticalBound(positive_points,
                                       opt,
                                       weights_positive,
                                       sum_of_weights_all, use_k_median)


        # bound the sensitivity of the points with the negative labels
        if n_negative > 0:
            sensitivity[negative_indices], dists_neg =  \
                self.getAnalyticalBound(negative_points * (-1.0),
                                       opt,
                                       weights_negative,
                                       sum_of_weights_all, use_k_median)

        # bound the sensitivity by 1 after element-wise multiplication with the weights
        sensitivity = np.multiply(sensitivity, weights.flatten())
        #sensitivity = np.minimum(sensitivity, 1)

        # compute the time needed for bounding the sensitivities
        self.coreset_time = time.time() - start_time
        print('Overall time needed for kmeans: {:.4f}'.format(self.coreset_time))
        print('total distance is {}'.format(dists_neg+dists_pos))

        S = np.sum(sensitivity)
        # print("Sum of sensitivities: {} ({:.1f}% of n={})".format(int(S), S/n*100.0, n))
        return sensitivity

    # def getAnalyticalBound(self, P, opt, weights, sum_of_weights_all):
    #     n = P.shape[0]
    #     #k = int(2*np.log(n))
    #     k = 1
    #     return self.getAnalyticalBoundCore(P, opt, k, weights, sum_of_weights_all)

    def getAnalyticalBound(self, PwithEmbedding, opt, weights, sum_of_weights_all, use_k_median=False):
        P = PwithEmbedding
        n = P.shape[0]
        if n <= 1:
            return np.ones(P.shape[0])

        startK = int(2*np.ceil(np.log(2*n)))
        startK = min(int(math.ceil(n/10.0)), startK)
        endK = int(np.ceil(n**(1.0/2)))
        endK = max(startK + 1, endK)
        numK = int(2*np.ceil(np.log(n))) if n > 1 else 1
        MIN_K = 3
        if self.FAST and numK > MIN_K:
            numK = MIN_K if n > 1 else 1
        kList = np.geomspace(startK, endK, num=numK, dtype=int)
        kList = np.unique(kList)

        bestSens = None
        best_dist = None
        bestSumOfSens = np.Inf
        strike = 0
        bestK = 0
        pre_compute = 'auto'
        for k in kList:
            sens,dists = self.getAnalyticalBoundCore(P, opt, k, weights, sum_of_weights_all, pre_compute, use_k_median=use_k_median)
            sensSum = np.sum(sens)
            if sensSum < bestSumOfSens:
                bestSumOfSens = sensSum
                bestSens = sens
                best_dist = dists
                strike = 0
                bestK = k
                pre_compute = True
            else:
                strike += 1

            # print('New k to try: {}'.format((sensSum - k)))
            # print('Cost({}): {}'.format(k, sensSum))
            if strike >= 2:
                break

        # print("Best k: {} (start {}; end {})\n".format(bestK, startK, endK))
        return bestSens,best_dist

    @staticmethod
    def advindexing_roll(A, r):
        rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
        r[r < 0] += A.shape[1]
        column_indices = (column_indices if column_indices.ndim > 1 else column_indices[:, np.newaxis].T) - r[:, np.newaxis]
        return A[rows, column_indices]

    def getAnalyticalBoundCore(self, P, f, k, weights, sum_of_weights_all, pre_compute='auto', use_k_median=False):
        """
        :param P: A numpy matrix of mxd of ublabled data.
        :param w_opt: The optimal solution.
        :param opt: The optimal value.
        :param mean_point: The mean of P with respect to a weight vector
        :param Kyi: A scalar.
        :return: A bound for the sensitivities of P.
        """
        n = P.shape[0]
        # print("Before shape {}".format(P.shape))
        # transformer = random_projection.SparseRandomProjection(n_components=10)
        # P = transformer.fit_transform(P)
        # print("After shape {}".format(P.shape))

        k = int(min(k, n ** (3.0 / 5.0)))
        n_init = int(10*math.ceil(np.log(n) + 1)) if self.FAST else 128
        n_init = max(3*k, n_init)
        # initial_centers = np.array(kmeans_plusplus_initializer(P, k).initialize())
        start = time.time()
        if not use_k_median:
            centers, labels = init_kmeans_fast(P, k, squared=True)
        else:
            centers, labels = init_kmeans_fast(P, k, squared=False)
        print("Time taken for computing the seeding is {}".format(time.time() - start))

        if False:
            if not use_k_median:
                kmeans = KMedians(k=k, power=2.0, itermax=(3 if self.FAST else 30))
                kmeans.fit(P, initial_centers)
                labels = kmeans.labels_
                centers = kmeans.centers_
            else:
                kmedians = KMedians(k=k, power=1.0, itermax=(3 if self.FAST else 30))
                kmedians.fit(P, initial_centers)
                labels = kmedians.labels_
                centers = kmedians.centers_
        if False:
            if not use_k_median:
                kmeans_instance = kmeans(P, initial_centers)
                kmeans_instance.process()
                clusters = kmeans_instance.get_clusters()
                labels = np.empty((P.shape[0], ), dtype=np.int)
                for i in range(len(clusters)):
                    for j in clusters[i]:
                        labels[j] = i
                centers = kmeans_instance.get_centers()
            else:
                kmedians_instance = kmeans(P, initial_centers, metric=distance_metric(type_metric.EUCLIDEAN))
                kmedians_instance.process()
                clusters = kmedians_instance.get_clusters()
                labels = np.empty((P.shape[0],), dtype=np.int)
                for i in range(len(clusters)):
                    for j in clusters[i]:
                        labels[j] = i
                centers = kmedians_instance.get_centers()




        # max_iter = 30 if self.FAST else 8
        # if n <= 5000:
        #     if not use_k_median:
        #         kmeans = KMeans(k=k, max_iter=max_iter)
        #     else:
        #         kmeans = KMedians(k=k, max_iter=max_iter)
        #     kmeans.fit(P)
        # else:
        #     if not use_k_median:
        #         kmeans = KMeans(k=k, max_iter=max_iter)
        #     else:
        #         kmeans = KMedians(k=k, max_iter=max_iter)
        #     kmeans.fit(P)
        #
        # labels = kmeans.labels_
        # centers = kmeans.cluster_centers_

        # labels = np.random.randint(0, k-1, n).flatten()

        indicator_matrix = MLProbs.advindexing_roll(
            np.repeat(np.eye(1, k, 0), np.ma.size(P, 0), axis=0), labels)

        # OLD Way
        # cOfP = np.dot(indicator_matrix, centers)

        # sum of weights per cluster
        denom = np.sum(np.multiply(np.expand_dims(weights, 1), indicator_matrix), 0)
        invSumWeightPerCluster = np.diag(np.divide(1.0, denom, out=np.zeros_like(denom), where=denom !=0))

        # Computing a matrix of centers
        # cOfP = np.dot(indicator_matrix, np.dot(np.dot(indicator_matrix, invSumWeightPerCluster).T,
        #                                        np.multiply(P, weights[:, np.newaxis])))
        cOfP = np.array([centers[x] for x in labels])

        weightPerCluster = np.dot(indicator_matrix, np.sum(np.multiply(indicator_matrix,
                                                                       np.expand_dims(weights, 1)), axis=0).T)
        p_delta = cOfP - P


        # alpha value
        a = (sum_of_weights_all - weightPerCluster) / (
                2.0*sum_of_weights_all*weightPerCluster)


        # compute the norms
        p_delta_norms = np.linalg.norm(p_delta, axis=1)
        # print('Sum of distances is {}'.format(np.sum(p_delta_norms)))
        p = p_delta_norms

        # print('sum of norms = {}'.format(np.sum(p)))
        # print('sum of a = {}'.format(np.sum(a)))


        # Complicated but improved one
        #expr = (2*(p - np.sqrt(2)*a*np.sqrt(f))**2*(np.sqrt(4*a**2 + (p**2*(p - 2*np.sqrt(2)*a*np.sqrt(f))**2)/(2*f*(p - np.sqrt(2)*a*np.sqrt(f))**2)) - 2*a))/(p - 2*np.sqrt(2)*a*np.sqrt(f))**2

        #expr = (np.sqrt(2)*np.sqrt(f)*p - 2*a*f)/f

        # Proper one
        expr = 9 / 2 * (np.sqrt(4 * a**2 + (2 * p**2) / (9 * f)) - 2 * a)
        ind = expr >= 3*a
        # print("f: {}".format(f))
        # print("Percentage activated: {}".format(np.sum(ind) / a.shape[0]*100.0))
        expr = np.maximum(3 * a, expr)
        # expr = np.maximum(2 * a, expr)

        term = 1.0 / weightPerCluster + expr
        # greaterThan1 = term >= 1
        # print("Percentage greater than 1: {}".format(np.sum(greaterThan1) / a.shape[0] * 100.0))
        term = np.minimum(term, 1)

        return np.maximum(term, 0.0), np.sum(p)


    def computeOptimalSolverUsingScipy(self, X, Y, weights, C):
        """
        :param X: A numpy array of nxd excluding the labels.
        :param Y: A numpy array containing the labels.
        :param weights: A weight vector with respect to X.
        :param C: A regularization parameter.
        :return: The optimal solution of the SVMs problem with respect to X,Y and weights using SciPy solver.
        """
        # define the cost function
        f = (lambda x, xTrain=X, yTrain=Y: self.evaluateSvmsPerS(xTrain, yTrain, weights, x[:-1], x[-1], C))

        # define the gradient of the cost function
        g = (lambda x, xTrain=X, yTrain=Y: self.gradient(xTrain, yTrain, weights, x[:-1], x[-1]))

        # sample random starting point
        x0 = np.random.rand(1, X.shape[1] + 1)

        # solve the problem
        # tol = 15
        max_iter = 100
        if self.FAST:
            max_iter = 10

        user_options = {'disp': False}
        if self.FAST:
            user_options['maxiter'] = 30

        res = minimize(f, x0, jac=g, tol=10.0, options=user_options)
        # attain the optimal variables (w,b)
        if self.params['fit_intercept']:
            w = res.x[:-1]
            b = res.x[-1]
        else:
            w = res.x
            b = 0.0
        return w, b


    def evaluateLogistic(self, w, b=0.0):
        reg = 0.5 * norm(w)**2
        likelihood = np.log(np.exp(np.multiply(-self.train['Y'], self.train['X'].dot(w) + b)) + 1)
        return reg + self.params['C'] * np.sum(likelihood)

    def computeSensitivityMyWay(self, P=None, weights=None):
        print('Start computing my own sensitivity')

        if P is None:
            X = self.params['X']
            Y = self.params['Y']
        else:
            X = P[:, :-1]
            Y = P[:, -1]

        n = X.shape[0]
        if weights is None:
            weights = self.weights

        # start time and compute the optimal solution of the SVM problem
        ts = time.time()

        # sensitivity = np.empty(P.shape[0], )

        pos_idxs = np.where(Y > 0)[0]
        neg_idxs = np.where(Y < 0)[0]


        U = np.empty((n, X.shape[1]))
        U[pos_idxs, :], _, _ = \
            np.linalg.svd(np.multiply(np.sqrt(self.weights[pos_idxs])[:, np.newaxis],
                                      X[pos_idxs]), full_matrices=False)
        U[neg_idxs, :], _, _ = \
            np.linalg.svd(np.multiply(np.sqrt(self.weights[neg_idxs])[:, np.newaxis],
                                      X[neg_idxs]), full_matrices=False)
        sum_old_weights = np.sum(self.weights)



        sensitivity = np.empty((n,))
        pos_weights = np.sum(self.weights[pos_idxs])
        neg_weights = np.sum(self.weights[neg_idxs])
        sensitivity[pos_idxs] = self._sense_bound_lambda(x=U[pos_idxs, :], w=self.weights[pos_idxs],
                                                               args=(pos_weights, neg_weights,
                                                                     np.sum(self.weights))
                                                               )
        sensitivity[neg_idxs] = self._sense_bound_lambda(x=U[neg_idxs, :], w=self.weights[neg_idxs],
                                                               args=(neg_weights, pos_weights, np.sum(self.weights)))
        self.our_coreset_time = time.time() - ts
        return sensitivity
