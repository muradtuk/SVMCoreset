import numpy as np
import copy
import time
from scipy import optimize
from datetime import datetime

class Coreset(object):
    def __init__(self):
        self.weights = []
        self.S = []
        self.probability = []


    @staticmethod
    def expectedUnique(probabilities, sampleSize):
        vals = 1 - (1 - probabilities) ** sampleSize
        expectation = np.sum(vals, axis=-1)
        return np.ceil(expectation)

    @staticmethod
    def adaptSampleSize(probabilities, targetSampleSize, uniform=False):

        # check for case of only one probability
        if probabilities.shape[0] == 1:
            return targetSampleSize
        if targetSampleSize == 0 or targetSampleSize == 1:
            return targetSampleSize

        probabilities = probabilities.flatten()
        idx = probabilities.nonzero()[0]
        targetSampleSize = int(targetSampleSize)
        n = idx.shape[0]

        if n == 0 or n == 1:
            return n

        targetSampleSize = min(targetSampleSize, n - 1)
        # p = np.array(probabilities[idx])
        p = probabilities[idx].flatten()

        f = lambda x, p=p: Coreset.expectedUnique(p, int(x)) - targetSampleSize
        # A lower bound on the number of required samples has a closed form
        # solution using Jensen's inequality and the concavity of 1 - (1-x)^m.
        meanP = p.mean()
        lower = np.log(n / (n - targetSampleSize)) / \
                np.log(1 / (1 - meanP))
        lower = int(np.floor(lower))

        upper = max(1, lower)
        while f(upper) < 0:
            upper *= 2

        sampleSize = int(optimize.bisect(f, lower, upper))

        return sampleSize

    def computeCoreset(self, P, sensitivity, targetSampleSize, weights=None, SEED = 1):

        if weights is None:
            weights = np.ones((P.shape[0], 1)).flatten()

        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)

        # The probability of a point prob(p_i) = s(p_i) / t
        self.probability = sensitivity.flatten() / t

        sampleSize = Coreset.adaptSampleSize(self.probability, targetSampleSize)
        # print('Orig sample size = {}, changed sample size = {}'.format(targetSampleSize, sampleSize))
        sampleSize = targetSampleSize
        startTime = time.time()
        #print('sampleSize: {} (target = {})'.format(sampleSize, targetSampleSize))
        # The number of points is equivalent to the number of rows in P.
        n = P.shape[0]

        # initialize new seed
        np.random.seed(SEED)

        # Multinomial Distribution.
        # indxs = np.random.choice(n, sampleSize, p=self.probability.flatten())
        #
        # # Compute the frequencies of each sampled item.
        # hist = np.histogram(indxs, bins=range(n))[0].flatten()
        # indxs = np.nonzero(hist)[0]

        hist = np.random.multinomial(sampleSize, self.probability.flatten()).flatten()
        # print('hist {}'.format(hist))
        # print('hist.shape {}'.format(hist.shape))
        indxs = np.nonzero(hist)[0]

        # if indxs.shape[0] == 0:
        #     return self.computeCoreset(P, sensitivity, sampleSize, weights)
        # Select the indices.
        if P.ndim < 2:
            self.S = P[indxs.flatten()]
        else:
            self.S = P[indxs, :]

        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        # TESTING THE REMOVAL OF
        #hist = np.minimum(np.ones(hist.shape), hist)
        weights = np.asarray(np.multiply(weights[indxs], hist[indxs]), dtype=float).flatten()

        # TESTING WEIGHTS = 1.
        #weights = np.asarray(weights[indxs])

        self.weights = np.multiply(weights, 1.0 / (self.probability[indxs]*sampleSize))
        #print("X: {} , Y: {}, W: {}, hist: {}".format(self.S[:,:-1].shape, self.S[:,-1].shape, self.weights.shape, indxs.shape))
        timeTaken = time.time() - startTime

        if P.ndim > 2:
            return self.S[:,:-1], self.S[:,-1], self.weights, timeTaken
        else:
            return self.S, self.weights, timeTaken


    def mergeCoreset(self, coreObj):
        self.S = np.concatenate((self.S, coreObj.S))
        self.weights = np.concatenate((self.weights, coreObj.weights))


    def computeStreamableCoreset(self, matEng, filePath, leafSize, sampleSize, centering, normalize, C, isUniform, outputVars):
        #X, Y, W = matEng.SVMStream(filePath, leafSize, sampleSize, centering, normalize, C, isUniform, nargout=outputVars)
        #print('Sample size: {}; Leaf size: {}'.format(sampleSize, leafSize))
        #print('Leaf size: {}'.format(leafSize))
        X, Y, W, timeTaken = matEng.SVMStreamV2(filePath, float(leafSize), float(sampleSize), centering, normalize, C, isUniform, nargout=outputVars)
        return np.array(X), np.array(Y).flatten(), np.array(W).flatten(), np.array(timeTaken)