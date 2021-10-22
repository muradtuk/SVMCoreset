import multiprocessing
import numpy as np
import copy
import sys
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock

#import matlab.engine
import MLprobs as MLp
import Coreset as CS
import Util
import Grapher
import itertools
import math, time
import os
from sklearn.datasets import dump_svmlight_file
from scipy.interpolate import pchip
from scipy import stats
import MergeAndReduceTree
from datetime import datetime

from scipy.io import savemat
from pathlib import Path
from scipy.io import loadmat

#eng = matlab.engine.start_matlab()


class Evaluator(object):
    mutex = Lock()
    NUM_THREADS = multiprocessing.cpu_count()
    NUM_THREADS =1
    NUM_THREADS_ALL = multiprocessing.cpu_count() // 4
    DEFAULT_NUMBER_OF_ADDED_TESTS = 1

    def __init__(self, dataName, fileName, opts):

        dataDir = 'data'
        normalize = opts['normalize']
        center = opts['center']
        numRepetitions = opts['numRepetitions']
        streaming = opts['streaming']
        evaluateCVM = opts['evaluateCVM']

        if streaming:
            self.NUM_THREADS = 1
        self.normalize = normalize
        self.center = center

        self.fileName = fileName
        self.numRepetitions = opts['numRepetitions']
        self.streamable = streaming

        
        self.fileNameFullPath = None # Put here path to dataset file


        self.C = 1.0
        self.classify = False
        self.data = Util.createData(self.fileNameFullPath, normalize, center)

        modelSpec = Util.LoadModelSpec(self.data, self.classify, self.C)
        self.mlProbs = MLp.MLProbs(copy.deepcopy(modelSpec), fileName, normalize, center, streaming)

        self.data['sensitivity_murad_cenk'] = self.mlProbs.computeSensitivity(use_k_median=True) if not self.streamable else None
        # temp = np.load('HTRU_2.npz')
        self.data['timeTaken_murad_cenk'] = self.mlProbs.coreset_time if not self.streamable else 0.0
        self.data['sensitivity_murad_cenk_means'] = self.mlProbs.computeSensitivity() if not self.streamable else None
        self.data['timeTaken_murad_cenk_means'] = self.mlProbs.coreset_time if not self.streamable else 0.0



        self.N = self.data['trainingData'].shape[0]

        # Load the matrix and save it as svm light file for CVM eval.
        X = self.data['trainingData'][:, 0:-1]
        Y = self.data['trainingData'][:, -1]
        self.cvmX = X
        self.cvmY = Y
        self.cvmFileName = fileName.replace('.mat', '')
        self.evaluateCVM = evaluateCVM

        if self.evaluateCVM:
            dump_svmlight_file(X, Y, self.cvmFileName)

        self.lambdaPegasos = 1.0 / (float(self.N) * self.C)
        self.iterPegasos = 10000
        self.timeStampDiff = 1e-3
        self.batchSize = math.ceil(math.log(self.N))

        self.fileNamePath = os.path.abspath("{}/{}".format(dataDir, fileName))

        print('Done generating sensitivities')
        print('Sum of sensitivities using $k$-median is {}'.format(np.sum(self.data['sensitivity_murad_cenk'])))
        print('Sum of sensitivities using $k$-means is {}'.format(np.sum(self.data['sensitivity_murad_cenk_means'])))

        self.samples = Util.Sampler(self.N, fileName, streaming).samples
        self.uniformCoresets = [CS.Coreset() for j in range(numRepetitions)]
        self.coresets = [CS.Coreset() for j in range(numRepetitions)]
        self.output = []
        self.title = (fileName.replace('.mat', '') + " ($N = {}$)".format(self.N)).replace(dataDir, '')
        self.pegasosOn = False
        self.batchSizeOn = (self.pegasosOn and False)
        self.weights = np.array(self.data['weights']).flatten()
        self.numOfAddedTests = self.DEFAULT_NUMBER_OF_ADDED_TESTS
        # self.legend = ['Uniform Sampling', 'Tukan et al., 2020', 'Our Coreset', 'All Data']
        self.legend = ['Uniform Sampling', 'Our Coreset: $k$-median', 'Our Coreset: $k$-means', 'All Data']

        self.algorithms = [lambda i, sampleSize, seed: self.uniformCoresets[i].computeCoreset(self.data['trainingData'], \
                                np.ones((self.N, 1)), sampleSize, self.weights, seed),
                           lambda i, sampleSize, seed: self.coresets[i].computeCoreset(self.data['trainingData'], \
                                self.data['sensitivity_murad_cenk'], sampleSize, self.weights, seed), \
                           lambda i, sampleSize, seed: self.coresets[i].computeCoreset(self.data['trainingData'], \
                               self.data['sensitivity_our'], sampleSize, self.weights, seed)] if not self.streamable else \
                           [lambda i, sampleSize, seed:
                            MergeAndReduceTree.MergeAndReduceTree(self.data['trainingData'], sampleSize, \
                               sampleSize, self.mlProbs, True).runMergeAndReduce(seed),
                           lambda i, sampleSize, seed: MergeAndReduceTree.MergeAndReduceTree(self.data['trainingData'], sampleSize, \
                               sampleSize, self.mlProbs, False).runMergeAndReduce(seed)]
        self.algorithms = [lambda i, sampleSize, seed: self.uniformCoresets[i].computeCoreset(self.data['trainingData'], \
                                np.ones((self.N, 1)), sampleSize, self.weights, seed),
                           lambda i, sampleSize, seed: self.coresets[i].computeCoreset(self.data['trainingData'], \
                                self.data['sensitivity_murad_cenk'], sampleSize, self.weights, seed),
                           lambda i, sampleSize, seed: self.coresets[i].computeCoreset(self.data['trainingData'], \
                                self.data['sensitivity_murad_cenk_means'], sampleSize, self.weights, seed)] \
            if not self.streamable else \
                           [lambda i, sampleSize, seed:
                            MergeAndReduceTree.MergeAndReduceTree(self.data['trainingData'], sampleSize, \
                               sampleSize, self.mlProbs, True).runMergeAndReduce(seed),
                           lambda i, sampleSize, seed: MergeAndReduceTree.MergeAndReduceTree(self.data['trainingData'], sampleSize, \
                               sampleSize, self.mlProbs, False).runMergeAndReduce(seed)]


        resultsDir = 'results'
        append = ''
        
        if self.streamable:
            append += '-streaming'

        resultsDir += append
        self.append = append

        self.EvaluateDataSet()


    
    def EvaluateCoreset(self, coresetFunc, sampleSize):

        DEBUG = False

        e = lambda x: self.mlProbs.evaluateRelativeError(x[0][:, :-1], x[0][:, -1], x[1])
        if self.classify:
            e = lambda x: self.mlProbs.evaluateAccuracy(x[0], x[1], x[2])
        if not DEBUG and self.NUM_THREADS > 1:
            pool = ThreadPool(self.NUM_THREADS)
            coresets = pool.map(lambda i, sampleSize=sampleSize: coresetFunc(i, sampleSize, i),
                                range(self.numRepetitions))
            results = pool.map(e, coresets)
            pool.close()
            pool.join()
        else:
            coresets = [coresetFunc(i, sampleSize, i) for i in range(
                self.numRepetitions)]

            results = [e(x) for x in coresets]

        # Return the (mean accuracy, mean computation time)
        stat = lambda x : np.mean(x)
        #stat = lambda x: np.median(x)
        return stat([x[0] for x in results]), \
               stat(np.array([x[1] for x in results]) + np.array([x[-1] for x in coresets])), \
               stat([x[0].shape[0] for x in coresets]), stats.median_absolute_deviation([x[0] for x in results]), \
               stats.median_absolute_deviation(np.array([x[1] for x in results]))

     

    def EvaluateDataSet(self):
        RAW_ERROR = False
        
        # Column 0: Uniform, Column 1: Ours, Column 2: CVM (if enabled), All Data
        numAlgorithms = len(self.algorithms)
        accuracyMat = np.zeros((numAlgorithms + self.numOfAddedTests, len(self.samples)))
        varianceMat = np.zeros((numAlgorithms + self.numOfAddedTests, len(self.samples)))
        varianceMatTime = np.zeros((numAlgorithms + self.numOfAddedTests, len(self.samples)))
        timeMat = np.zeros((numAlgorithms + self.numOfAddedTests, len(self.samples)))
        sampleSizeMat = np.zeros((numAlgorithms + self.numOfAddedTests, len(self.samples)))


        for i, sampleSize in enumerate(self.samples):
            print("Sample size: {}".format(sampleSize))
            for j in range(numAlgorithms):
                thisSampleSize = sampleSize
                startTime = time.time()
                coresetFunc = self.algorithms[j]
                meanAccuracy, meanTime, meanSampleSize, variance, varianceTime = self.EvaluateCoreset(coresetFunc, thisSampleSize)

                accuracyMat[j, i] = meanAccuracy
                varianceMat[j, i] = variance
                varianceMatTime[j, i] = varianceTime
                sampleSizeMat[j, i] = sampleSize
                timeMat[j, i] = (meanTime + self.data['timeTaken_murad_cenk']) if j == 1 else \
                    (meanTime + self.data['timeTaken_murad_cenk_means'] if j == 2 else meanTime)
                print("Algorithm {} took {:.3f} seconds".format(j, time.time() - startTime))

        timeMat[-1, :] = np.ones((1, len(self.samples))) * self.mlProbs.optTime

        minNumSamples = np.min(np.min(sampleSizeMat))
        maxSamples = np.max(self.samples)
        sampleSizeMat[-1, :] = np.linspace(minNumSamples, maxSamples, len(self.samples))

        suffix = '-streaming-' if self.streamable else ('-CVM-' if self.evaluateCVM else '-')
        if self.classify:
            saveFileName = 'ClassificationResults'
        else:
            saveFileName = 'LossResultsTest' + suffix

        nameDir = 'results' + suffix[:-1] + '/' + self.fileName.replace('Coreset.mat', '')
        file_path = str(Path.cwd()) + '/' + nameDir + '/'
        Util.CreateDirectory(file_path)
        saveFileName = saveFileName + self.fileName.replace('.mat', '')
        timeStamp = '{:%m-%d-%Y-%H-%M-%S}'.format(datetime.now())

        if self.classify:
            accuracyMat[-1, :] = np.ones((1, accuracyMat.shape[1]))*self.mlProbs.allDataAccuracy

        savemat(file_path + saveFileName + timeStamp + '.mat', dict(timeMat=timeMat, accuracyMat=accuracyMat,
                                            varianceMat=varianceMat, varianceMatTime=varianceMatTime,
                                            sampleSizeMat=sampleSizeMat, legend=self.legend))

        print("********************************************")
        print('{} Done!'.format(bareTitle))
        print("********************************************")


    def getEpsWrtWeights(self, X, Y, W):
        self.mutex.acquire()
        EPS =abs(np.sum(W) / self.N - 1)

        self.mutex.release()
        return EPS, 0.0

def main(numRepetitions, streaming=False):

    permutations = itertools.product([False, True], repeat=2)

    for permutation in permutations:

        normalize = permutation[0]
        centering = permutation[1]
        if normalize == False or centering == False:
            continue
        normalize = True
        centering = True

        opts = {
            'numRepetitions': numRepetitions,
            'normalize': normalize,
            'center': centering,
            'streaming': streaming,
            'evaluateCVM': False
        }

        # example command
        Evaluator('W1Data.mat', 'W1Coreset.mat', opts)
        

if __name__ == '__main__':
    numRepetitions = 40
    streaming = False

    if streaming:
        numRepetitions = 20

    if len(sys.argv) > 1:
        numRepetitions = int(sys.argv[1])

    main(numRepetitions, streaming)
