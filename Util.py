import numpy as np
import scipy.io as sio
#import matlab.engine
import os, json, math
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


color_matching = {
    'Our Coreset': 'red',
    'Tukan et al., 2020': 'red',
    'Uniform Sampling': 'blue',
    'All Data': 'black',
}
color_matching = {
    'Our Coreset: $k$-median': 'red',
    'Our Coreset: $k$-means': 'green',
    'Uniform Sampling': 'blue',
    'All Data': 'black'
}

def LoadEntry(entryName, matFile):
    return np.array(matFile.get(entryName), dtype='f')
    #return np.array(matFile.get(entryName))

def LoadMatFile(filePath):
    matFile = sio.loadmat(filePath)
    sensitivity = LoadEntry('sensitivity', matFile)
    trainingData = LoadEntry('trainingData', matFile)
    testingData = LoadEntry('testingData', matFile) if 'testingData' in matFile.keys() else None
    weights = LoadEntry('weights', matFile)
    timeTaken = float(matFile.get('timeTaken'))

    return {'sensitivity': sensitivity,
            'trainingData': trainingData,
            'testingData': testingData,
            'timeTaken': timeTaken,
            'weights': weights}

def CreateDirectory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def WriteToFile(filePath, output):
    with open(filePath, 'w') as f:
        f.write('\n'.join(output))

def LoadModelSpec(data, classification, C):
    X = data['trainingData'][:, 0:-1]
    Y = data['trainingData'][:, -1]
    weights = data['weights']

    XTest = data['testingData'][:, 0:-1]
    YTest = data['testingData'][:, -1]
    dict = {'X': X.tolist(), 'Y': Y.tolist(),
            'XTest': XTest.tolist(), 'YTest': YTest.tolist(),
            'classify': classification,
            'C': C,
            'weights': weights.tolist()}

    return json.dumps(dict)

class Sampler(object):

    def __init__(self, n, dataSet, streaming=False):

        NUM_INTERVALS = 15
        #constant_max = 18 # was 20
        constant_max = 10  # was 20
        frac = 0.2
        minVal = math.ceil(max(150, math.log(n)))
        maxVal = max(min(1000, n * frac), minVal * 25)

        if 20000 > n > 10000:
            minVal = max(math.log(n)**2, 200)
            maxVal = math.log(n) ** 2 * constant_max

        if n > 10000:
            minVal = max(math.log(n) ** 2, 200)
            maxVal = math.log(n) ** 2 * constant_max

        if n > 20000:
            maxVal = min(int(n ** (4.0 / 5.0)), int(maxVal))

        if n > 20000:
            minVal = max(minVal, 1000)
            maxVal = min(int(n**(4.0/5.0)), int(maxVal))

        if 'Adult' in dataSet or 'Credit' in dataSet:
            minVal = 4000
            maxVal = 15000
            print('hhm')

        if 'W8' in dataSet:
            minVal = 5800
            maxVal = 15000

        if 'Credit' in dataSet:
            minVal = 150
            maxVal = 1000

        if 'W1' in dataSet:
            minVal = 5800
            maxVal = 15000

        if 'USPS' in dataSet:
            minVal = 500
            maxVal = 2000

        #if 'Cod' in dataSet:


        # NUM_INTERVALS = 10

        if streaming:
            # minVal = n//4
            # maxVal = n
            NUM_INTERVALS = 10

            self.samples = list(np.around(np.linspace(minVal, maxVal, num=NUM_INTERVALS, dtype=int)))
        else:
            maxVal = min(int(n*4.0/5.0), int(maxVal))
            self.samples = list(np.around(np.geomspace(minVal, maxVal, num=NUM_INTERVALS, dtype=int)))
        if 'W8' in dataSet:
            minVal = 4000
            maxVal = 12000
        elif 'Cod' in dataSet:
            minVal = 50
            maxVal = 500
        elif 'W1' in dataSet:
            minVal = 5800
            maxVal = 15000
        elif 'Credit' in dataSet:
            minVal = 150
            maxVal = 1000
        elif 'Skin' in dataSet:
            minVal = 250
            maxVal = 3000
        else:
            minVal = 50
            maxVal = 500

        NUM_INTERVALS = 15
        # constant_max = 18 # was 20
        constant_max = 10  # was 20
        frac = 0.2
        minVal = math.ceil(max(150, math.log(n)))
        maxVal = max(min(1000, n * frac), minVal * 25)

        if n > 10000:
            minVal = max(math.log(n) ** 2, 200)
            maxVal = math.log(n) ** 2 * constant_max

        if n > 20000:
            maxVal = min(int(n ** (4.0 / 5.0)), int(maxVal))

        if 'Adult' in dataSet or 'Credit' in dataSet:
            minVal = 4000
            maxVal = 15000

        if 'W1' in dataSet:
            minVal = 5800
            maxVal = 15000

        if 'Credit' in dataSet:
            minVal = 150
            maxVal = 1000

        if 'USPS' in dataSet:
            minVal = 500
            maxVal = 2000

        # if 'Cod' in dataSet:

        NUM_INTERVALS = 10

        if streaming:
            # minVal = n//4
            # maxVal = n
            NUM_INTERVALS = 10

            self.samples = list(np.around(np.linspace(minVal, maxVal, num=NUM_INTERVALS, dtype=int)))
        else:
            maxVal = min(int(n * 4.0 / 5.0), int(maxVal))
            self.samples = list(np.around(np.geomspace(minVal, maxVal, num=NUM_INTERVALS, dtype=int)))

        # self.samples = np.linspace([minVal], [maxVal], NUM_INTERVALS).flatten().astype(np.int)



def preprocessData(P, normalize=True, centeralize=True, p=2):
        """
        ##################### preprocessData ####################
        Input:
            - P: A nxd numpy matrix such that the labels are stated at the last column.
            - normalize: A binary variable with respect to whether to normalize the data or not (Default value: True).
            - centralize: A binary variable with respect to whether to centralize the data or not (Default value: True).
            - p: An integer stating the p-norm (Default value: 2).

        Output:
            - P: A dataset which has been either normalized, centralized, both or neither.

        Description:
            This process is responsible for applying standard pre-processing techniques specified by the user.
        """
        X = P[:, :-1]  # attain the data points
        Y = P[:, -1]  # attain the labels

        # Change the smaller label to -1 and the larger label to 1 (Handling binary case)
        unique_Y = np.unique(Y)
        zero = np.min(unique_Y)
        one = np.max(unique_Y)
        Y[Y == zero] = -1
        Y[Y == one] = 1

        # If centralization is specified
        if centeralize and False:
            s = np.std(X, axis=0)
            X = X - np.mean(X, axis=0)
            s[s == 0] = 1.0
            X = np.divide(X, s)

        # If normalization is specified
        if normalize:
            # norms = np.power(np.sum(np.power(X, p), 1), 1.0 / p)
            # X = X / np.max(norms)
            norms = np.sqrt(np.sum(X ** 2, axis=1))
            max_norm = np.max(norms)
            if max_norm > 1:
                X /= max_norm
        if centeralize:
            X = StandardScaler().fit_transform(X=P[:, :-1], y=P[:, -1])



        # "packing" the data-points with their respected label
        P = np.hstack((X, np.expand_dims(Y, axis=1)))

        return P


def analyzeData(P, W, is_labeled=True):
    """
        ##################### analyzeData ####################
        Input:
            - P: A nxd numpy array which holds the datasets that is being used.
            - W: A numpy array of n entries which represents the weight vector with respect to the rows of P.
            - is_labeled: A boolean variable for stating whether the dataset is labeled or not.

        Output:
            - n: The number of rows of P.
            - d: The number of columns of P.
            - mean_P: The mean of the rows of P.
            - X: The submatrix of n x (d -1) of P (excluding the last oclumn).
            - Y: The labels with respect to each row of P which is the last column of P.
            - positive_indices: A numpy array containing the indices of rows of P with positive label (1).
            - negative_indices: A numpy array containing the indices of rows of P with negative label (-1).
            - mean_positive: The mean of the rows of P with positive label.
            - mean_negative: The mean of the rows of P with negative label.

        Description:
            This process is responsible for preparing all the required information needed for computing the sensitivity
            with respect to P under the problem of SVM.
    """

    n = np.ma.size(P, axis=0)  # number of rows
    d = np.ma.size(P, axis=1)  # number of columns

    mean_P = np.average(P[:, :-1], weights=W, axis=0)  # compute mean

    if is_labeled:
        X = P[:, :-1]  # exclude the last column for it the labels
        Y = P[:, -1]   # the labels
        positive_indices = np.array(np.where(Y == 1)).flatten()  # indices of positive labeled rows
        negative_indices = np.setdiff1d(range(0, n), positive_indices)  # indices of negative labeled rows

        if positive_indices.size != 0:
            mean_positive = np.average(X[positive_indices, :], weights=W[positive_indices], axis=0)
        else:
            mean_positive = None

        if negative_indices.size != 0:
            mean_negative = np.average(X[negative_indices, :], weights=W[negative_indices], axis=0)
        else:
            mean_negative = None
    else: # if data is not labeled, no need for such information thus returning None
        X = Y = positive_indices = negative_indices = mean_negative = mean_positive = None

    return n, d, mean_P, X, Y, positive_indices, negative_indices, mean_positive, mean_negative


def createData(file_path, normalization, centering):
    matFile = sio.loadmat(file_path)
    P = LoadEntry('data', matFile)
    P = preprocessData(P, normalize=normalization, centeralize=centering, p=2)
    weights = np.ones((np.ma.size(P, 0), ))
    X = P[:, :-1]
    Y = P[:, -1]

    return {'trainingData': P,
            'testingData': P,
            'weights': weights}



