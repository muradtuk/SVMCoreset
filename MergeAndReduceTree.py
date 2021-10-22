import numpy as np
import Coreset as CS
from collections import deque
import time
import copy

class MergeAndReduceTree(object):
    def __init__(self, P, leaf_size, sample_size, ml_probs, is_uniform):
        self.P = copy.deepcopy(P)
        self.leaf_size = leaf_size
        self.sample_size = sample_size
        self.stack = []
        self.coreset_generator = CS.Coreset()
        self.ml_probs = ml_probs
        self.is_uniform = is_uniform

        self.numPos = np.sum(P[:,-1] == 1)
        self.numNeg = P.shape[0] - self.numPos
        self.minRatio = min(self.numPos, self.numNeg) / (P.shape[0] * np.log(P.shape[0] + 1))
        #print('Min ratio: {}'.format(self.minRatio))
        start = -time.time()
        self.batches = self.getBatches()
        #print('Getting batches took {}s'.format(time.time() + start))

    def getBatches(self):
        np.random.shuffle(self.P)
        indices = range(0, np.ma.size(self.P, 0))
        batches = np.array_split(indices, np.ma.size(self.P, 0) // self.leaf_size)
        # batches = [x for x in batches if x.shape[0] > 1]

        for idxs in batches:
            numPos = np.sum((self.P[idxs, -1] == 1))
            numNeg = idxs.shape[0] - numPos
            # print('n: {}'.format(P.shape[0]))
            # print('numPos: {}'.format(numPos))
            # print('numNeg: {}'.format(numNeg))
            if min(numPos, numNeg) < 1:
                return self.getBatches()

        n = self.P.shape[0]
        batches = deque([(batch, np.ones((np.ma .size(batch, 0), ))) for batch in batches])
        return batches


    def runMergeAndReduce(self, seed):
        start_time = time.time()
        batches = self.batches
        while len(batches) > 1:
            C1 = batches.popleft()
            C2 = batches.popleft()
            idxs,u = (np.hstack((C1[0].flatten(), C2[0].flatten())), np.hstack((C1[1], C2[1])))
            if not self.is_uniform:
                sens = self.ml_probs.computeSensitivity(self.P[idxs,:], u)
            else:
                sens = np.ones((np.ma.size(idxs, 0), ))
            S, v, _= self.coreset_generator.computeCoreset(idxs, sens, self.sample_size, u, seed)
            batches.append((S, v.flatten()))

        time_taken = time.time() - start_time

        return self.P[batches[0][0], :-1], self.P[batches[0][0], -1], batches[0][1], time_taken
