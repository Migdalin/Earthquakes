

import numpy as np
from tqdm import tqdm
from ConvLearner import ConvLearner
from TrainingDataMgr import TrainingDataMgr
from Globals import SAMPLE_LENGTH, RESHAPED_DIMS

class Trainer:
    def __init__(self):
        self.mgr = TrainingDataMgr(SAMPLE_LENGTH, "train.csv")
        self.batchSize = 32
        self.eof = False
        self.learner = ConvLearner()
        
    def Train(self, numEpochs):
        for i in range(numEpochs):
            self.RunOneEpoch()
            self.learner.Save()
            
    def RunOneEpoch(self):
        while(self.eof == False):
            for i in tqdm(range(100)):
                self.RunOneBatch()
                if(self.eof==True):
                    break
    
        self.eof = False
        self.mgr.Reset()
        
    def RunOneBatch(self):        
        batchList = []
        labels = []
        for _ in range(self.batchSize):
            nextSample = self.mgr.Next()
            if(nextSample == None):
                self.eof = True
                break
            batchList.append(np.reshape(nextSample, RESHAPED_DIMS))
            labels.append(self.mgr.currentLabel)
        
        if(len(batchList) == 0):
            assert (self.eof == True), "Unexplained empty batch."
            return
        
        batchTensor = np.stack(batchList, axis=0)
        self.learner.Fit(batchTensor, np.reshape(labels, (len(labels), 1)))
        
trainer = Trainer()
trainer.Train(3)


        
