

import numpy as np
from tqdm import tqdm
from ConvLearner import ConvLearner
from TrainingDataMgr import TrainingDataMgr
from Globals import SAMPLE_LENGTH

class Trainer:
    def __init__(self):
        self.mgr = TrainingDataMgr(SAMPLE_LENGTH, "train.csv")
        self.eof = False
        self.learner = ConvLearner()
        
    def Train(self, numEpochs):
        for i in range(numEpochs):
            self.RunOneEpoch()
            
            
    def RunOneEpoch(self):
        grossCount = 0
        while(self.eof == False):
            for i in tqdm(range(100)):
                self.RunOneBatch()
                if(self.eof==True):
                    break
            grossCount += 1
            if((grossCount % 10) == 0):
                self.learner.Save()
    
        self.eof = False
        self.mgr.Reset()
        
    def RunOneBatch(self):        
        batchList, labels = self.mgr.NextBatch()
        if(batchList == None):
            self.eof = True
            return

        batchTensor = np.stack(batchList, axis=0)
        self.learner.Fit(batchTensor, np.reshape(labels, (len(labels), 1)))
        
        
trainer = Trainer()
trainer.Train(2)


        
