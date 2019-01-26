

import csv
import random
import numpy as np
from Globals import RESHAPED_DIMS, BATCH_SIZE

class TrainingDataMgr:
    def __init__(self, sampleLength, iDataPath):
        self.sampleLength = sampleLength
        self.data = []
        self.discontinuity = False
        self.currentLabel = None
        self.inputFile = open(iDataPath, "r", newline='')
        self.csvReader = csv.reader(self.inputFile)
        self._fillBuffer()
        
    def NextBatch(self):
        batchList, labels = self._getBatchInternal()
        
        while(self.discontinuity == True and batchList != None):
            print("Encountered label discontinuity.  Resetting buffer.")
            self._refillBuffer()
            batchList, labels = self._getBatchInternal()
        
        return batchList, labels

    def _getBatchInternal(self):
        if(self._skip() == None):
            return None, None

        batchList = []
        labels = []
        for _ in range(BATCH_SIZE):
            if(self._getNextRow() == None):
                return None, None
            
            if(self.discontinuity == True):
                break

            self.data.pop(0)            
            batchList.append(np.reshape(self.data, RESHAPED_DIMS))
            labels.append(self.currentLabel)
        
        return batchList, labels

    def _skip(self):
        numToSkip = random.randint(500, 5000)
        for _ in range(numToSkip):
            if(self._getNextRow() == None):
                return None
            self.data.pop(0)
            if(self.discontinuity == True):
                break
        return numToSkip
            
    def _fillBuffer(self):
        while(len(self.data) < self.sampleLength):
            nextRow = self._getNextRow()
            if(nextRow == None):
                return None
        assert(self.discontinuity == False), "Encountered discontinuity while filling read buffer."
        return self.data            
    
    def _getNextRow(self):
        try:
            nextRow = self.csvReader.__next__()
            self.data.append(float(nextRow[0]))
            nextLabel = float(nextRow[1])
            if(self.currentLabel != None and self.currentLabel < nextLabel):
                self.discontinuity = True
                
            self.currentLabel = nextLabel
            return nextRow
        except StopIteration:
            return None
        
    def Reset(self):
        self.inputFile.seek(0)
        self._refillBuffer()
        
    def _refillBuffer(self):
        self.discontinuity = False
        self.data.clear()
        self.currentLabel = None
        self._fillBuffer()
        