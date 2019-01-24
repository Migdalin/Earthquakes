

import csv
import random

class TrainingDataMgr:
    def __init__(self, sampleLength, iDataPath):
        self.sampleLength = sampleLength
        self.data = []
        self.inputFile = open(iDataPath, "r", newline='')
        self.csvReader = csv.reader(self.inputFile)
        self.currentLabel = None
        self._fillBuffer()
        
    def Next(self):
        if(self._skip() == None):
            return None
        
        return self.data

    def _skip(self):
        numToSkip = random.randint(100, 1000)
        for _ in range(numToSkip):
            if(self._getNextRow() == None):
                return None
            self.data.pop(0)
        return numToSkip
            
    def _fillBuffer(self):
        while(len(self.data) < self.sampleLength):
            nextRow = self._getNextRow()
            if(nextRow == None):
                return None
        return self.data            
    
    def _getNextRow(self):
        try:
            nextRow = self.csvReader.__next__()
            self.data.append(nextRow[0])
            self.currentLabel = nextRow[1]
            return nextRow
        except StopIteration:
            return None
        
    def Reset(self):
        self.inputFile.seek(0)
        self.data.clear()
        self.currentLabel = None
        self._fillBuffer()
        