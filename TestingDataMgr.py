
import os
import csv
import numpy as np
from Globals import RESHAPED_DIMS

class TestingDataMgr:
    def GetFileNames(self, dataDir):
        files = []
        for f in os.listdir(dataDir):
            if f.endswith(".csv"):
                files.append(os.path.join(dataDir, f))
        return files
    
    def GetData(self, fileName):
        data = []
        with open(fileName, "r") as inputFile:
            reader = csv.reader(inputFile)
            #Skip header
            prevLine = reader.__next__()
            while(True):
                try:
                    curLine = reader.__next__()
                    data.append(float(curLine[0]))
                    prevLine = curLine
                except StopIteration:
                    break
            
            label = None
            if(len(prevLine) > 1):
                label = float(prevLine[1])
            
            reshaped = np.reshape(data, (1,) + RESHAPED_DIMS)
            return reshaped, label
