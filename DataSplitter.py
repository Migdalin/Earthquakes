

import csv
import os

VALIDATION_FILE_LENGTH = 150000

class DataSplitter:
    def __init__(self, inFile):
        self.csvReader = csv.reader(inFile)
        
        self.header = self.csvReader.__next__()
        self.prevLabel = 120.0
        self.discontinuity = False
        self.curValidationFileIndex = 1

    def BuildValidationFiles(self):
        self.discontinuity = False
        while (self.discontinuity == False):
            self.BuildValidationFile(
                    f"validate{str(self.curValidationFileIndex)}.csv")
            self.curValidationFileIndex += 1

    def BuildValidationFile(self, fileName):
        with open(fileName, "w") as outFile:
            writer = csv.writer(outFile)
            writer.writerow(self.header)
            
            for rowCount in range(VALIDATION_FILE_LENGTH):
                nextRow = self._getNextRow()
                writer.writerow(nextRow)
                if(self.discontinuity == True):
                    break
                
        if(self.discontinuity == True):
            os.remove(fileName)
        else:
            print(f"Built validation file:  fileName")
        
    def _getNextRow(self):
        try:
            result = self.csvReader.__next__()
            nextLabel = float(result[1])
            if(nextLabel > self.prevLabel):
                self.discontinuity = True

            self.prevLabel = nextLabel
            return result
        except StopIteration:
            return None
    
    def BuildTrainingFile(self):
        with open("train.csv", "w") as trainingFile:
            writer = csv.writer(trainingFile)
            writer.writerow(self.header)
            while(True):
                nextRow = self._getNextRow()
                if(nextRow == None):
                    break
                writer.writerow(nextRow)



with open("train_original.csv", "r") as inFile:
    splitter = DataSplitter(inFile)
    
    for _ in range(2):
        splitter.BuildValidationFiles()
    
    splitter.BuildTrainingFile()
