
import os
import csv
from tqdm import tqdm
from ConvLearner import ConvLearner
from TestingDataMgr import TestingDataMgr



class Tester:
    def __init__(self, dataDir, outputFileName):
        self.mgr = TestingDataMgr()
        self.fileNames = self.mgr.GetFileNames(dataDir)
        self.outputFileName = outputFileName
        self.learner = ConvLearner()
        self.learner.Load()
        
    def RunTest(self, header):
        with open(self.outputFileName, "w", newline='') as oFile:
            writer = csv.writer(oFile)
            writer.writerow(header)
            
            numFiles = len(self.fileNames)
            for i in tqdm(range(numFiles)):
                fname = self.fileNames[i]
                testResult = self.TestSingleFile(fname)
                writer.writerow(testResult)
                
    def TestSingleFile(self, fileName):
        data, label = self.mgr.GetData(fileName)
        prediction = self.learner.Predict(data)[0]
        result = [os.path.basename(fileName), prediction]
        if(label != None):
            result.append(label)
            result.append(abs(label - prediction))
        return result
                
Validate = True
tester = None
header = ["seg_id", 	"time_to_failure"]

if(Validate == True):
    tester = Tester("validate", "validation_results.csv")
    header.append("yTrue")
    header.append("delta")
else:
    tester = Tester("test", "test_submission.csv")

tester.RunTest(header)

