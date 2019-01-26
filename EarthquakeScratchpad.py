



#import csv
#
#
#inputFile = open("train.csv", "r", newline='')
#csvReader = csv.reader(inputFile)
#
## Skip header
#csvReader.__next__()
#prevLabel = 120.0
#
#numRows = 0
#while True:
#    try:
#        row = csvReader.__next__()
#        curLabel = float(row[1])
#        if(prevLabel < curLabel):
#            print(f"Discontinuity.  Row: {numRows}; Prev:  {prevLabel:.12f}; Cur:  {curLabel: .23f}")
#        prevLabel = curLabel
#        numRows += 1
#    except StopIteration:
#        break
#    
#print(numRows)


import numpy as np


a = np.array([[1]])
print(a.flatten())





