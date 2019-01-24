



import csv


inputFile = open("train.csv", "r", newline='')
csvReader = csv.reader(inputFile)

numRows = 0
while True:
    try:
        csvReader.__next__()
        numRows += 1
    except StopIteration:
        break
    
print(numRows)



