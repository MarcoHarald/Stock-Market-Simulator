# decomposing a graph curve
import csv
import numpy as np
import random
import matplotlib.pylab as plt
import datetime


# ----functions----

def convTime(data):
         #date_time_str = data[1][0][:10]+' '+data[1][0][11:] 
    date_time_str = data
    date_time_obj = datetime.datetime.strptime(date_time_str, '%b %d, %Y')
    return date_time_obj


# select data file to open and save
def readCSV(filepath):
    csv_file = filepath
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = [row for row in csv.reader(open(csv_file))]
    return data

# preview first few lines of data file
def previewFile(data):
    print('---Previewing file---')
    for i in range(3):
        print('Row',i,'    ', data[i])
    print('--- --- ---')
    print()
    return data

def movAvg(price, gen, target, bound):
    # find local moving average
    # target : today date / date of evaluation
    # bound : how many days back
    avgPrice = 0
    for day in range(target - bound, target):
        avgPrice = price[gen, day] + avgPrice
    return avgPrice / (bound)

def midMovAvg(price, gen, target, bound):
    # find local moving average
    # target : today date / date of evaluation
    # bound : how many days back
    avgPrice = 0
    for day in range(target-bound, target+bound):
        avgPrice = price[gen, day] + avgPrice
    return avgPrice / (2*bound)


# transpose input data array
def transposeArr(data):
    outArr = []
    
    # setup new blank
    for i in range(len(data[0])):
        outArr += [[]]
    
    # copy values over in label-columns instead of timestamp-rows
    for i in range(len(data)):
        # loop through new columns
        for j in range(len(outArr)):
            outArr[j] += [data[i][j]]

    return outArr
 
# --end functions--


# fetching data files
import glob
dataFiles =(glob.glob("data/* Historical Data.csv"))

# checking files
print('')
print('--- List of all data files: ---')
for i in range(len(dataFiles)):
    print('      ',i,'   ',dataFiles[i])

print('-------------------------------')
print('')

arr1 = []
# select data file to open and save
for i in range(len(dataFiles)): 
    arr1 += [readCSV(dataFiles[i])]
    previewFile(arr1[i])

    # converting dates to computer friendly format
    for j in range(1,len(arr1[i])):
        arr1[i][j][0] = convTime(arr1[i][j][0])
        
# convert array setup, loop through all data sets
arr2 = []
for i in range(len(arr1)):
    data = arr1[1]
    arr2 += [transposeArr(data)]

# print(len(arr2),len(arr2[1]))    # checkTEST

# convert to floats 
targetCols =  [1,2,3,4,5]

# loop through file sources
for k in range(5,len(arr2)):
    # loop through target cols
    for i in targetCols:
        # loop through relevant values
        for j in range(1,len(arr2[k][i])):
            #print(i,j,k)
            b =(arr2[k][i][j])

            if '-' in arr2[k][i][j]:
                try:
                    arr2[k][i][j] = 0
            else:
                arr2[k][i][j] = float(arr2[k][i][j].replace(',', '').replace("K",""))
            
            print(b,arr2[k][i][j])

                
targetStock = 1

data = arr1[targetStock]
samplingLength = 1     # how large the sampling frequency will be
# calculate % change for series

# append column with label
arr2[targetStock] += ['%d -day Return' % (samplingLength) ]
intervalSize  = 3
targetCol  = 2
outCol = 7

outData = []
# add blank spaces where change cant be calc
for i in range(intervalSize):
    outData += ['']
    
data = arr2[targetStock][targetCol]
print(arr2[targetStock][targetCol])
# loop through all timestamps and append value
for i in range(intervalSize,len(data)):
    # calc percentage change
    outData += [float(data[i])/float(data[i-intervalSize])]
    
print(outData)
    
    
       
    


# plotting values
# plot ------------------------------------------------------------------------------------------------

targetStocks = [0,4]
targetDataTypes = [1]

for j in targetStocks:
    targetStock = j
    
    for targetDataType in targetDataTypes:
        # target specific stock data
        targData = arr1[targetStock]
        
        t = []
        ax = []
        
        # make a plot line: loop through a single data type for a single stock
        for i in range(1,len(targData)):

            if not '-' in targData[i][targetDataType]:            
               #print('A1',j, i,targData[i][0],targData[i][targetDataType].replace(',', '').replace("-"," "))
                
                t += [targData[i][0]]
                ax += [float(targData[i][targetDataType].replace(',', '').replace("K",""))]
        
        # set data type & stock names
        plt.plot(t, ax, label=dataFiles[targetStock][5:8]+'_'+targData[0][targetDataType])

plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()





