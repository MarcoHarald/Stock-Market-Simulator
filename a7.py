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





