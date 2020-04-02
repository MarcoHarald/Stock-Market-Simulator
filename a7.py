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
dataFiles =(glob.glob("data/*"))

# checking files
print('')
print('--- List of all data files: ---')
for i in range(len(dataFiles)):
    print('      ',i,'   ',dataFiles[i])

print('-------------------------------')
print('')


# select data file to open and save
csv_file = dataFiles[0]
data = readCSV(csv_file)

previewFile(data)
# converting dates to computer friendly format

date_time_str = convTime(data[1][0])
print(date_time_str)

