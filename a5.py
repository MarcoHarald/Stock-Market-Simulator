# decomposing a graph curve
import csv
import numpy as np
import random
import matplotlib.pylab as plt


def movAvg(price, gen, target, bound):
    # find local moving average
    # target : today date / date of evaluation
    # bound : how many days back 
    avgPrice = 0
    for day in range(target-bound,target-2):
        avgPrice = price[gen, day]+ avgPrice
    return avgPrice/(bound-1)

#----------------------

# setting variables
simulations = 10
lowerBound = 100
upperBound =  6000

dataMinBound = max(90,lowerBound)
dataMaxBound = min(len(data),upperBound)
#----------------------

csv_file = 'dataBP.csv'
csv_reader = csv.reader(csv_file, delimiter=',')
data = [row for row in csv.reader(open(csv_file))]

# initiate separate arrray
realPrices = np.zeros(len(data))
gssPrice = np.zeros((simulations,len(realPrices)))

# initiate price prediction array
# save values of closing price [4] to separate array
for i in range(0,len(data)-1):
    realPrices[i] = data[i+1][4]

print('Closing price data saved to separate array.')
print('Data points:',len(data))

#---------------------

# intial: assign real prices to all simulations
for gen in range(simulations):
    # within each simulation iterate through all previous days to assign true value
    for i in range(len(realPrices)):
        gssPrice[gen, i] = realPrices[i]

    
#automate creation of moving avg names

movAvgValues = [3,7,30,90]
movAvgTrends = np.zeros((len(movAvgValues),len(realPrices)))
subGlobT = np.zeros((len(movAvgValues),len(realPrices)))

counterPrices = []

for movAvgIndex in range(len(movAvgValues),0):    
    # global trend graph
    for i in range(movAvgValues[movAvgIndex],dataMaxBound):
        # determine moving avg : from first column, starting date, how many days back
        period = movAvgValues[movAvgIndex]  
        movAvgTrends[movAvgIndex][i] = movAvg(gssPrice, 1, i+period, period)
        counterPrices[i]=realPrices[i]
        #FIX check subtraction sequence
 
    for i in range(movAvgValues[movAvgIndex],dataMaxBound):
        subGlobT[movAvgIndex-1][i] = counterPrices[i]-movAvgTrends[movAvgIndex][i]
        
        

for i in range(movAvgValues[movAvgIndex],dataMaxBound):
        subGlobT[3][i] = subGlobT[3][i]-subGlobT[3][i]
    



# plot
t = []
for i in range(lowerBound,upperBound):
    t += [i]

# create vertical axis
yAxis1 = []
for i in range(lowerBound,upperBound):
    yAxis1 += [movAvgTrends[3][i]]

yAxis2 = []
for i in range(lowerBound,upperBound):
    yAxis2 += [realPrices[i]]


plt.plot(t, yAxis1, 'b', label='Quarterly Trend')
plt.plot(t, yAxis2, 'g', label='Real Price')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()

# --------------------

t = []
for i in range(lowerBound,upperBound):
    t += [i]

#original price

yAxis0 = []
for i in range(lowerBound,upperBound):
    yAxis0 += [realPrices[i]]

# create vertical axis
yAxis1 = []
for i in range(lowerBound,upperBound):
    yAxis1 += [subGlobT[2][i]]

yAxis2 = []
for i in range(lowerBound,upperBound):
    yAxis2 += [movAvgTrends[3][i]]


plt.plot(t, yAxis1, 'r', label='Monthly Trend')
plt.plot(t, yAxis1, 'b', label='Weekly Trend')
#plt.plot(t, yAxis0, 'g', label='Real Price')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()
    