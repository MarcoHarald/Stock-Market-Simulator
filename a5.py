<<<<<<< HEAD
# IMPROVEMENTS

# oscillation analysis to determine future oscillaiton
# doing this for all trend lines


=======
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee
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
    for day in range(target - bound, target):
        avgPrice = price[gen, day] + avgPrice
    return avgPrice / (bound)

<<<<<<< HEAD
def midMovAvg(price, gen, target, bound):
    # find local moving average
    # target : today date / date of evaluation
    # bound : how many days back
    avgPrice = 0
    for day in range(target-bound, target+bound):
        avgPrice = price[gen, day] + avgPrice
    return avgPrice / (2*bound)


=======
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee
# ----------------------

# setting variables
simulations = 10
lowerBound = 100
upperBound = 600

dataMinBound = max(90, lowerBound)
dataMaxBound = upperBound

# ----------------------

csv_file = 'dataBP.csv'
csv_reader = csv.reader(csv_file, delimiter=',')
data = [row for row in csv.reader(open(csv_file))]

# initiate separate arrray
realPrices = np.zeros(len(data))
counterPrices = np.zeros(len(data))
gssPrice = np.zeros((simulations, len(realPrices)))

# initiate price prediction array
# save values of closing price [4] to separate array
for i in range(0, len(data) - 1):
    realPrices[i] = data[i + 1][4]

print('Closing price data saved to separate array.')
print('Data points:', len(data))

# ---------------------

# intial: assign real prices to all simulations
for gen in range(simulations):
    # within each simulation iterate through all previous days to assign true value
    for i in range(len(realPrices)):
        gssPrice[gen, i] = realPrices[i]

# automate creation of moving avg names

movAvgValues = [3, 7, 30, 90]
movAvgTrends = np.zeros((len(movAvgValues), len(realPrices)))
subGlobT = np.zeros((len(movAvgValues), len(realPrices)))
<<<<<<< HEAD
subGlobT2 = np.zeros((len(movAvgValues), len(realPrices)))
=======
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee

print('CHECK 1')
print(len(movAvgValues))

for movAvgIndex in range(0,len(movAvgValues)):
    # global trend graph
    print('CHECK 2', movAvgIndex)

    for i in range(movAvgValues[movAvgIndex], dataMaxBound-1):
        # determine moving avg : from first column, starting date, how many days back
        period = movAvgValues[movAvgIndex]
<<<<<<< HEAD
        movAvgTrends[movAvgIndex][i] = midMovAvg(gssPrice, 1, i + period, period)
=======
        movAvgTrends[movAvgIndex][i] = movAvg(gssPrice, 1, i + period, period)
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee

        counterPrices[i] = realPrices[i]

    for i in range(movAvgValues[movAvgIndex], dataMaxBound):
        subGlobT[movAvgIndex][i] = counterPrices[i] - movAvgTrends[movAvgIndex][i]

<<<<<<< HEAD
movAvgIndex = 3
for counterIndex in range(movAvgIndex, 0, -1):
    for i in range(movAvgValues[movAvgIndex], dataMaxBound):
        subGlobT2[movAvgIndex][i] = counterPrices[i] - movAvgTrends[counterIndex][i]

# subtracting trends
for i in range(movAvgValues[movAvgIndex], dataMaxBound):
    subGlobT[3][i] = subGlobT[3][i] - subGlobT[2][i]

#for i in range(movAvgValues[movAvgIndex], dataMaxBound):
 #   subGlobT[2][i] = subGlobT[2][i]-subGlobT[1][i]
=======
# subtracting trends
for i in range(movAvgValues[movAvgIndex], dataMaxBound):
    subGlobT[3][i] = subGlobT[3][i] - subGlobT[0][i]
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee

# plot
t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

yAxis0 = []
for i in range(lowerBound, upperBound-1):
    yAxis0 += [realPrices[i]]

# create vertical axis
yAxis1 = []
for i in range(lowerBound, upperBound-1):
    yAxis1 += [movAvgTrends[0][i]]

yAxis2 = []
for i in range(lowerBound, upperBound-1):
    yAxis2 += [movAvgTrends[1][i]]

yAxis3 = []
for i in range(lowerBound, upperBound-1):
    yAxis3 += [movAvgTrends[2][i]]

yAxis4 = []
for i in range(lowerBound, upperBound-1):
    yAxis4 += [movAvgTrends[3][i]]

plt.plot(t, yAxis0, 'black', label='Real Price')
plt.plot(t, yAxis4, 'r', label='Quarterly Trend')
plt.plot(t, yAxis3, 'y', label='Monthly Trend')
plt.plot(t, yAxis2, 'g', label='Weekly Trend')
plt.plot(t, yAxis1, 'b', label='Multi-day Trend')
plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()

# --------------------

t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

# create vertical axis
yAxis1 = []
for i in range(lowerBound, upperBound-1):
    yAxis1 += [subGlobT[0][i]]

yAxis2 = []
for i in range(lowerBound, upperBound-1):
    yAxis2 += [subGlobT[1][i]]

yAxis3 = []
for i in range(lowerBound, upperBound-1):
    yAxis3 += [subGlobT[2][i]]

yAxis4 = []
for i in range(lowerBound, upperBound-1):
    yAxis4 += [subGlobT[3][i]]

<<<<<<< HEAD
yAxis5 = []
for i in range(lowerBound, upperBound-1):
    yAxis5 += [subGlobT[3][i]]

=======
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee
plt.plot(t, yAxis4, 'r', label='Quarterly Trend')
plt.plot(t, yAxis3, 'y', label='Monthly Trend')
plt.plot(t, yAxis2, 'g', label='Weekly Trend')
plt.plot(t, yAxis1, 'b', label='Multi-day Trend')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()



t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

# create vertical axis
yAxis1 = []
for i in range(lowerBound, upperBound-1):
    yAxis1 += [subGlobT[0][i]]

yAxis2 = []
for i in range(lowerBound, upperBound-1):
    yAxis2 += [subGlobT[1][i]]

yAxis3 = []
for i in range(lowerBound, upperBound-1):
    yAxis3 += [subGlobT[2][i]]

yAxis4 = []
for i in range(lowerBound, upperBound-1):
    yAxis4 += [subGlobT[3][i]]

#plt.plot(t, yAxis4, 'r', label='Quarterly Trend')
#plt.plot(t, yAxis3, 'y', label='Monthly Trend')
plt.plot(t, yAxis2, 'g', label='Weekly Trend')
plt.plot(t, yAxis1, 'b', label='Multi-day Trend')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()


<<<<<<< HEAD
t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

# create vertical axis
yAxis1 = []
for i in range(lowerBound, upperBound-1):
    yAxis1 += [subGlobT[0][i]]

yAxis2 = []
for i in range(lowerBound, upperBound-1):
    yAxis2 += [subGlobT[1][i]]

yAxis3 = []
for i in range(lowerBound, upperBound-1):
    yAxis3 += [subGlobT2[0][i]]

yAxis4 = []
for i in range(lowerBound, upperBound-1):
    yAxis4 += [subGlobT[3][i]]

#plt.plot(t, yAxis4, 'r', label='Quarterly Trend')
#plt.plot(t, yAxis3, 'y', label='Monthly Trend')
plt.plot(t, yAxis2, 'g', label='Weekly Trend')
plt.plot(t, yAxis1, 'b', label='Multi-day Trend')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()



=======
>>>>>>> b2ea9d2de7f44e327bfb03c48edd891cfa3756ee











