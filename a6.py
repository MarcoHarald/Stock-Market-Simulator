# IMPROVEMENTS

# oscillation analysis to determine future oscillaiton
# doing this for all trend lines


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

def midMovAvg(price, gen, target, bound):
    # find local moving average
    # target : today date / date of evaluation
    # bound : how many days back
    avgPrice = 0
    for day in range(target-bound, target+bound):
        avgPrice = price[gen, day] + avgPrice
    return avgPrice / (2*bound)

# porfolio management

def makeTransaction(stock, day, cash, shares, action):
    if action == -1:
        cash = cash + shares*stock[day]
        shares = 0
    #   print('sold.', shares)
    if action == 1:
        shares = shares + cash/stock[day]
        cash = 0
     #   print('bought.', shares)
    portfolio[0] = cash
    portfolio[1] = shares


# ----------------------
# select data file to open and save
csv_file = 'dataENI.csv'
csv_reader = csv.reader(csv_file, delimiter=',')
data = [row for row in csv.reader(open(csv_file))]

# ----------------------

# setting variables
simulations = 10
#movAvgValues = [10, 20, 30, 40, 50, 60, 74, 90, 100, 110, 120, 130, 140]

movAvgValues = []
optimalInvestPeriod = 50
numberMovAvg = 20

stepSize = optimalInvestPeriod / numberMovAvg

for i in range(1,numberMovAvg):
    movAvgValues += [int(stepSize*i*2)]

#movAvgValues = [[i] for i in A]


# setting the risk profiles to be investigated: nunber of moving averages that need to be crossed
numberMovAvg = len(movAvgValues)
riskBound = [-int(numberMovAvg*2/5),-int(numberMovAvg*1/5),0,int(numberMovAvg*1/5),int(numberMovAvg*2/5)]


# automated creation of risk profiles. riskBound is summed to number[half of the MAs].
# higher number will allow action to be taken with fewer MA's crossing the mid MA level
# lower number will force program to wait until certain nunmber of MA's have crossed the middle MA

riskBound = []
number = 5
for i in range(0,number):
    riskBound += [int(numberMovAvg*i/(number+1)/2)]

# data range to be investigated
lowerBound = int(max(movAvgValues))
upperBound = int(len(data)-lowerBound)

dataMinBound = int(max(optimalInvestPeriod*2, lowerBound))
dataMaxBound = int(upperBound)


# price shift pattern : array collating the revenue from a certain investment period
maxInvestLength = min(upperBound-lowerBound,optimalInvestPeriod*4)
startDate = lowerBound
endDate = upperBound-maxInvestLength-1
periodLength = endDate-startDate

# setting portfolio values
startingCash = data[lowerBound][4]
startingShares = 0
portfolio = [startingCash, startingShares]

# riskBound: used to raise or lower the response barrier (if higher, more mov avg have to cross before action)

riskCounter = 0
# ----------------------


# initiate separate arrray
realPrices = np.zeros(len(data))
counterPrices = np.zeros(len(data))
gssPrice = np.zeros((simulations, len(realPrices)))
priceShiftPattern = np.zeros(len(data))
portfolioHist = np.zeros((len(riskBound),len(data)))

# automate creation of moving avg names
movAvgTrends = np.zeros((len(movAvgValues), len(realPrices)))
subGlobT = np.zeros((len(movAvgValues), len(realPrices)))
scoreMA = np.zeros(len(realPrices))
subGlobT2 = np.zeros((len(movAvgValues), len(realPrices)))


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


print('# of mov. averages', len(movAvgValues))

for movAvgIndex in range(0,len(movAvgValues)):
    # global trend graph

    for i in range(movAvgValues[movAvgIndex], dataMaxBound-1):
        # determine moving avg : from first column, starting date, how many days back
        period = movAvgValues[movAvgIndex]
        #movAvgTrends[movAvgIndex][i] = midMovAvg(gssPrice, 1, i + period, period)
        movAvgTrends[movAvgIndex][i] = movAvg(gssPrice, 1, i + period, period)

        counterPrices[i] = realPrices[i]


print('finished calculating moving averages.')


# -----------------------------------

# find moving global prices without local oscillation

#netDaily = counterPrices[i] - movAvgTrends[3][i]
#subGlobT3[3][i] = counterPrices[i]-

# -----------------------------------
print('length of mov avg values',len(movAvgValues))
for movAvgIndex in range(0,len(movAvgValues)):
    for i in range(movAvgValues[movAvgIndex], dataMaxBound):
        subGlobT[movAvgIndex][i] = movAvgTrends[movAvgIndex][i] # counterPrices[i] -

# finding the difference between the midpoint of the Moving averages and the MA of interest
# will be used to determine which direction the stock will shift

midMovAvg = int(len(movAvgValues)/2)
print('mid Moving Average', midMovAvg)

for ident in range(len(movAvgValues)):
    for i in range(movAvgValues[len(movAvgValues)-1], dataMaxBound):
        subGlobT[ident][i] = subGlobT[ident][i]-subGlobT[midMovAvg][i]

# sum all MA scores at a point in time to give the stock a prediction score
# if MA length is shorter than midMA then +1 & if MA length is longer than midMA then -1
# when score is +ve BUY & when score is -VE sell
# if MA is above line of midMA, then score is assigned & if below score is subtracted

# for MA length shorter than midMA

longMA = []
shortMA = []

#splitting the MA into two categories
for i in range(0,len(movAvgValues)):
    if i < midMovAvg:
        shortMA += [i]
    elif i > midMovAvg:
        longMA += [i]

print('long MA index',longMA)
print('short MA index',shortMA)

# summing score from the MA along time
for i in range(lowerBound,upperBound):
#take into account longMA
    for index in longMA:
        if subGlobT[index][i] < 0:
            scoreMA[i] += -1
        else:
            scoreMA[i] += +1

    for index in shortMA:
        if subGlobT[index][i] > 0:
            scoreMA[i] += -1
        else:
            scoreMA[i] += +1
print('finished scoring MA.')


# price shift pattern : array collating the revenue from a certain investment period

for day in range(startDate, endDate):
    for period in range(1, maxInvestLength):
       # print(day,period,priceShiftPattern[period])
        priceShiftPattern[period] = priceShiftPattern[period] + (realPrices[day+period]-realPrices[day])

print('midway evaluating optimal investment period.')


# determine an average by dividing by how many inputs. Then considering avg return per day that money is left invested

for period in range(1, maxInvestLength):
    priceShiftPattern[period] = priceShiftPattern[period]/periodLength/period

print('finished evaluating optimal investment period.')

# log with all the times transaction occured.
transactionHist = []
riskCounter = 0

for risk in riskBound:
#setting starting conditions for all trials
    cash = realPrices[lowerBound]
    shares = 0

    portfolio = [cash, shares]

    transactionHist += [[[],[],[],[]]]


    for i in range(lowerBound+1, upperBound):
        # buying shares
        if scoreMA[i] > midMovAvg+risk:
            if scoreMA[i-1] <= midMovAvg+risk:
                makeTransaction(realPrices,i,portfolio[0],portfolio[1],1)
                #saving any transaction made (recorded after shares are set to correct value)
                transactionHist[riskCounter][0] += [i]
                transactionHist[riskCounter][1] += [portfolio[1]]
                transactionHist[riskCounter][2] += [0]
                transactionHist[riskCounter][3] += [realPrices[i]]

        # selling shares
        if scoreMA[i] < midMovAvg-risk:
            if scoreMA[i-1] >= midMovAvg-risk:

                #saving any transaction made (recorded before shares are set to zero)
                transactionHist[riskCounter][0] += [i]
                transactionHist[riskCounter][1] += [portfolio[1]]
                transactionHist[riskCounter][2] += [0]
                transactionHist[riskCounter][3] += [realPrices[i]]

                makeTransaction(realPrices, i, portfolio[0], portfolio[1],-1)

        #logging any changes


        portfolioHist[riskCounter][i] = portfolio[0]+portfolio[1]*realPrices[i]
    riskCounter += 1
     #   print('day:', i, '    cash:', portfolio[0], '    shares:', portfolio[1])

print(len(transactionHist[1][2]))

print('finished estimating portfolio value.')
print('finished calculations. Now printing graphs.')

# plot ------------------------------------------------------------------------------------------------
print('done.')


riskRun = 0
dateColumn = 0
shareBuyColumn = 1
shareSellColumn = 2
sharePriceColumn = 3

t0 = []
t1 = []
t2 = []
t3 = []
t4 = []

yAxis0 = []
yAxis1 = []
yAxis2 = []
yAxis3 = []
yAxis4 = []

for i in range(1, len(transactionHist[riskRun][dateColumn])-1, 2):
    deltaDays = [transactionHist[riskRun][dateColumn][i]-transactionHist[riskRun][dateColumn][i-1]]
    t0 += deltaDays
    yAxis0 += [transactionHist[riskRun][sharePriceColumn][i]-transactionHist[riskRun][sharePriceColumn][i-1]]

print(riskRun,len(t0))

riskRun = 1

for i in range(1, len(transactionHist[riskRun][dateColumn])-1, 2):
    deltaDays = [transactionHist[riskRun][dateColumn][i]-transactionHist[riskRun][dateColumn][i-1]]
    t1 += deltaDays
    yAxis1 += [transactionHist[riskRun][sharePriceColumn][i]-transactionHist[riskRun][sharePriceColumn][i-1]]

print(riskRun,len(t1))

riskRun = 2

for i in range(1, len(transactionHist[riskRun][dateColumn])-1, 2):
    deltaDays = [transactionHist[riskRun][dateColumn][i]-transactionHist[riskRun][dateColumn][i-1]]
    t2 += deltaDays
    yAxis2 += [transactionHist[riskRun][sharePriceColumn][i]-transactionHist[riskRun][sharePriceColumn][i-1]]

print(riskRun,len(t2))

riskRun = 3

for i in range(1, len(transactionHist[riskRun][dateColumn])-1, 2):
    deltaDays = [transactionHist[riskRun][dateColumn][i]-transactionHist[riskRun][dateColumn][i-1]]
    t3 += deltaDays
    yAxis3 += [transactionHist[riskRun][sharePriceColumn][i]-transactionHist[riskRun][sharePriceColumn][i-1]]

print(riskRun,len(t3))


riskRun = 4

for i in range(1, len(transactionHist[riskRun][dateColumn])-1, 2):
    deltaDays = [transactionHist[riskRun][dateColumn][i]-transactionHist[riskRun][dateColumn][i-1]]
    t4 += deltaDays
    yAxis4 += [transactionHist[riskRun][sharePriceColumn][i]-transactionHist[riskRun][sharePriceColumn][i-1]]


print(riskRun,len(t4))

# return per day on investment, analysed for varying investment period lengths (based on transactions)
#
plt.scatter(t0, yAxis0, c='red', label='Portfolio A')
plt.scatter(t1, yAxis1, c='yellow',label='Portfolio B')
plt.scatter(t2, yAxis2, c='green',label='Portfolio C')
plt.scatter(t3, yAxis3, c='blue',label='Portfolio D')
plt.scatter(t4, yAxis4, c='black',label='Portfolio E')
plt.legend(loc='upper right')
plt.ylabel('Return per day ($/day)', fontsize=12)
plt.xlabel('Investment Length (days)', fontsize=12)

ax2 = plt.twinx()
# --------

# plot
t = []
for i in range(1,maxInvestLength):
    t += [i]

yAxis0 = []
for i in range(1,maxInvestLength):
    yAxis0 += [priceShiftPattern[i]]

ax2.plot(t, yAxis0, 'black', label='Return per day')

# --------


#plt.imshow(t, yAxis0, cmap='hot', interpolation='nearest')

plt.legend(loc='lower left')
plt.ylabel('Return per day ($/day)', fontsize=12)
plt.title('Evaluating MA Method Investment Length')
plt.show()


# plot ------------------------------------------------------------------------------------------------
alphabet = ['a','b','c','d','e','f','g','h','i','j']
for i in range(0,len(riskBound)):
    print(alphabet[i], riskBound[i], (100*(riskBound[i]+0.5*len(movAvgValues))/len(movAvgValues), '%'))

t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

yAxis0 = []
for i in range(lowerBound, upperBound-1):
    yAxis0 += [realPrices[i]]

# create vertical axis
yAxis1 = []
for i in range(lowerBound, upperBound-1):
    yAxis1 += [portfolioHist[0][i]]

yAxis2 = []
for i in range(lowerBound, upperBound-1):
    yAxis2 += [portfolioHist[1][i]]

yAxis3 = []
for i in range(lowerBound, upperBound-1):
    yAxis3 += [portfolioHist[2][i]]

yAxis4 = []
for i in range(lowerBound, upperBound-1):
    yAxis4 += [portfolioHist[3][i]]

yAxis5 = []
for i in range(lowerBound, upperBound-1):
    yAxis5 += [portfolioHist[4][i]]

#(action after 70% of MAs cross)
plt.plot(t, yAxis0, 'black', label='Real Price')
plt.plot(t, yAxis5, 'red', label='Portfolio E')
plt.plot(t, yAxis4, 'yellow', label='Portfolio D')
plt.plot(t, yAxis3, 'green', label='Portfolio C')
plt.plot(t, yAxis2, 'blue', label='Portfolio B')
plt.plot(t, yAxis1, 'purple', label='Portfolio A')
plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=12)
plt.xlabel('Time (days)', fontsize=12)
plt.title('Performance of Portfolios with different risk factors')
plt.show()


# --------------------

# plot
t = []
for i in range(1,maxInvestLength):
    t += [i]

yAxis0 = []
for i in range(1,maxInvestLength):
    yAxis0 += [priceShiftPattern[i]]

plt.plot(t, yAxis0, 'black')
plt.legend(loc='upper left')
plt.ylabel('Return ($)', fontsize=12)
plt.xlabel('Investment Time Period (days)', fontsize=12)
plt.show()


# --------------------

# plot
t = []
for i in range(lowerBound,upperBound):
    t += [i]

yAxis0 = []
for i in range(lowerBound,upperBound):
    yAxis0 += [scoreMA[i]]

yAxis1 = []
for i in range(lowerBound, upperBound):
    yAxis1 += [200*(realPrices[i+1]/realPrices[i]-1)]

plt.plot(t, yAxis1, 'red', label='Real Price')
plt.plot(t, yAxis0, 'blue', label='Mov.Avg. Score')
plt.legend(loc='upper left')
plt.ylabel('Moving Average Score', fontsize=12)
plt.xlabel('Investment Time Period (days)', fontsize=12)
plt.show()


# --------------------

t = []
for i in range(lowerBound, upperBound-1):
    t += [i]

# create longMA
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

yAxis5 = []
for i in range(lowerBound, upperBound-1):
    yAxis5 += [subGlobT[4][i]]

yAxis6 = []
for i in range(lowerBound, upperBound-1):
    yAxis6 += [subGlobT[5][i]]

# create short MA

yAxis8 = []
for i in range(lowerBound, upperBound - 1):
    yAxis8 += [subGlobT[7][i]]

yAxis9 = []
for i in range(lowerBound, upperBound - 1):
    yAxis9 += [subGlobT[8][i]]

yAxis10 = []
for i in range(lowerBound, upperBound - 1):
    yAxis10 += [subGlobT[9][i]]

yAxis11 = []
for i in range(lowerBound, upperBound - 1):
    yAxis11 += [subGlobT[10][i]]

yAxis12 = []
for i in range(lowerBound, upperBound - 1):
    yAxis12 += [subGlobT[11][i]]

yAxis13 = []
for i in range(lowerBound, upperBound - 1):
    yAxis13 += [subGlobT[12][i]]

# create mid MA

yAxis7 = []
for i in range(lowerBound, upperBound - 1):
    yAxis7 += [subGlobT[6][i]]

plt.plot(t, yAxis8,  'r', label='Short Moving Average')
plt.plot(t, yAxis9,  'b', label='Short Moving Average')
plt.plot(t, yAxis10, 'g', label='Short Moving Average')
plt.plot(t, yAxis11, 'p', label='Short Moving Average')
plt.plot(t, yAxis12, 'b', label='Short Moving Average')
plt.plot(t, yAxis13, 'y', label='Short Moving Average')

#plt.plot(t, yAxis1, 'y', label='Long Moving Average')
#plt.plot(t, yAxis2, 'b', label='Long Moving Average')
#plt.plot(t, yAxis3, 'r', label='Long Moving Average')
#plt.plot(t, yAxis4, 'g', label='Long Moving Average')
#plt.plot(t, yAxis5, 'p', label='Long Moving Average')
#plt.plot(t, yAxis6, 'v', label='Long Moving Average')

plt.plot(t, yAxis7, 'green', label='Middle Moving Average')


plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()

#--------------------------------------------------------

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


# plot ------------------------------------------------------------------------------------------------
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









