import csv
import numpy as np
import random
import matplotlib.pylab as plt


# -----------FUNCTIONS SETUP-----------

# initialising Geometric Mean
def geoMean(array):
    val = 1.0
    for prices in range(len(array)):
        val = val*(array[prices]/1000)
    val = pow(val*1000, 1 / len(array))
    # print('geometric mean is:  ', val)
    return val


def average(array):
    val = sum(array)/len(array)
    # print('mean is:  ', val)
    return val


# determine mu: price Historical, last known price date, how many days back to calculate value
def findMu(price, today, daysBack, gen):
    mu = price[gen,today]-price[gen,today-daysBack]
    return mu


def movAvg(price, gen, target, bound):
    # find local moving average
    avgPrice = 0
    for day in range(target-bound,target-1):
        avgPrice = price[gen, day]+ avgPrice
    return avgPrice/(bound-1)


# determine sigma: price Historical, last known price date, how many days back to calculate value, mu
def findSigma(price, today, daysBack, mu, gen):
    sigma = 0
    counter = 0
    for day in range(daysBack):
        localDeviation = movAvg(price, gen, day, 10)
        sigma = pow(localDeviation,2)+sigma
    return (pow(sigma/float(daysBack),0.5)/price[gen,5])

# -----------VARIABLE SETUP-----------


# number of scenarios produced
simulations = 10

# days into future prediction
itrtns = 20

# extent of simulation
future = 100

# prediction start date, database extent backwards
today = 200
daysBack = 10

# set bound for moving average operation
avgBound = 10
# set margin for graphing grouping options
margin = 0.99

# correction coefficients
drift0 = 0.1
sigma0 = 0.001

lowerBound = today-daysBack
upperBound = today + future


# -----------DATA SETUP-----------

csv_file = 'dataShell.csv'
csv_reader = csv.reader(csv_file, delimiter=',')
data = [row for row in csv.reader(open(csv_file))]

# initiate separate arrray
realPrices = np.zeros(len(data))
# save values of closing price [4] to separate array
for i in range(0,len(data)-1):
    realPrices[i] = data[i+1][4]

print('Number of data points:', len(realPrices))
print('Closing price data saved to separate array.')
# initiate price prediction array
gssPrice = np.zeros((simulations, len(realPrices)))
localPrice = np.zeros((itrtns+1))

# intial: assign real prices to all simulations
#for gen in range(simulations):
    # within each simulation iterate through all previous days to assign true value
#    for i in range(0,today+daysBack):
#        gssPrice[gen, i] = realPrices[i]

#print('number of data points:',len(data))
#print('drift:',drift,'    sigma:',sigma)


print('-----------PHASE 2-----------')
print('')


# Calculate!
for gen in range(simulations):
    # update with data of the day from which one is calculating.
    gssPrice[gen, today] = realPrices[today]
    print('gen',gen)

    # individual day within each simulation. 'Present/Active day'.
    for activeDay in range(today, today+future):
        drift = drift0
        sigma = sigma0
        # keep track of how many day into the future
        day = activeDay
        # calculate drift and volatility (sigma) on present day
        drift = drift * findMu(gssPrice, day, daysBack, 1)
        sigma = sigma * findSigma(gssPrice, day, daysBack, drift, 1)
        # create starting price for the simulation of the future
        localPrice[0] = realPrices[activeDay] #+ realPrices[activeDay] * random.normalvariate(drift, sigma)


        for t in range(itrtns):
                # new share price calculation   {{ drift * gssPrice[gen, day] }}

                day = activeDay + t
                # determine randomised drift and volatility as simulation iterates in the future
                drift = drift * findMu(gssPrice, day, daysBack, 1)
                sigma = sigma * findSigma(gssPrice, day, daysBack, drift, 1)
                drift = 0.01
                sigma = 0.0075

                # record prediction of price into the future
                localPrice[t + 1] = localPrice[t] + localPrice[t]*random.normalvariate(drift, sigma)

        gssPrice[gen, day + itrtns] = localPrice[itrtns-1]
    gen = +1

# create log of all prices at certain time into simulation
lastPrice = []
for gen in range(simulations):
    lastPrice += [gssPrice[gen, itrtns+today]]

# find geometric mean of all final values
print('geometric mean:',geoMean(lastPrice),'  average:', average(lastPrice))


# find all values of similar result
tier0 = []
tier1 = []
tier2 = []
averagePrediction = average(lastPrice)

for gen in range(simulations):
    tier1 += [gen]

    if averagePrediction*(1 + margin) > gssPrice[gen, itrtns + today] > averagePrediction*(1 - margin):
      tier1 += [gen]
#    elif averagePrediction*(1 + margin) < gssPrice[gen, itrtns + today]:
#        tier0 += [gen]
#    elif averagePrediction*(1 - margin) > gssPrice[gen, itrtns + today]:
#        tier2 += [gen]

print(f'Simulations within {margin} of average prediction {len(tier1)} : {tier1}')

# create log of all prices at certain time into simulation
cPrice = []
for gen in range(simulations):
    cPrice += [gssPrice[gen, itrtns+today-1]]

# calculate moving average for values in selected streams
avgPrices = np.zeros((len(tier1), len(realPrices)))
avgPrice = np.zeros((len(realPrices)))

# PRINT SIMPLE GRAPH
# create time axis
t = []
for i in range(lowerBound,upperBound):
    t += [i]

print('avgBound',avgBound, lowerBound, upperBound)
# create predicted price axis
for gen in tier1:
    avgPrice = []
    counter = 0
    for i in range(lowerBound,upperBound):
        avgPrice[i] = [movAvg(gssPrice, gen, i, avgBound)]
        avgPrices[counter, i-lowerBound] = avgPrice[i-lowerBound]
    counter = counter + 1
    plt.plot(t, avgPrice)

#create actual price axis
realPrice = []
for i in range(lowerBound,upperBound):
    realPrice += [realPrices[i]]
plt.plot(t, realPrice, 'b', label='Real Price')

plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()


# ------------------------

# PRINT SIMPLE GRAPH
# create time axis
t = []
for i in range(lowerBound,upperBound):
    t += [i]

# create predicted price axis
for sims in tier1:
    predPrice0 = []
    for i in range(lowerBound,upperBound):
        predPrice0 += [gssPrice[sims, i]]
    plt.plot(t, predPrice0)

#create actual price axis
realPrice = []
for i in range(lowerBound,upperBound):
    realPrice += [realPrices[i]]
plt.plot(t, realPrice, 'b', label='Real Price')

plt.legend(loc='upper left')
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
plt.show()


# PRINT Quantile Distribution
# create time axis
xAxis = []
for i in range(simulations):
    xAxis += [i]

# create price axis (ordering the prices: increasing magnitude)
orderedPrices = np.sort(lastPrice)

#find geometric mean

plt.plot(xAxis, orderedPrices)
plt.ylabel('Final Price ($)', fontsize=10)
plt.xlabel('# of simulations', fontsize=10)
plt.show()

# PRINT CDF
# create price axis (ordering the prices: increasing magnitude)
cumul = np.zeros((len(orderedPrices)))
for i in range(0, len(orderedPrices)):
    cumul[i] = 1.0 - float(i) / float(simulations)

plt.plot(orderedPrices, cumul)
plt.ylabel('Fraction Above Given Price', fontsize=10)
plt.xlabel('Share Price ($)', fontsize=10)
plt.show()


exit()


# -----------END OF EXECUTION--------------

# PRINT Quantile Distribution
# create time axis
xAxis = []
for i in range(simulations):
    xAxis += [i]

# create price axis (ordering the prices: increasing magnitude)
orderedPrices = np.sort(lastPrice)

#find geometric mean

plt.plot(xAxis, orderedPrices)
plt.ylabel('Final Price ($)', fontsize=10)
plt.xlabel('# of simulations', fontsize=10)
plt.show()

# PRINT CDF
# create price axis (ordering the prices: increasing magnitude)
cumul = np.zeros(len(orderedPrices))
for i in range(0, len(orderedPrices)):
    cumul[i] = 1.0 - float(i) / float(simulations)

plt.plot(orderedPrices, cumul)
plt.ylabel('Fraction Above Given Price', fontsize=10)
plt.xlabel('Share Price ($)', fontsize=10)
plt.show()

#  ------------

day = 3
daysBack = 5
today = 10
mu = 2

localDeviation = realPrices[day] - mu * day - realPrices[today - daysBack]
print('dev:',localDeviation)

# ------------

#quantifying the variables: sigma & drift
stupid = np.array([[13,35,51,19,77],[42,26,40,84,66]])
print('testing arrays in functions',dumb(stupid,1,2))

# ------------

print('randomness:', day,random.normalvariate(0, sigma))
print('sigma:', sigma)
print('drift:', drift)
print('price:', gen, day, gssPrice[gen, day], gssPrice[gen, day + 1])


print('today is:',today)
print('len(realPrices):',len(realPrices))
print('gen is:',gen)
print('test price today',gssPrice[gen,today])
print('actual price today',realPrices[today])

print('length of gssPrice',len(gssPrice-2))


# --------------

# create predicted price axis
predPrice = []
for i in range(0,itrtns+today):
    predPrice += [gssPrice[40, i]]

# create predicted price axis
predPrice2 = []
for i in range(0,itrtns+today):
    predPrice2 += [gssPrice[50, i]]

plt.plot(t, predPrice2, 'g', label='3° Quintile (Predicted Price)')
plt.plot(t, predPrice3, 'y', label='4° Quintile (Predicted Price)')

#finding avg value
for prices in range(len(array)):
    val = val + prices
val = val / len(array)

# --------------

exit()

# --------------

downside = 0
upside = 0
countUpside = 0
countDownside = 0
if price[gen, today] < 1.0:
    print('today price', price[gen, today])
    print('hist price', price[gen, today - day])
    print('gen', gen, 'today', today, 'day', day)
localChange = price[gen, today] - price[gen, today - day]

if localChange > 0:
    upside = localChange * localChange + upside
    countUpside += 1
elif localChange < 0:
    downside = localChange * localChange + downside
    countDownside += 1

weightedChange = pow(upside / max(countUpside, 1), 0.5) - pow(downside / max(countDownside, 1), 0.5)
# print('mu change',weightedChange)
#return weightedChange / price[gen, today - daysBack]





