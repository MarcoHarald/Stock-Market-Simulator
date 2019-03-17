import numpy as np
import random
import matplotlib.pylab as plt


# initialising Geometric Mean
def geoMean(array):
    val = 1.0
    for prices in range(len(array)):
        val = array[prices]
    val = pow(val, 1 / len(array))
    print('geometric mean is:  ', val)

# determine volatility function
def volatility(recentPrices):
    diff = 0
    backlog = 10
    for price in range(len(recentPrices) - backlog, len(recentPrices)):
        diff = diff + abs(recentPrices(price) - recentPrices(price - 1))

# calculate probability of certain margin of loss
def lossProb(startPrices, orderedPrices, fraction):
    i = 0
    while i < len(orderedPrices):
        if orderedPrices[i] > fraction * startPrice:
            break
        i = i + 1
    print('Prob of making a', fraction * 100, '% loss: ', (float(i) / len(orderedPrices)))

# prob of loss
def probLoss(startPrice, orderedPrices)
i = 0
while i < len(orderedPrices):
    if orderedPrices[i] > startPrice:
        break
    i = i + 1
    print('Prob of making a loss: ', (float(i) / len(orderedPrices)))

# find Kelly criterion
def fKelly(mu,sigma):
    fKelly = min(mu / (sigma * sigma), 1)
    print('Kelly criterion is:   ', fKelly)


data = [10,12,15,14,16,15,15,14,13,17,19,18,19,16,18]

itrtns = 50
simulations = 1000

mu = 0.003
sigma = 0.075
drift = 0.001

startPrice = 10

# INITIALISE DATA SET ARRAY
priceHist = np.zeros((simulations, itrtns + 1))

# (np.random.normal(mu, sigma, 1))


# CREATE DATA SETS
for gen in range(simulations):
    priceHist[gen, 0] = (startPrice)

    for t in range(itrtns):
        # change in share price:   deltaPrice
        # new share price iteration
        priceHist[gen, t + 1] = priceHist[gen, t] + drift * priceHist[gen, t] + priceHist[
            gen, t] * random.normalvariate(0, sigma)
    #        print('t is:',t, 'price is:  ',  priceHist[gen, t+1])
    gen = +1

# create log of all prices at certain time into simulation
lastPrice = []
for gen in range(simulations):
    lastPrice += [priceHist[gen, itrtns]]

# variable drift
# drift = priceHist(sim, t)-priceHist(sim, t-5)


# PRINT Quantile Distribution
# create time axis
xAxis = []
for i in range(simulations):
    xAxis += [i]

# create price axis (ordering the prices: increasing magnitude)
orderedPrices = np.sort(lastPrice)

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
plt.ylabel('fraction above', fontsize=10)
plt.xlabel('share price', fontsize=10)
plt.show()

# PRINT SIMPLE GRAPH
# create time axis
t = []
for i in range(itrtns):
    t += [i]

# create price axis
price = []
for i in range(itrtns):
    price += [priceHist[1, i]]

plt.plot(t, price)
plt.ylabel('Price ($)', fontsize=10)
plt.xlabel('Time (days)', fontsize=10)
fraction = 0.9
plt.show()



#print('Volatility')
# expected prices
# expPrice = 0
# for i in simulations:
#    expPrice = expPrice+orderedPrices[i]*(1/len(orderedPrices))
#    i=i+1
# print('Expected price is:',expPrice)

# INITIALISE DATA SET ARRAY
#priceData = np.zeros((simulations, itrtns + 1))
