import numpy as np
import random
import matplotlib.pylab as plt

drift = 0.1

itrtns = 50
simulations = 100

mu = 0
sigma = 0.075

#INITIALISE DATA SET ARRAY
priceHist = np.zeros((simulations, itrtns+1))

#(np.random.normal(mu, sigma, 1))


#CREATE DATA SETS
for gen in range(simulations): 
    priceHist[gen,0] = (10)
    
    for t in range(itrtns):
        # change in share price:   deltaPrice
        #new share price iteration
        priceHist[gen, t+1] = priceHist[gen,t] + drift*priceHist[gen,t]+priceHist[gen,t]*random.gauss(0,sigma)
        print('t is:',t, 'price is:  ',  priceHist[gen, t+1])
    gen =+1
    
    
#create log of all prices at certain time into simulation
lastPrice = []
for gen in range(simulations):
    lastPrice += [priceHist[gen, itrtns]]
    

#PRINT SIMPLE GRAPH
#create time axis    
xAxis=[]
for i in range(simulations):
    xAxis+=[i]

#create price axis (ordering the prices: increasing magnitude)
orderedPrices  = np.sort(lastPrice) 
yAxis = orderedPrices      

    
plt.plot(xAxis, yAxis)


#PRINT SIMPLE GRAPH
#create time axis    
t=[]
for i in range(itrtns):
    t+=[i]

#create price axis
price = []
for i in range(itrtns):
    price+=[priceHist[1,i]]
    
plt.plot(t, price)
    
