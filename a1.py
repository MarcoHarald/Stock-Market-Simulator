import numpy as np
import random
import matplotlib.pylab as plt

drift = 0.1
volatl = 1
xRand = 1.0
length = 51
interval = 50

price = [10]

for t in range(interval):
    # change in share price
    deltaPrice = drift*price[t]+volatl*price[t]*(random.random()-0.5)
    #new share price
    price.append(price[t] + deltaPrice)
    
    
for i in range(interval):
   print (i, price[i])
    
t=[0]    
for i in range(interval):
    t.append(i)

    
plt.plot(t, price)
    
    