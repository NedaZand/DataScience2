#%% import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pdb

#%% Q1a
#Your code is here

# x=[2005,2006,2007,2008,2009]
# xReshape=np.array(x).reshape(-1,1)

# y=[12000000,19000000,29000000,37000000,45000000]
# yReshape=np.array(y)

# LinearR= linear_model.LinearRegression()
# LinearR.fit(xReshape,yReshape)
# pred= LinearR.predict(xReshape)
# #print(pred)
# print('beta')
# print(LinearR.coef_)
# print('alpha')
# print( LinearR.intercept_)

X = np.array([2005,2006,2007,2008,2009])
Y = np.array([12000000,19000000,29000000,37000000,45000000])
#if graph is needed code is here
plt.scatter(X, Y)

model = linear_model.LinearRegression()
model.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
be = model.coef_
al = model.intercept_
print('beta')
print('beta = %.2f' % model.coef_)
print('\n\nalpha')
print( 'beta = %.2f' % model.intercept_)
# y=al+(be.X)
# plt.plot(X, y)

m, b = np.polyfit(X, Y, 1)
print(m,b)
plt.plot(X, m*X + b ,'r-')
plt.xlabel('year')
plt.ylabel('price')
plt.title('Q1a: ordinary least square')
plt.show()

#%%Q1b
#Your code is here
xtest=np.array([2012]).reshape(-1,1)

pred1= model.predict(xtest)
print("\n\nsales of the company in 2012 ",pred1)
#%% Q2 setup
import scipy.stats as stats
import numpy.random as rn
#%% Q2a
rn.seed(0)
N = 16;
m = 10**5;
# 10**5 by 16 = 16 + 16 + ... + 16
# use the uniform or binomial functions to generate the data

nHeads = (rn.uniform(size=(m, N))>0.5).sum(1)


#%% Q2a plot the histgram
# Your code is here
plt.hist(nHeads, bins=range(18))
plt.xlabel('nHeads')
plt.ylabel('Number of coins')
plt.title('Q2a: Histogram')
plt.show()

#%% Q2b
# using plt.hist with parameter and density
#Your code is here
numOfBars=plt.hist(nHeads, bins=range(18), density=True)
plt.xlabel('k')
plt.ylabel('P(nHeads = k)')
plt.title('Q2b: Probability Mass Function')
plt.show()


#%% Q2c

#You can calculate the probabilities/counts with your own code or using values returned from 2a/2b.
# Your code is here


cdf = np.cumsum(numOfBars[0])
plt.plot(numOfBars[1][:-1], cdf)
plt.xlabel('k')
plt.ylabel('P(nHeads <= k)')
plt.title('Q2c: Cumulative Distribution Function')
plt.show()

#this is another way to do this part. please take in consider
prob=[np.equal(nHeads,i).mean() for i in range(16)]
Tcdf=[]
Tcdf.append(prob[0])
for i in range(len(prob)-1):
	 value=cdf[i]+prob[i+1]
	 Tcdf.append(value)
plt.plot(Tcdf)
plt.xlabel('k')
plt.ylabel('P(nHeads <= k)')
plt.title('Q2c: Cumulative Distribution Function second solution')
plt.show()
#%% Q2d
# Use the binomial distribution CDF (use scipy.stats.binom.cdf)

# Your code is here

# scatter plot
# Your code is here
import scipy
import matplotlib.pyplot as plt
def Ecdf():
    p = 0.5
    n = 16
    i = 0
    output = []
    for a in range(16):
        output.append(scipy.stats.binom.cdf(i, n, p))
        i += 1
    return output

print('\n\nbinomial',Ecdf())
plt.scatter(Tcdf,Ecdf())
plt.xlabel('Empirical CDF')
plt.ylabel('Theoritical CDF')
plt.title('Q2d: scatter plot')

plt.show()

# line plot
#Your code is here
plt.plot(Tcdf,'bo-')
plt.plot(Ecdf(),'rd:')
plt.xlabel('k')
plt.ylabel('CDF')
plt.title('Q2d: line plot')
plt.legend(['Empirical CDF','Theoritical CDF'])

plt.show()

# Loglog scale plot
#your code is here
plt.loglog(Tcdf , Ecdf(),'o')
plt.xlabel('Empirical CDF')
plt.ylabel('Theoritical CDF')
plt.title('Q2d: loglog plot')
plt.show()




