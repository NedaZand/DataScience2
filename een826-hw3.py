# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

#%% Q1 Matrix multiplication 



A = np.array([[2, 8, 4],[5, 4, 2]])
B = np.array([[4, 1],[6, 4],[5, 3]])
C = np.array([[4, 1, 2],[6, 4, 3],[5, 3, 4]])
D = np.array([[4, 1, 2],[6, 4, 3]])

print('A is\n' , A)
print('B is \n', B)
print(' A .  B\n')

try:
    print(np.dot(A, B))
except ValueError:
    print('A and B are not multiplicable because their dimention are not match the number of culomn of the first matrix should be equel to the number of rows in the second matrix')

print('A is\n' , A)
print('C is \n', C)
print('A .  C\n')
try: 
    print(np.dot(A, C))
except ValueError:
    print('A and B are not multiplicable because their dimention are not match the number of culomns of the first matrix should be equel to the number of rows in the second matrix')

print('A is\n' , A)
print('D is \n', D)
print(' A .  D\n')
try:
    print(np.dot(A, D))
except ValueError:
    print('A and B are not multiplicable because their dimention are not match the number of culomn of the first matrix should be equel to the number of rows in the second matrix')

#%% Q2

import numpy as np
x = np.array([50, 68, 74, 70, 65, 61, 63, 74, 62])
y = np.array([170, 193, 209, 185, 195, 188, 188, 202, 183])

print('question2 a\n')

plt.scatter(x, y)
plt.show()

#question2 b

def zscore(array):
    return (array - np.mean(array))/np.std(array)

print('question2 c\n')
zscoreX = zscore(x)
zscoreY = zscore(y)
plt.scatter(zscoreX, zscoreY)
plt.show()

print('question2 d\n')


ownOutput = (np.dot(zscoreX, zscoreY) / len(x))
print('corrilation coefficient with my own output')
print(ownOutput)

numpyOutput = np.corrcoef(x, y)
print('corrilation coefficient with numpy functions')
print(numpyOutput)
print('yes they are same')

#%% Q3
x = np.array([ 50, 68, 74, 70, 65, 61, 63, 74, 62, 80])
y = np.array([170, 193, 209, 185, 195, 188, 188, 202, 183, 1000])

print('question3 a')

plt.scatter(x, y)
plt.show()

print('question3 b')

print('Pearson correlation coefficient using the corrcoef function')
pearsonCorrcoef = np.corrcoef(x, y)[0,1]
print(pearsonCorrcoef)

print('question3 c')

print('Spearman rank correlation coefficient using the corrcoef function')
rankX = np.argsort(np.argsort(x))
rankY = np.argsort(np.argsort(y))
spearmanCorrcoef = np.corrcoef(rankX, rankX)[0,1]
print(spearmanCorrcoef)
print('\n')
print('question3 d\n')

print('Pearson is sensitive to outliers as you can see from plots there is actually high correlation between variables but the existence of outlier will reduce the'
'correlation . Here mention linear relationship instead of relationship'
'The reason is that the definition is based on mean and std and outliers will'
'change these metrics dramatically.'
'But spearman is ranked based and it converts the values to ranks so even a'
'very large values won’t affect the final result')

print('question3 e\n')
print('\n')
print('Value -1 shows an inverse relationship')
print('Value 0 means there is no relationship between two variables')
print('Value +1 means two varieties have strong correlation')



#%% Q4

print('question4 a')
print('\n')
normCdf = norm.cdf(0.5)
print('normCdf')



normCdf2 = 2*(0.5-norm.cdf(-0.5))
print(normCdf2)


#%% Q5

print(" X’ = aX + b; X’ ~ N(aµ+b, a^2 σ^2")

print(' if X ~ N(µ, σ^2), then Z = (X- µ)/σ ~ N(0,1)')

print(' normal distribution:')
print(norm.cdf(7,5,3)-norm.cdf(2,5,3))

print(' standard normal distribution')
print(norm.cdf(1.5)-norm.cdf(-1.5))
#%%Question6

print('question6 a\n')
print('P(A ∪ B) = P(A) + P(B) - P(A ∩ B)')
print('question6 b\n')
print('P(A|B) = P(A ∩ B) / P(B)')
print('question6 c\n')
print('P(A ∩ B) = P(A | B) * P(B)')
print('question6 d\n')
print('P(A ∩ B) = P(A) * P(B)')

#%% Q7

print("question7")
print("(P(d = even and d < 5) = 1/4) == (P(d = even) * P(d < 5) = 1/4), so it is independent ")
print('\n')

#%% Q8

print('question8')
print('P(6666 | loaded) * P(loaded) / P(6666) = 0.5^4 * 0.05 / (0.5 * 0.05 + (1/6)^4 * 0.95)')
print( (0.5**4 * 0.05) / (0.5 * 0.05 + ((1/6)**4 )* 0.95))

#%%
print('question 9')

print('p(A|B)=?')
b=(0.990*(10)**-3)+0.02*(1-((10)**(-3)))
print('p(B)=(0.990*(10)**-3)+0.02*(1-((10)**(-3)))')
print(0.999*(10**(-3))/(0.990*(10)**-3)+0.02*(1-((10)**(-3))) )
print('question9 b')
print((1-0.02)*(1-((10)**(-3)))/(1-b))

#%% Q10
# a. plot annual total measles cases in each year
#%% load data
import pandas as pd 
measles=pd.read_csv('Measles.csv',header=None).values
mumps=pd.read_csv('Mumps.csv',header=None).values
chickenPox=pd.read_csv('chickenPox.csv',header=None).values

# close all existing floating figures
plt.close('all')
print('question10 a')

plt.figure()
plt.title('Fig 1: NYC measles cases')


measlesYear=measles[:,0]  
measlesMonth=measles[:,1:] 
MeasleToYear=measlesMonth.sum(axis=1)  

plt.plot(measlesYear,MeasleToYear, marker='*', linestyle='solid')
plt.ylabel("Number of cases")
plt.xlabel("Year")
plt.show()




#%% Q10.b 

#%% b. bar plot average mumps cases for each month of the year
print('question10 b')
plt.figure()
plt.title('b. Average monthly mumps cases')

avgMonthMumps=measlesMonth.mean(axis=0)
#print(avgMonthMumps)
axisX = np.arange(1,13) 
plt.bar(axisX, avgMonthMumps)
plt.ylabel("Average number of cases")
plt.xlabel("Month")


# 2. Your code is here

plt.show()


#%% Q10.c scatter plot monthly mumps cases against measles cases
mumpsCases = mumps[:, 1:].reshape(41*12) 
measlesCases = measles[:, 1:].reshape(41*12)

plt.figure()
plt.title('Fig 4: Monthly mumps vs measles cases')
plt.scatter(mumpsCases, measlesCases)
plt.xlabel("Number of Mumps Cases")
plt.ylabel("Number of Measles Cases")
plt.show()




#%% Q10.d plot monthly mumps cases against measles cases in log scale
print('question10 d')
plt.figure()
plt.loglog(mumpsCases, measlesCases,'o', )
plt.xlabel("Number of Mumps Cases")
plt.ylabel("Number of Measles Cases")
plt.show()




#%% Answer to Q10.e

print('question10 e')

# your code is here
str = """Linear scaling plot over x and y shows diagram on x and y with equal interval . But
sometimes changes are exponentially dependent on one variable so if we use
standard way we should use wide range and changes for small values will not be
showing in proper way . In this case we can use logarithm scale which compact the
range."""

print(str)