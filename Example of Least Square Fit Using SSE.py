# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:49:04 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pdb

experience = np.array([2,3,5,13,8])
salary = np.array([15,28,42,64,50])

exp_mean = np.mean(experience)
print('the mean of experience (x) is' , exp_mean)
salary_mean = np.mean(salary)
print('the mean of alary (y) is', salary_mean)

n = (experience - exp_mean).dot(salary-salary_mean)
d = (experience - exp_mean).dot(experience-exp_mean)

beta = n/d
alpha = salary_mean - beta*exp_mean
print('slop beta= ', beta)
print('intercept alpha =', alpha)

