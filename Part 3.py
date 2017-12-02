#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:32:36 2017

@author: chenyingying
"""

# power analysis-Regression VS ANOVA

import pandas as pd 
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2

# Customer Function for null hypothesis 
# refer to the course material 
def null(p,obs):
    B0=p[0]
    sigma=p[1]
    
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs).sum()
    return nll

# Customer Function for alternative hypothesis_significant effect between x and y
# refer to the course material 
def nllike(p,obs1,obs2):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs1
    nll=-1*norm(expected,sigma).logpdf(obs2).sum()
    return nll     

# Customer Function for getting mean of p-value from regression design by monte carlo approach

def regression(slope,intercept,sigma,iteration,obs_n):
    p_regression_sum=0
    for i in range(iteration):
        # generate random floats in the range of 0-50, total number=obs_n
        x=np.random.uniform(0,50,obs_n)
        # generate the y with standard deviation sigma
        y=slope*x+intercept
        y=y+np.random.randn(24)*sigma
        # get the likelihood of null hypothesis 
        initialGuess_null=np.array([1,1])
        fit_null=minimize(null,initialGuess_null,method="Nelder-Mead",options={'disp':False},args=y)
        # get the likelihood of alternative hypothesis
        initialGuess=np.array([1,1,1])
        fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp':False},args=(x,y))
        # calculate the p-value
        p_tmp=1-chi2.cdf(x=2*(fit_null.fun-fit.fun),df=1) 
        p_regression_sum=p_tmp+p_regression_sum
    # calculate the average of p-value after iteration 
    p_regression=p_regression_sum/iteration
    return p_regression



 # Customer Function for getting p-value from anova design by monte carlo approach

def anova(slope,intercept,sigma,iteration,obs_n,level):
    p_anova_sum=0
    for i in range(iteration):
        # generate random floats in the range of 0-50, repeat level by obs_n/level times
        x_anova=np.random.uniform(0,50,level)
        x=np.repeat(x_anova,obs_n/level)
        # generate the y with standard deviation sigma
        y=slope*x+intercept
        y=y+np.random.randn(24)*sigma
        # get the likelihood of null hypothesis 
        initialGuess_null=np.array([1,1])
        fit_null=minimize(null,initialGuess_null,method="Nelder-Mead",options={'disp':False},args=y)
        # get the likelihood of alternative hypothesis
        initialGuess=np.array([1,1,1])
        fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp':False},args=(x,y))
        # calculate the p-value, note here the df=level-1
        p_tmp=1-chi2.cdf(x=2*(fit_null.fun-fit.fun),df=level-1) 
        p_anova_sum=p_tmp+p_anova_sum
    # calculate the average of p-value after iteration 
    p_anova=p_anova_sum/iteration
    return p_anova   

# Compare the p-value from the regression design and anova design 

# slope=0.4; Intercept=10; iteration=10; obs_n (total experiment units)=24
# for the anova design, three levels 2,4,8 were choosen 
# standard deviation were selected to be 1,2,4,6,8,12,24
level=[2,4,8]
sigma=[1,2,4,6,8,12,24]
# present the power analysis by summarizing all the average P values in a data frame 
index=['regression','2 level','4 level','8 level']
p_df=pd.DataFrame(index=index,columns=sigma)
# fill the p-value of regression design under each sigma
for i in range(7):
    p_df.iloc[0,i]=regression(0.4,10,sigma[i],10,24)
# fill the p-value of anova design under each sigma and level
for i in range(7):
    for j in range(3):
        p_df.iloc[j+1,i]=anova(0.4,10,sigma[i],10,24,level[j])
# print the p-value table for two different designs, where the anova design has three different levels 2,4,8    
p_df    
    
 
    