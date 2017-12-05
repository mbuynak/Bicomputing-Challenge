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

# Regression-Customer Function for alternative hypothesis_significant effect between x and y

# refer to the course material
    
def nllike(p,obs1,obs2):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs1
    nll=-1*norm(expected,sigma).logpdf(obs2).sum()
    return nll     

# ANOVA-Customer Function for alternative hypothesis_significant effect between x and y

# 2 levels 
    
def nllike_anova_2(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x1
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll  


# ANOVA-Customer Function for alternative hypothesis_significant effect between x and y

# 4 levels 
    
def nllike_anova_4(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    B3=p[3]
    sigma=p[4]
    expected=B0+B1*obs.x1+B2*obs.x2+B3*obs.x3
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll   


# ANOVA-Customer Function for alternative hypothesis_significant effect between x and y

# 8 levels
    
def nllike_anova_8(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    B3=p[3]
    B4=p[4]
    B5=p[5]
    B6=p[6]
    B7=p[7]
    sigma=p[8]
    expected=B0+B1*obs.x1+B2*obs.x2+B3*obs.x3+B4*obs.x4+B5*obs.x5+B6*obs.x6+B7*obs.x7
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll   

# Customer Function for getting mean of p-value from regression design by monte carlo approach

def regression(slope,intercept,sigma,iteration,obs_n):
    p_regression_sum=0
    for i in range(iteration):
        # generate random floats in the range of 0-50, total number=obs_n
        x=np.random.uniform(0,50,obs_n)
        # generate y with standard deviation sigma
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

# Customer Function for getting p-value from anova (level 2) by monte carlo approach

# x matrix
 
treatment_2=np.diag(np.ones(1),0)
zero = np.array([[0]])
treatment_2=np.concatenate((treatment_2, zero))
treatment_2=np.tile(treatment_2,(12,1))
df_2=pd.DataFrame(treatment_2,columns=['x1'])

def anova_2(slope,intercept,sigma,iteration):
    p_anova_sum=0
    for i in range(iteration):
        # set two levels-level 1 (x=1), level 2 (x=40)
        x=np.array([1,40])
        x=np.tile(x,(1,12))
        # generate y with standard deviation sigma
        y=slope*x+intercept+np.random.randn(24)*sigma
        y=y.reshape(-1)
        # append the y to the x data frame
        df_2['y']=y
        # get the likelihood of null hypothesis 
        initialGuess_null=np.array([1,1])
        fit_null=minimize(null,initialGuess_null,method="Nelder-Mead",options={'disp':False},args=y)
        # get the likelihood of alternative hypothesis
        initialGuess=np.array([1,1,1])
        fit=minimize(nllike_anova_2,initialGuess,method="Nelder-Mead",options={'disp':False},args=df_2)
        # calculate the p-value, note here the df=level-1
        p_tmp=1-chi2.cdf(x=2*(fit_null.fun-fit.fun),df=1) 
        p_anova_sum=p_tmp+p_anova_sum
        # calculate the average of p-value after iteration 
    p_anova=p_anova_sum/iteration
    return p_anova   

# Customer Function for getting p-value from anova (level 4) by monte carlo approach

# x matrix
    
treatment_4=np.diag(np.ones(3),0)
zero = np.array([[0,0,0]])
treatment_4=np.concatenate((treatment_4, zero))
treatment_4=np.tile(treatment_4,(6,1))
df_4=pd.DataFrame(treatment_4,columns=['x1', 'x2', 'x3'])


def anova_4(slope,intercept,sigma,iteration):
    p_anova_sum=0
    for i in range(iteration):
        # set four levels-level 1 (x=1), level 2 (x=10), level 3 (x=30), level 4 (x=40)
        x=np.array([1,10,30,40])
        x=np.tile(x,(1,6))
        # generate y with standard deviation sigma
        y=slope*x+intercept+np.random.randn(24)*sigma
        y=y.reshape(-1)
        # append the y to the x data frame
        df_4['y']=y 
        # get the likelihood of null hypothesis 
        initialGuess_null=np.array([1,1])
        fit_null=minimize(null,initialGuess_null,method="Nelder-Mead",options={'disp':False},args=y)
        # get the likelihood of alternative hypothesis
        initialGuess=np.array([1,1,1,1,1])
        fit=minimize(nllike_anova_4,initialGuess,method="Nelder-Mead",options={'disp':False},args=df_4)
        # calculate the p-value, note here the df=level-1
        p_tmp=1-chi2.cdf(x=2*(fit_null.fun-fit.fun),df=3) 
        p_anova_sum=p_tmp+p_anova_sum
    # calculate the average of p-value after iteration 
    p_anova=p_anova_sum/iteration
    return p_anova   

# Customer Function for getting p-value from anova (level 8) design by monte carlo approach

# x matrix
    
treatment_8=np.diag(np.ones(7),0)
zero = np.array([[0,0,0,0,0,0,0]])
treatment_8=np.concatenate((treatment_8, zero))
treatment_8=np.tile(treatment_8,(3,1))
df_8=pd.DataFrame(treatment_8,columns=['x1', 'x2', 'x3', 'x4', 'x5','x6','x7'])

        
def anova_8(slope,intercept,sigma,iteration):
    p_anova_sum=0
    for i in range(iteration):
        # set eight levels-level 1 (x=1), level 2 (x=7), level 3 (x=13), level 4 (x=19), level 5 (x=25), level 6 (x=31), level 7 (x=37), level 8 (x=48)
        x=np.array([1,7,13,19,25,31,37,48])
        x=np.tile(x,(1,3))
        # generate y with standard deviation sigma
        y=slope*x+intercept+np.random.randn(24)*sigma
        y=y.reshape(-1)
        # append y to the x data frame
        df_8['y']=y        
        # get the likelihood of null hypothesis 
        initialGuess_null=np.array([1,1])
        fit_null=minimize(null,initialGuess_null,method="Nelder-Mead",options={'disp':False},args=y)
        # get the likelihood of alternative hypothesis
        initialGuess=np.array([1,1,1,1,1,1,1,1,1])
        fit=minimize(nllike_anova_8,initialGuess,method="Nelder-Mead",options={'disp':False},args=df_8)
        # calculate the p-value, note here the df=level-1
        p_tmp=1-chi2.cdf(x=2*(fit_null.fun-fit.fun),df=7) 
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
    p_df.iloc[1,i]=anova_2(0.4,10,sigma[i],10)
    p_df.iloc[2,i]=anova_4(0.4,10,sigma[i],10)    
    p_df.iloc[3,i]=anova_8(0.4,10,sigma[i],10)
# print the p-value table for two different designs, where the anova design has three different levels 2,4,8    
