import numpy
import pandas
import scipy.stats
from scipy.optimize import minimize
from plotnine import *

#negative loglikelihood function for y = B0 + B1 * x + E
def nllike_alt (p, obs):
    B0 = p[0]
    B1 = p[1]
    sigma = p[2]
    expected = B0 + B1 * obs.x
    nll = -1 * scipy.stats.norm(expected, sigma).logpdf(obs.y).sum()
    return nll

#negative loglikelihood function for null model y = B0 + E
def nllike_null (p, obs):
    B0 = p[0]
    sigma = p[1]
    expected = B0
    nll = -1 * scipy.stats.norm(expected, sigma).logpdf(obs.y).sum()
    return nll
    
#function for returning p-value for t-test
def anova (data):
    #calculate degrees of freedom as number of groups minus one
    ### the max of data.group is actually one less than the number of groups, so no subtraction is needed
    degrees_free = max(data["group"]) 
   
    # y = B0 + B1*x + E
    alt_model = minimize(nllike_alt, [1, 1, 1], method = "Nelder-Mead", options={'disp': False}, args = data)
    
    # y = B0 + E
    null_model = minimize(nllike_null, [1, 1], method = "Nelder-Mead", options={'disp': False}, args = data)

    #Get differences in fit
    D = (null_model.fun - alt_model.fun) * 2
    
    #Use chi3.sf() for returning p-value
    p = scipy.stats.chi2.sf(D, degrees_free)
    
    return p

### This function gets the average p value from the monte carlo approach
### with a given number of groups and sigma value
### the number of groups corresponds to whether the function runs a regression (24 groups)
### or an 'n'-level ANOVA
def get_power (num_groups, sigma):
    num_reps = 10
    
    data_list = []
    p_value_list = []
    
    #establish some variables
    slope = 0.4
    intercept = 10
    
    for d in range(num_reps):
        #Use numpy to append 24 random, uniform x values
        x = numpy.random.uniform(low=0, high=50, size=24)
        
        #Generate a list of 24 y values related to the x values
        y = x * slope + intercept + numpy.random.randn(24) * sigma
        
        ###create dataframe by group
        data = pandas.DataFrame()
        #append x to column
        data["x"] = x
        #append y to column
        data["y"] = y
        
        #assign groups to data based on number of groups
        group = []
        group_len = 24 / num_groups
        for i in range(num_groups):
            for j in range(group_len):
                group.append(i)
        
        #append group to column
        data["group"] = group
        
        #if ANOVA, replace x values with group number
        if (num_groups =< 24):
            data["x"] = group
    
        p_value = anova(data)
        p_value_list.append(p_value)
        
    return (sum(p_value_list) / len(p_value_list))
    
### Here is the actual line of code to run all the functions
### num_group_list runs a regression, 2-level ANOVA, 4-level ANOVA, and 8-level ANOVA
### sigma_list holds the values of the sigma we want to test
num_group_list = [24, 2, 4, 8]
sigma_list = [1, 2, 4, 6, 8, 12, 16, 24]

### Loops through both num_group_list and sigma_list
### Prints out formatted p-values by group number and sigma value
for i in range(len(num_group_list)):
    result_list = []
    for j in range(len(sigma_list)):
        result_list.append(get_power(num_group_list[i], sigma_list[j]))
    
    print("Number groups: " + str(num_group_list[i]))
    for k in range(len(sigma_list)):
        # spaces variable just for formatting to look nice
        spaces = "   "
        if (sigma_list[k] >= 10):
            spaces = "  "
        print("sigma=" + str(sigma_list[k]) + spaces + str(result_list[k]))

