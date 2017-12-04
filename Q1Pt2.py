import numpy
import pandas
import scipy
import scipy.integrate as spint
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import chi2
import plotnine
from plotnine import *

#Load Data
data=pandas.read_csv("sugar.csv")

#Generate plot that shows a linear regression of the E.coli growth with linear trendline
plot1= ggplot(data,aes(x="sugar",y="growth"))+theme_classic()+geom_point()+stat_smooth(method="lm")
print plot1


dataFrame=pandas.DataFrame({'y':data.growth,'x':0})

#Define null hypothesis 
def nllikeNull(p,obs):
    B0=p[0]
    sigma=p[1]
    
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
# estimate parameters by minimizing the NLL for data
initialGuess=numpy.array([1,1])
fitNull=minimize(nllikeNull,initialGuess,method="Nelder-Mead",options={'disp': True},args=dataFrame)
nllNull= fitNull.fun #gives NLL value for null
print("Null model negative log liklihood value = ")
print(nllNull)

#Define Alternative
def nllikeAlt(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expectedAlt=B0+B1*obs.x
    nll=-1*norm(expectedAlt,sigma).logpdf(obs.y).sum()
    return nll
    
# estimate parameters by minimizing the NLL for data
initialGuess=numpy.array([1,1,1])
fitAlt=minimize(nllikeAlt,initialGuess,method="Nelder-Mead",options={'disp': True},args=dataFrame)
nllAlt= fitAlt.fun #gives NLL value for null
print("Alternative model negative log liklihood value = ") 
print(nllAlt)

### Calculating D values
Dval = 2*(nllNull-nllAlt)
print("D-value= ",Dval)

### P value calculation
pval1=1-scipy.stats.chi2.cdf(x=Dval,df=1)
print(pval1)

