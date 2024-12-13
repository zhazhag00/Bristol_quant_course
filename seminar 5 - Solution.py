
################################################
# Momentum Portfolio                           #
# Oct 2024                                     #  
# Ran Tao                                      #
################################################

import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime

###################
# CRSP Block      #
###################

path1 = ".\\"
path2 = ".\\"

# Read data sets
crsp_m = pd.read_csv(path1 + "Monthly_Stock_Returns_gw.csv")
gkx = pd.read_csv(path2 + "datashare_gw.csv")

sample = gkx.head(1000)

# Make sure that the identifying variables have the same name across data sets:
crsp_m = crsp_m.rename(columns = {'PERMNO':'permno'})
gkx = gkx.rename(columns = {'DATE':'date'})

# Convert string datetime into datetime object
crsp_m['date'] = pd.to_datetime(crsp_m['date'].astype(str))
gkx['date'] = pd.to_datetime(gkx['date'].astype(str))

crsp_m['date'] = crsp_m['date'] + MonthEnd(0)
gkx['date'] = gkx['date'] + MonthEnd(0)

# Print out all the features/characteristics
print(gkx.columns)

# Merge gkx and crsp_m (on all shared keys, i.e., 'date' and 'permno')
# Keep only the overlapping key observations (i.e., use inner join)
data = pd.merge(gkx[['date','permno','mom12m']],\
                crsp_m[['date','permno','RET']],how ='inner')

# Create a leading return column:
data.sort_values(['permno','date'],ascending=True,inplace=True)
data['RET_ahead'] = data.groupby('permno')['RET'].shift(-1)
print(data.head())


##############
# Formation  #
##############
# Formation of 10 Momentum Portfolios

# For each date: assign ranking 1-10 based on cumret
# 1=lowest 10=highest cumret
# Check the number and percentage of missing returns in gkx:
data['mom12m'].isna().sum()
data['mom12m'].isna().sum()/data.shape[0]
data = data.dropna(axis=0, subset=['mom12m'])
data['rank'] = data.groupby('date')['mom12m']\
                   .transform(lambda x: pd.qcut(x, 10, labels=False))
data['rank'] = data['rank'].astype(int)
data['rank'] = data['rank']+1

# compute equally-weighted portfolio return
data_ew = data.groupby(['date','rank'])['RET_ahead'].mean()\
              .rename('ewret').reset_index()

# Transpose portfolio layout to have columns as portfolio returns
retdat = data_ew.pivot(index='date', columns='rank', values='ewret')

retdat = retdat[(retdat.index<=datetime(2008,12,31))&\
                (retdat.index>=datetime(1979,1,31))]

# Add prefix port in front of each column
retdat = retdat.add_prefix('port')
retdat = retdat.rename(columns={'port1':'losers', 'port10':'winners'})
retdat['long_short'] = retdat['winners'] - retdat['losers']

# Compute Long-Short Portfolio Cumulative Returns
retdat2 = retdat
retdat2['1+losers']=1+retdat2['losers']
retdat2['1+winners']=1+retdat2['winners']
retdat2['1+ls'] = 1+retdat2['long_short']

retdat2['cumret_winners']=retdat2['1+winners'].cumprod()-1
retdat2['cumret_losers']=retdat2['1+losers'].cumprod()-1
retdat2['cumret_long_short']=retdat2['1+ls'].cumprod()-1

######################
# Portfolio Analysis #
###################### 

# Mean 
mom_mean = retdat2.mean().to_frame()
mom_mean = mom_mean.reset_index().rename(columns={0:'mean','index':'rank'})

def ffreg(x):
    # Newey-West adjust t-statistics
    return sm.OLS(x,np.ones(len(x))).fit(cov_type='HAC',cov_kwds={'maxlags':6})\
             .tvalues

# T-Value and P-Value
t_output = retdat2.apply(ffreg,axis=0).T.reset_index()\
                  .rename(columns={'const':'t-stat','index':'rank'})

# Combine mean, t and p
mom_output = pd.merge(mom_mean, t_output, on=['rank'], how='inner')                 
print(mom_output)

gkx_head = gkx.head()
crsp_m_head = crsp_m.head()

###########################
# Fama-MacBeth Regression #
########################### 

# Get back to pooled data
regdat = data[(data['date']<=datetime(2008,12,31))&\
              (data['date']>=datetime(1979,1,31))]

# set up multi-index 
regdat = regdat.set_index(['permno','date'])

def ols_coef(x,formula):
    return smf.ols(formula,data=x).fit().params

def ols_r_squared(x,formula):
    return smf.ols(formula,data=x).fit().rsquared_adj

def ols_nobs(x,formula):
    return smf.ols(formula,data=x).fit().nobs

formula = 'RET_ahead ~ 1 + mom12m'

gamma = (regdat.groupby(level=1).apply(ols_coef,formula))
r_squared = (regdat.groupby(level=1).apply(ols_r_squared,formula))
nobs = (regdat.groupby(level=1).apply(ols_nobs,formula))

def fm_summary(p):
    s = p.describe().T
    s['std_error'] = s['std']/np.sqrt(s['count'])
    s['tstat'] = s['mean']/s['std_error']
    return s[['mean','std_error','tstat']]

output = fm_summary(gamma)
output['adj_r^2'] = r_squared.mean()
output['nobs'] = nobs.sum()
print(output)



