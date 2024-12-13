
import pandas as pd
import numpy as np

path1 = "D:\\布大机器学习\\pythonProject1\\"

path2 = path1

# Read data sets
crsp_m = pd.read_csv(path1 + "Monthly_Stock_Returns.csv")
gkx = pd.read_csv(path2 + "datashare.csv")

# Check the dimensions and get information on data sets:
print(crsp_m.shape)
print(crsp_m.head())

print(gkx.shape)
print(gkx.head())
print(gkx.columns)

print(crsp_m.info())
print(gkx.info())

# Make sure that the identifying variables have the same name across data sets:
crsp_m = crsp_m.rename(columns = {'PERMNO':'permno'})
gkx = gkx.rename(columns = {'DATE':'date'})

# Check the number and percentage of missing returns in crsp_m
print(crsp_m['RET'].isna().sum())
print(crsp_m['RET'].isna().sum()/crsp_m.shape[0])

# Convert string datetime into datetime object
crsp_m['date'] = pd.to_datetime(crsp_m['date'].astype(str))
gkx['date'] = pd.to_datetime(gkx['date'].astype(str))

from pandas.tseries.offsets import *
crsp_m['date'] = crsp_m['date'] + MonthEnd(0)
gkx['date'] = gkx['date'] + MonthEnd(0)

from datetime import datetime
# Choose gkx observations after 2010:
gkx = gkx[gkx['date'] > datetime(2009,12,31)]

# Check the number of unique stocks in the data sets:
gkx['permno'].nunique()
crsp_m['permno'].nunique()

# Check the number of months in gkx
gkx['date'].nunique()

# Merge gkx and crsp_m (on all shared keys, i.e., 'date' and 'permno')
# Keep only the overlapping key observations (i.e., use inner join)
data = pd.merge(gkx,crsp_m[['date','permno','RET']],how ='inner')
# Alternatively:
# gkx = gkx.merge(crsp_m[['date','permno','RET']],on = ['date','permno'],\
#                how = 'inner')

# Check the number and percentage of missing returns in gkx:
print(data['RET'].isna().sum())
print(data['RET'].isna().sum()/gkx.shape[0])

# Set up the firm entity and datetime as multi-index for the data
# Then separate the return and all the other features/characteristics
# into two datasets
ret = data[['date','permno','RET']]#.set_index(['date','permno'])
# Drop the column: 'RET' and 'sic2'
data = data.set_index(['date','permno']).drop(['RET','sic2'],axis=1)

# Fix the random seed so that the results are reproducable:
np.random.seed(100)

# Randomly select 20 column numbers based on the chosen random seed:
selected_columns = np.random.randint(1,94,20)
data = data.iloc[:,selected_columns]
print(data.info())

# merge RET and all the 20 selected features together
data = data.reset_index()
data = pd.merge(ret,data,how='inner')
print(data.head())

# Create a leading return column:
data.sort_values(['permno','date'],ascending=True,inplace=True)
data['RET_ahead'] = data.groupby('permno')['RET'].shift(-1)
print(data.head())

# Create a new data frame to store the dense ranked predictors
df_ranked = data.copy()

# Define the columns to be ranked 
# date, permno, and RET_lead will not be selected
columns_to_rank = df_ranked.columns.difference(['date','permno','RET_lead'])

# Normalization to a Uniform Scale (0 to 1) for all the features (i.e., predictors)
# Dense ranking and normalization in machine learning for ensuring 
# that the model treats all features fairly

# Apply dense ranking to the rest of the columns within each group
def rank_norm(column):
    rank = column.rank(method='dense')
    return (rank-1)/(np.nanmax(rank)-1)

df_ranked = df_ranked.groupby('date')[columns_to_rank].transform(rank_norm)
df_ranked = df_ranked.add_suffix('_rank')
print(df_ranked.head())

# Concatenate the original DataFrame with the ranked DataFrame
data = pd.concat([data,df_ranked], axis=1)         
print(data.head())









