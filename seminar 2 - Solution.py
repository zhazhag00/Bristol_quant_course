import pandas as pd
import numpy as np

path = "D:\\布大机器学习\\pythonProject1\\"

# load data sets
ftse = pd.read_excel(path + "FTSE.xlsx")

tesco = pd.read_excel(path + "Tesco.xlsx")

#######
# Ret #
#######

# sort values by date, the most oldest date first
ftse = ftse.sort_values(by=['Date'])

# calculate the log ret
ftse['log_ret'] = ftse['Close']/ftse['Close'].shift(1)
ftse['log_ret'] = ftse['log_ret'].apply(lambda x: np.log(x))

# sort values by date, the most oldest date first
tesco = tesco.sort_values(by=['Date'])

# calculate the log ret
tesco['log_ret'] = tesco['Close']/tesco['Close'].shift(1)
tesco['log_ret'] = tesco['log_ret'].apply(lambda x: np.log(x))

############
# SumStats #
############

ftse['log_ret'].describe([0.01, 0.1, 0.9, 0.95, 0.99])
ftse['log_ret'].mean()
ftse['log_ret'].median()
ftse['log_ret'].std()

for i in ['FTSE100', 'S&P500', 'NASDAQ']:
    print(f"The mean of {i}", ftse['log_ret'].mean().round(4))

# print the summary statistics of FTSE100
print("The mean of FTSE100:",ftse['log_ret'].mean())
print("The median of FTSE100:",ftse['log_ret'].median())
print("The standard deviation of FTSE100:",ftse['log_ret'].std())
print("The skewness of FTSE100:",ftse['log_ret'].skew())
print("The kurtosis of FTSE100:",ftse['log_ret'].kurtosis())

# print the summary statistics of Tesco
print("The mean of Tesco:",tesco['log_ret'].mean())
print("The median of Tesco:",tesco['log_ret'].median())
print("The standard deviation of Tesco:",tesco['log_ret'].std())
print("The skewness of Tesco:",tesco['log_ret'].skew())
print("The kurtosis of Tesco:",tesco['log_ret'].kurtosis())

########
# Hist #
########

import matplotlib.pyplot as plt
# Plot the histogram
plt.hist(ftse['log_ret'], bins=30, edgecolor='black', alpha=0.7)
# plt.hist(tesco['log_ret'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

##############
# Regression #
##############
import statsmodels.formula.api as smf

# rename columns
ftse.rename(columns={'Close':'FTSE100_price',\
                     'log_ret':'FTSE100_ret',\
                     'Volume':'FTSE100_vol'},inplace=True)
# rename columns
tesco.rename(columns={'Close':'Tesco_price',\
                      'log_ret':'Tesco_ret',\
                      'Volume':'Tesco_vol'},inplace=True)
# combine 
reg_data = pd.merge(ftse,tesco)

formula = 'Tesco_price ~ FTSE100_price'
results = smf.ols(formula, reg_data).fit()
print(results.summary())

############
# Reg Plot #
############

# Create the scatter plot
plt.scatter(reg_data['FTSE100_price'], reg_data['Tesco_price'], label='Data')

# Plot the regression line
plt.plot(reg_data['FTSE100_price'], results.predict(), color='red', label='Regression Line')

# Add labels and title
plt.xlabel('FTSE100 Price')
plt.ylabel('Tesco Price')
plt.title('Scatter Plot with Regression Line')
plt.legend()

# Show the plot
plt.show()


