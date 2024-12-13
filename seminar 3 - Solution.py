import pandas as pd
import numpy as np

path = ".\\"

# load data sets
capm = pd.read_excel(path + "capm.xlsx")

#######
# Ret #
#######

# sort values by date, the most oldest date first
capm = capm.sort_values(by=['Date'])

# calculate the log ret
capm['Ford_log_ret'] = capm['FORD']/capm['FORD'].shift(1)
capm['Ford_log_ret'] = capm['Ford_log_ret'].apply(lambda x: np.log(x))*100

capm['SP_log_ret'] = capm['SANDP']/capm['SANDP'].shift(1)
capm['SP_log_ret'] = capm['SP_log_ret'].apply(lambda x: np.log(x))*100

# make annual Treasury bill yields into monthly
capm['USTB3M'] = capm['USTB3M']/12

# excess return over risk-free rate
capm['Ford_exret'] = capm['Ford_log_ret'] - capm['USTB3M']
capm['SP_exret'] = capm['SP_log_ret'] - capm['USTB3M']

########
# Plot #
########
import matplotlib.pyplot as plt

# Plotting Excess Returns
plt.figure(figsize=(10, 6))

plt.plot(capm['Date'], capm['Ford_exret'], label='Ford Excess Returns')
plt.plot(capm['Date'], capm['SP_exret'], label='S&P 500 Excess Returns')

plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Zero Excess Return')

plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.title('Excess Returns of Ford and S&P 500')
plt.legend()
plt.grid(True)

plt.show()

##############
# Regression #
##############
import statsmodels.formula.api as smf

formula = 'Ford_exret ~ SP_exret'
results = smf.ols(formula, capm).fit()
print(results.summary())

######################
# Hypothesis Testing #
######################
# Hypothesis test for the population coefficient (slope) being equal to 1
hypothesis = "SP_exret = 1"
f_test = results.f_test(hypothesis)

print("Hypothesis Test for Ford Excess Returns:")
print("H0: Population Coefficient (Slope) = 1")
print("F-statistic:", f_test.fvalue)
print("p-value:", f_test.pvalue)
print("")

# Joint test for both intercept and slope being equal to 1
hypothesis = "Intercept = 1, SP_exret = 1"
f_test = results.f_test(hypothesis)

print("Joint Test for Intercept and Slope:")
print("Null Hypothesis:", hypothesis)
print("F-statistic:", f_test.fvalue)
print("p-value:", f_test.pvalue)



######################
# t-Test & Normality #
######################

import scipy.stats as stats

# Perform a normality test on the Ford and S&P 500 excess returns
ford_exret = capm['Ford_exret'].dropna()
sp_exret = capm['SP_exret'].dropna()

ford_norm_tstat, ford_norm_pvalue = stats.shapiro(ford_exret)
sp_norm_tstat, sp_norm_pvalue = stats.shapiro(sp_exret)

alpha = 0.05  # Significance level

print("Normality Test:")
print("Ford Excess Returns - p-value:", ford_norm_pvalue)
print("S&P 500 Excess Returns - p-value:", sp_norm_pvalue)
print("")

if ford_norm_pvalue < alpha or sp_norm_pvalue < alpha:
    print("At least one sample is not normally distributed. Proceed with caution.")
else:
    print("Both samples appear to be normally distributed.")

# Conduct a t-test for differences in means
t_stat, p_value = stats.ttest_ind(ford_exret, sp_exret)

print("T-Test for Differences in Means:")
print("H0: Mean(Ford Excess Returns) = Mean(S&P 500 Excess Returns)")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < alpha:
    print("Reject null hypothesis. There is a significant difference in means.")
else:
    print("Fail to reject null hypothesis. No significant difference in means.")

