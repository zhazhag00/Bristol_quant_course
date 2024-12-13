import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.rolling as rolling
import matplotlib.pyplot as plt

path = ".\\"

# load data sets
macro = pd.read_excel(path + "macro.xlsx")

##############
# Ret | Diff #
##############

# sort values by date, the most oldest date first
macro = macro.sort_values(by=['Date'])

# lower the column names
macro.columns = map(str.lower, macro.columns)

def LogDiff(x):
    x_diff = 100*np.log(x/x.shift(1))
    x_diff = x_diff.dropna()
    return x_diff

# calculate the difference
macro['dspread'] = macro['bminusa'].diff()
macro['dcredit'] = macro['ccredit'].diff()
macro['dprod'] = macro['indpro'].diff()
macro['dmoney'] = macro['m1supply'].diff()
macro['inflation'] = LogDiff(macro['cpi'])
macro['rterm'] = (macro['ustb10y'] - macro['ustb3m']).diff()
macro['dinflation'] = macro['inflation'].diff()
macro['rsandp'] = LogDiff(macro['sandp'])

# calculate the excess returns for the stock and for the index
macro['ermsoft'] = LogDiff(macro['microsoft']) - macro['ustb3m']/12
macro['ersandp'] = macro['rsandp'] - macro['ustb3m']/12

########
# Corr #
########
corr_mat = macro[["ermsoft","ersandp","dprod","dcredit","dinflation",\
       "dmoney","dspread","rterm"]].corr()


##############
# Regression #
##############

formula = 'ermsoft ~ ersandp + dprod + dcredit + dinflation + dmoney + dspread + rterm'
results = smf.ols(formula, macro).fit()
print(results.summary())

# robust standard errors
# White's correct for heteroscedasticity
results = smf.ols(formula, macro).fit(cov_type='HC1')
print(results.summary())

# Newey-West adjusted for heteroscedasticity and autocorrelation
results = smf.ols(formula, macro).fit(cov_type='HAC',
                                      cov_kwds={'maxlags':6,'use_correction':True})
print(results.summary())

######################
# Hypothesis Testing #
######################
# Joint test 
hypothesis = "dprod = dcredit = dmoney = dspread = 0"
f_test = results.f_test(hypothesis)

print("Joint Test for Intercept and Slope:")
print("Null Hypothesis:", hypothesis)
print("F-statistic:", f_test.fvalue)
print("p-value:", f_test.pvalue)


###############
# Rolling Reg #
###############

# Select variables for regression
X = macro[['ersandp', 'dprod', 'dcredit', 'dinflation', 'dmoney', 'dspread', 'rterm']]
X = sm.add_constant(X)  # Add constant (intercept)
y = macro['ermsoft']

# Define the rolling window size (e.g., 60 months or 5 years)
window = 60

# Use RollingOLS from statsmodels to fit a rolling OLS regression
rolling_model = rolling.RollingOLS(y, X, window=window)
rolling_results = rolling_model.fit()

# Extract the rolling parameter estimates (betas) and standard errors
betas = rolling_results.params['ersandp'].dropna()
standard_errors = rolling_results.bse['ersandp'].dropna()

# Get the dates
dates = macro[macro.index.isin(betas.index)]['date']

# Compute 95% confidence intervals
upper_bound = betas + 1.96 * standard_errors
lower_bound = betas - 1.96 * standard_errors

# Plot the rolling estimates and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(dates, betas, label=r'$\beta_{ersandp}$')
plt.plot(dates, upper_bound, 'r--', label=r'$\beta_{ersandp} + 2 * SE$')
plt.plot(dates, lower_bound, 'g--', label=r'$\beta_{ersandp} - 2 * SE$')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel(r'$\beta_{ersandp}$')
plt.title('Rolling Window Regression Coefficients using RollingOLS')
plt.legend()

# Show plot
plt.grid(True)
plt.show()










