
import pandas as pd
import numpy as np
import statsmodels.api as sm

path = ".\\"

# Read data 
gkx = pd.read_csv(path + "gkx_post2000_small.csv")

# Check the dimensions and get information on data sets:
gkx.shape
gkx.head()
gkx.columns
gkx.info()

gkx.dropna(subset=['date'], inplace=True)
gkx['date'] = gkx['date'].astype(int)
gkx['permno'] = gkx['permno'].astype(int)

######### Replace missing values in each variable with cross-sectional median ####

gkx_filled = gkx.drop(columns = ['permno','RET_lead'])

def missing_to_median(column):
    median_value = column.median(skipna=True)
    return column.fillna(median_value)

# Apply the function to each column within each group
gkx_filled = gkx_filled.groupby('date').transform(missing_to_median)

print('\nOriginal data:')
print(gkx.head()) # before filling

columns_filled = gkx.columns.difference(['date','permno','RET_lead'])
gkx[columns_filled] = gkx_filled

print('\nFilled data:')
print(gkx.head()) # after filling

#### Rank normalize the predictors ##########################

# sort gkx on date and permno, since we will group by date
gkx = gkx.sort_values(by=['date','permno']) 
gkx = gkx.reset_index(drop=True)

# Create a new data frame to store the dense ranked predictors
df_ranked = gkx.copy()

# Define the columns to be ranked
columns_to_rank = df_ranked.columns.difference(['date','permno','RET_lead'])

# Apply dense ranking to the rest of the columns within each group

# Define a function that (dense) ranks the columns and maps to [0,1]:
def rank_norm(column):
    rank = column.rank(method='dense')
    return (rank-1)/(np.nanmax(rank)-1)

# Apply the function to df_ranked and store the new columns with '_rank' suffix:
df_ranked = df_ranked.groupby('date')[columns_to_rank].transform(rank_norm)
df_ranked = df_ranked.add_suffix('_rank')

df_ranked.head()

# Concatenate the original dataframe with the rank-normalized dataframe
gkx = pd.concat([gkx,df_ranked], axis=1)   

print('\nData after rank normalization:')      
print(gkx.head())      
del(df_ranked)

############# Correlation Table ##################################

ranked_columns = [col for col in gkx.columns if col.endswith('_rank')]
corr = gkx[['RET_lead']+ranked_columns].corr()

# Print out the summary statistics of all the characteristics
ss=gkx.drop(columns = ['date','permno']+ranked_columns).describe().T

###################################################################
################# Computing Bias and Variance ######################

# Choose training and test set:

train = gkx[(gkx['date'] < 201801) & (gkx['date'] >= 201001)]  
test = gkx[gkx['date'] >= 201801] 

train = train.reset_index(drop=True) 
test = test.reset_index(drop=True)

train.dropna(subset=['RET_lead'], inplace=True)

#####

rng = np.random.default_rng(100)

fitted_values = pd.DataFrame()

# Define a function that takes training set, test set,
# and number of bootstrap samples, trains OLS, and outputs
# fitted values over the test set
def bootstrap_ols(train_set, test_set, num_samples):
    for i in range(num_samples):
        # Bootstrap sampling of the training set
        
        # first input: it is the population from which the samples are drawn
        # second input: the number of samples to draw
        # thrid input: the same row can be selected multiple times in a single bootstrap sample.
        idx_trn = rng.choice(train.shape[0],train.shape[0],replace=True)
        
        # Extract features and target variables
        train_new = train.iloc[idx_trn,:]
        X_train = train_new[ranked_columns]
        # add the intercept for the regression
        X_train = sm.add_constant(X_train)

        # Train an OLS model
        model = sm.OLS(train_new['RET_lead'], X_train).fit()

        # Predict on the test set i.e., out-of-the-sample prediction 
        predictions = model.predict(sm.add_constant(test[ranked_columns]))  # Adjust 'target_variable' with your actual target column

        # Store the fitted values in the DataFrame
        fitted_values[f'Model_{i+1}'] = predictions # using an f-string to add i+1 as a suffix to the column name
        
    return fitted_values

fitted_test = bootstrap_ols(train,test,10)


##################### Computing the squared bias ##############################
# Step 1: Compute the average prediction over the models (trainded on different sets) for each test set observation:
average_predictions = np.mean(fitted_test,axis=1)
# Step 2: Compute the squared difference between the average prediction and the actual test value:
actual_values = test['RET_lead']
squared_errors = (average_predictions - actual_values)**2
# Step 3: Compute the average of squared errors over the test set:
sqrd_bias = np.mean(squared_errors)

#################### Computing the variance ##################################
# Step 1: Compute the variance of predictions over the models for each test set observation:
variance_of_predictions = np.var(fitted_test,axis = 1)
# Step 2: Compute the average of the variances over the test set:
variance = np.mean(variance_of_predictions)

print(f'\nSquared bias = {sqrd_bias}')
print(f'\nVariance = {variance}')




