
################################################
# Shrinkage and Dimension Reduction Methods    #
# Nov 11, 2024                                 #  
# Ran Tao                                      #
################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# Read data 

path = ".\\"

gkx = pd.read_csv(path + "gkx_post2010_small.csv")
#gkx = gkx.drop(columns = ['sic2'])

######### Replace missing values in each variable with cross-sectional median ####

gkx_filled = gkx.drop(columns = ['permno','RET_lead'])

def missing_to_median(column):
    if column.notnull().any():  # Check if there is at least one non-missing observation
        median_val = column.median()
        return column.fillna(median_val)
    else:
        return column

# Apply the function to each column within each group
gkx_filled = gkx_filled.groupby('date').transform(missing_to_median)

columns_filled = gkx.columns.difference(['date','permno','RET_lead'])
gkx[columns_filled] = gkx_filled
# just to double check there should not contain any missing values now
gkx = gkx.dropna(subset=list(gkx_filled))

#### Rank normalize the predictors ##########################

gkx = gkx.sort_values(by=['date','permno']) # sort gkx on date and permno, since we will group by date
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
      
# del(df_ranked)

###################################################################

# Choose training and test set:

train = gkx[(gkx['date'] < 201801) & (gkx['date'] >= 201001)]  
test = gkx[gkx['date'] >= 201801] 

train = train.dropna(subset=['RET_lead'])
test = test.dropna(subset=['RET_lead'])

train.reset_index(drop=True,inplace = True) 
test.reset_index(drop=True, inplace = True)

ranked_columns = [col for col in gkx.columns if col.endswith('_rank')]

X_Train = train[ranked_columns]
y_Train = train['RET_lead']
X_Test = test[ranked_columns]
y_Test = test['RET_lead']

# Standardize the features:
# making them mean-centered with unit variance
scaler = StandardScaler()
X_Train_scaled = scaler.fit_transform(X_Train)
X_Test_scaled = scaler.transform(X_Test)
X_Train_scaled = pd.DataFrame(X_Train_scaled, columns=ranked_columns)
X_Test_scaled = pd.DataFrame(X_Test_scaled, columns=ranked_columns)

######################################################
################### OLS #############################

ols = sm.OLS(y_Train, sm.add_constant(X_Train_scaled)).fit()
print(ols.summary())

#######################################################
########### Ridge ############################
# lambdas is a range of penalty values (or alpha values) for Ridge,
# covering a large span from 10^10 to 10^-10,
# scaled by the standard deviation of y_Train
lambdas = 10**np.linspace(10, -10, 50) / y_Train.std()

# Ridge regression is implemented here using ElasticNet with l1_ratio=1e-6,
# essentially making it a Ridge model (Elastic Net mixes L1 and L2 penalties,
# but a near-zero L1 ratio effectively makes it Ridge)

# ElasticNet.path() calculates the coefficients for each value of lambda,
# returning an array of coefficients for each feature.

# ElasticNet doesn’t allow l1_ratio=0 directly
soln_array = skl.ElasticNet.path(X_Train_scaled,\
                                 y_Train,
                                 l1_ratio=1e-6,
                                 alphas=lambdas)[1]

# Taking the log of lambda transforms this wide range into a more manageable
# scale, making it easier to observe changes in the coefficients across
# a broad range of regularization strengths.
soln_path = pd.DataFrame(soln_array.T,
                         columns=ranked_columns ,
                         index=-np.log(lambdas))

path_fig, ax = plt.subplots(figsize=(12,12))
soln_path.plot(ax=ax)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left')
plt.show()

###################################################
################ LASSO ###########################

# LASSO is more sensitive to the change of lambdas
lambdas = 10**np.linspace(1, -10, 50) / y_Train.std()

# l1_ratio = 1 is the lasso penalty
soln_array = skl.ElasticNet.path(X_Train_scaled,\
                                 y_Train,
                                 l1_ratio=1.,
                                 alphas=lambdas)[1]

soln_path = pd.DataFrame(soln_array.T,
                         columns=ranked_columns,
                         index=-np.log(lambdas))
                         
path_fig, ax = plt.subplots(figsize=(12,12))

soln_path.plot(ax=ax)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left')
plt.show()

##################################################
''' Plot Ridge and LASSO on the same graph, using l2 and l1 norms
    on the x axis: '''

# Ridge:
lambdas = 10**np.linspace(10, -10, 50) / y_Train.std()

X_train_with_intercept = np.column_stack((np.ones(X_Train_scaled.shape[0]),\
                                          X_Train_scaled))

# Compute the coefficients for each lambda; coefficients on rows
# l1_ratio specifies the mix ratio between L1 and L2 regularization.
# L1 Regularization: this is the penalty term for LASSO
# L2 Regularization: this is the penalty term for Ridge Regression
# l1_ratio=1: Pure Lasso regularization (L1 only).
# l1_ratio=0: Pure Ridge regularization (L2 only) (though it cannot be exactly 0)
soln_array = skl.ElasticNet.path(X_train_with_intercept,\
                                 y_Train,
                                 l1_ratio=1e-6,
                                 alphas=lambdas)[1]
  
# Compute the relative L2 norm:
beta_ridge = np.sum(soln_array**2, axis=0)/(ols.params**2).sum()
beta_ridge = np.sqrt(beta_ridge)

# Remove the row that gives the intercept values (not interested in them)
soln_array = soln_array[1:]

soln_path = pd.DataFrame(soln_array.T,
                         columns=ranked_columns ,
                         index=beta_ridge)

path_fig, ax = plt.subplots(1,2,figsize=(12,6))

for column in soln_path.columns:
    ax[0].plot(soln_path.index, soln_path[column], label=column)

# The following line also works in some versions of Python without the need of 
# a for loop:
# ax[0].plot(soln_path.index, soln_path, label=soln_path.columns)

ax[0].set_xlabel(r'$||\beta||_2/||\beta_{OLS}||_2$', fontsize=20)
ax[0].set_ylabel('Standardized coefficients', fontsize=20)
ax[0].legend(loc='upper left', fontsize=5)

# LASSO:
lambdas = 10**np.linspace(1, -10, 50) / y_Train.std()

soln_array = skl.ElasticNet.path(X_train_with_intercept,\
                                 y_Train,
                                 l1_ratio=1.,
                                 alphas=lambdas)[1]

beta_lasso = np.sum(np.abs(soln_array), axis=0)/ols.params.abs().sum()

soln_array = soln_array[1:]

soln_path = pd.DataFrame(soln_array.T,
                         columns=ranked_columns,
                         index=beta_lasso)

for column in soln_path.columns:
    ax[1].plot(soln_path.index, soln_path[column], label=column)                      

# The following line also works in some versions of Python without the need of 
# a for loop:
# ax[1].plot(soln_path.index, soln_path, label=soln_path.columns)

ax[1].set_xlabel(r'$||\beta||_1/||\beta_{OLS}||_1$', fontsize=20)
ax[1].set_ylabel('Standardized coefficients', fontsize=20)

plt.tight_layout()
plt.show()

##########################################################################
########################### PCR and PLS ##################################

# Trin a PCA on the training data:
# Note that we've imported PCA function from sklearn.decomposition
pca = PCA(len(ranked_columns))
# transforming the dataset into a new set of features (principal components)
# that capture the maximum variance
X_Train_PCA = pca.fit_transform(X_Train_scaled)

PCR_coefs_df = pd.DataFrame(columns=ranked_columns)
PLS_coefs_df = pd.DataFrame(columns=ranked_columns)

# responses need to be centered as well    
y_Train_centered = y_Train - y_Train.mean()
# iterates through the number of components from 1 to the total number of
# columns (features). This allows the model to test different numbers of 
# components to evaluate the performance
for num_comp in range(1,len(ranked_columns)+1):

    # Get the first num_comp principal components
    # And then get the 1st and 2nd PCs for the next round
    X_Train_PCA_subset = X_Train_PCA[:, :num_comp]

    # Perform linear regression on the selected principal components
    model = sm.OLS(y_Train_centered, X_Train_PCA_subset).fit()
    # print(model.summary())

    # Get the loadings from the PCA transformation
    # i.e. pca.components_ each row corresponds one PC
    # each column corresponds the weighting of (original) feature i.e., theta
    loadings = pca.components_[:num_comp, :].T

    # Calculate the coefficients of the original features
    # loadings: theta
    # model.params: phi
    coefficients_original_features = np.dot(loadings, model.params)

    # Store the PCR coefficients
    PCR_coefs_df.loc[f'{num_comp}_comp'] = coefficients_original_features
    
    # Perform PLS regression
    pls = PLSRegression(n_components=num_comp)
    pls.fit(X_Train_scaled, y_Train_centered)

    # Get the coefficients
    # The pls.coef_ array typically has a shape of (n_features, n_targets).
    # If you have a single response (which is common in regression tasks),
    # the shape of pls.coef_ will be (n_features, 1)
    coefficients_original_features = pls.coef_.flatten()
    
    # Store the PLS coefficients
    PLS_coefs_df.loc[f'{num_comp}_comp'] = coefficients_original_features
    
    
PCR_coefs_df.reset_index(drop=True, inplace=True)
PCR_coefs_df.index = range(1,len(ranked_columns)+1)

PLS_coefs_df.reset_index(drop=True, inplace=True)
PLS_coefs_df.index = range(1,len(ranked_columns)+1)

fig, ax = plt.subplots(1,2,figsize=(20,10))

for column in PCR_coefs_df.columns:
    ax[0].plot(PCR_coefs_df.index, PCR_coefs_df[column], label=column)

#ax[0].plot(PCR_coefs_df.index, PCR_coefs_df, label=PCR_coefs_df.columns)
ax[0].set_title('PCR Coefficients of Features')
ax[0].set_xlabel('Number of Components', fontsize=20)
ax[0].set_ylabel('Standardized coefficients', fontsize=20)
ax[0].legend(loc='upper left', fontsize=6)

for column in PLS_coefs_df.columns:
    ax[1].plot(PLS_coefs_df.index, PLS_coefs_df[column], label=column)

#ax[1].plot(PLS_coefs_df.index, PLS_coefs_df, label=PLS_coefs_df.columns)
ax[1].set_title('PLS Coefficients of Features')
ax[1].set_xlabel('Number of Components', fontsize=20)
ax[1].set_ylabel('Standardized coefficients', fontsize=20)
#ax[1].legend(loc='upper left', fontsize=8)

plt.xticks(range(2,20,2))
plt.show()

##############################################################################
####################### Cross-Validation #####################################


def computes_test_mse(kf):
    
    ''' 
    Applies cross-validation according to the input kf to find the 
    
    optimal parameter value for each model. It then trains each model
    
    on the full training data using the optimal parameter value and computes
    
    the MSE of the predictions of the trained model on the test data.
    
    '''
    
    model = LinearRegression()
    model.fit(X_Train, y_Train)
    
    test_predictions = pd.DataFrame()
    optimal_params = pd.DataFrame(columns=['Optimal Parameter'])

    test_predictions['OLS'] = model.predict(X_Test)

    #####

    ridge = skl.ElasticNet(l1_ratio=1e-6)
    
    scaler = StandardScaler()
    pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])

    lambdas = 10**np.linspace(10, -10, 50)/y_Train.std() 

    param_grid = {'ridge__alpha': lambdas}
    grid = skm.GridSearchCV(pipe,\
                        param_grid,
                        cv=kf,
                        scoring='neg_mean_squared_error')
    # MSE is always non-negative, and a lower MSE indicates a better fit of
    # the model to the data. 
    # Scikit-learn's GridSearchCV function expects a scoring parameter
    # where higher values indicate better models. Many scoring metrics, 
    # like accuracy, precision, or R^2, naturally work this way because a 
    # higher value means better performance.
    # To reconcile this, Scikit-learn introduces the negative mean squared error 
        
    grid.fit(X_Train, y_Train)

    print(f"Optimal Ridge lambda: {grid.best_params_['ridge__alpha']}")

    model = grid.best_estimator_.fit(X_Train_scaled,y_Train)

    optimal_params.loc['Ridge'] = grid.best_params_['ridge__alpha']

    test_predictions['Ridge'] = model.predict(X_Test_scaled)


    ###############

    lasso = skl.ElasticNet(l1_ratio=1)
    lambdas = 10**np.linspace(1, -10, 50)/y_Train.std() 
    pipe = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
    param_grid = {'lasso__alpha': lambdas}
    grid = skm.GridSearchCV(pipe,\
                        param_grid,
                        cv=kf,
                        scoring='neg_mean_squared_error')
    grid.fit(X_Train, y_Train)

    print(f"Optimal LASSO lambda: {grid.best_params_['lasso__alpha']}")

    model = grid.best_estimator_.fit(X_Train_scaled,y_Train)

    optimal_params.loc['LASSO'] = grid.best_params_['lasso__alpha']

    test_predictions['LASSO'] = model.predict(X_Test_scaled)

    ###############

    pipe = Pipeline([ \
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('regressor', LinearRegression())
        ])

    param_grid = {'pca__n_components': range(1,len(ranked_columns)+1)}
    # This is the k-th fold cross-validation function
    grid = skm.GridSearchCV(pipe,\
                        param_grid,
                        cv=kf, # kf is given as input
                        scoring='neg_mean_squared_error')
    grid.fit(X_Train, y_Train)

    print(f"Optimal PCR number of components: {grid.best_params_['pca__n_components']}")

    model = grid.best_estimator_.fit(X_Train,y_Train)

    optimal_params.loc['PCR'] = grid.best_params_['pca__n_components']

    test_predictions['PCR'] = model.predict(X_Test)

    #################

    pipe = Pipeline([ 
    ('scaler', StandardScaler()),
    ('pls', PLSRegression())
    ])

    param_grid = {'pls__n_components': range(1,len(ranked_columns)+1)}
    grid = skm.GridSearchCV(pipe,\
                        param_grid,
                        cv=kf,
                        scoring='neg_mean_squared_error')
    grid.fit(X_Train, y_Train)

    print(f"Optimal PLS number of components: {grid.best_params_['pls__n_components']}")

    model = grid.best_estimator_.fit(X_Train,y_Train)

    optimal_params.loc['PLS'] = grid.best_params_['pls__n_components']

    test_predictions['PLS'] = model.predict(X_Test)
    
    test_error = test_predictions.sub(y_Test, axis=0)
    test_mse = (test_error**2).mean(0)
    
    return((optimal_params,test_mse))

##################

n_folds = 5

##################

# k-fold random CV:
    
''' 
Kfold splits the data into n_splits folds. If shuffle = True, it randomly
shuffles the data before the split. For training, it uses all data except the
hold-out fold. The line below creates an instance of Kfold called kf, with 
n_splits = n_folds and shuffling. If used, random_state argument makes sure
that subsequent runs of the code give the same split.
'''   
    
# the data will be randomly shuffled before being split into n folds.
# Sets a random seed to ensure reproducibility
kf = skm.KFold(n_splits=n_folds, shuffle=True, random_state=100)

optimal_params, test_mse = computes_test_mse(kf)

print('\nOptimal parameters:\n')
print(optimal_params)

print('\nTest MSEs:\n')
print(test_mse)

# Compute the TSS over the test set using the mean of the target over the
# training set. This will be our benchmark for the test MSEs of our models.
TSS = ((y_Test - y_Train.mean())**2).mean()

print(f"\nR2s:\n\n{1-test_mse/TSS}")

###################

# k-fold time-series (recursive) CV:

# TimeSeriesSplit splits the data into n_splits+1 folds, respecting the
# order of the data. For training, it uses only the folds that come before
# the validation set. The line below creates an instance of TimeSeriesSplit
# called kf, with n_splits = n_folds.

# It creates a series of training and validation sets, where each training 
# set consists of all data up to a certain point in time, and the validation 
# set is the data immediately following that training set.
# By doing so, we avoid the look-ahead bias i.e.,  "future" data being used 
# to predict "past" data

kf = skm.TimeSeriesSplit(n_splits=n_folds)

optimal_params, test_mse = computes_test_mse(kf)

print('\nOptimal parameters:\n')
print(optimal_params)

print('\nTest MSEs:\n')
print(test_mse)

print(f"\nR2s:\n\n{1-test_mse/TSS}")

########################
# Time series CV with gap (hv cross-validation) 
# hv stands for Horizon-Validation

# In some cases, there might be concerns about temporal autocorrelation or 
# data leakage where the most recent observations in the training set are too 
# close to those in the validation set, leading to artificially inflated 
# performance metrics.

class CustomTimeSeriesSplit(skm.BaseCrossValidator):
    
    '''
    A custom skm cross-validator class to be used in skm.GridsearchCV.
    
    The new CV splits the data exactly like a TimeSeriesSplit. However,
    
    it removes gap_size from the end of each training set.
    
    '''
    def __init__(self, n_splits, gap_size):
        self.n_splits = n_splits
        self.gap_size = gap_size
        self.tscv = skm.TimeSeriesSplit(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        for train_index, test_index in self.tscv.split(X, y, groups):
            # Introduce a gap in the training set
            gap_start = max(0, len(train_index) - self.gap_size)
            train_index_with_gap = train_index[:gap_start]
            yield train_index_with_gap, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

custom_cv = CustomTimeSeriesSplit(n_splits=n_folds, gap_size=40000)

optimal_params, test_mse = computes_test_mse(custom_cv)

print('\nOptimal parameters:\n')
print(optimal_params)

print('\nTest MSEs:\n')
print(test_mse)

print(f"\nR2s:\n\n{1-test_mse/TSS}")