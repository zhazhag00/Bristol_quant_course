
######################
# Tree-Based Methods #
######################

# Import necessary packages 
import pandas as pd
import numpy as np
import sklearn.model_selection as skm
import matplotlib.pyplot as plt

from sklearn.tree import (DecisionTreeRegressor as DTR, plot_tree)
from sklearn.ensemble import (RandomForestRegressor as RF,
                              GradientBoostingRegressor as GBR)

##########################

# Read data 
# path = "C:\\Users\\fz22427\\OneDrive - University of Bristol\\Quantitative Methods, Big Data, and Machine Learning\\2024-2025\\Seminars\\Seminar 7\\"
path = ".\\"

gkx = pd.read_csv(path + "gkx_post2010_small.csv")
# gkx = pd.read_csv(path + "gkx_post2000_small.csv")
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
      
del(df_ranked)

###################################################################

gkx['year'] = gkx['date'].apply(lambda x: str(x)[:4]).astype(int)
gkx = gkx.dropna(subset=['RET_lead'])
gkx.reset_index(drop=True,inplace = True) 

# Choose initial training and test sets:

train = gkx[(gkx['date'] < 201801) & (gkx['date'] >= 201001)]  
test = gkx[gkx['date'] >= 201801] 

ranked_columns = [col for col in gkx.columns if col.endswith('_rank')]

X_train = train[ranked_columns]
y_train = train['RET_lead']
X_test = test[ranked_columns]
y_test = test['RET_lead']

TSS_test = ((y_test - y_train.mean())**2).mean()

#######################################################################
################## DECISION TREE ######################################

reg = DTR(max_depth=3)
reg.fit(X_train , y_train)
ax = plt.subplots(figsize=(24,12))[1]
plot_tree(reg,feature_names=ranked_columns, ax=ax, fontsize = 14, precision=6)

# value in the plot is predicted stock return


# Create a time series split:
n_folds = 5
kf = skm.TimeSeriesSplit(n_splits=n_folds)

# Create a cost complexity prunin (ccp) object and 
# store the alphas in param_grid:
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)

## Need to find the trade-off between impurity and alpha
## Lower alpha (impurity) means that a large tree, which leads to overfitting
## Higher alpha (impurity) means that a small tree, which leads to high variance

# ---------------------------------------------------------------
# fig, ax = plt.subplots()
# ax.plot(ccp_path.ccp_alphas[:-1], ccp_path.impurities[:-1],\
#         marker="o", drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")
# ---------------------------------------------------------------


param_grid = {'ccp_alpha': ccp_path.ccp_alphas}
# kfold = kf

# Define a function that performs CV using a specific model,
# parameter grid, and validation type:
def tree_models(model, param_grid, val_type, X, y):
    grid = skm.GridSearchCV(model,
                            param_grid,
                            refit=True,
                            cv=val_type,
                            scoring='neg_mean_squared_error')
    grid.fit(X, y)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    return((best_model, best_params))

# Fit the model using CV and record the best model and best parameters:
best_model, best_params = tree_models(reg, param_grid, kf, X_train, y_train) 

# Calculate the test MSE and R2:
print("test MSE:",np.mean((y_test - best_model.predict(X_test))**2))
print("R2:",1-np.mean((y_test - best_model.predict(X_test))**2)/TSS_test)

#######################################################################
######################## RANDOM FOREST #################################

# "random_state": Controls both the randomness of the bootstrapping of 
# the samples used when building trees (if bootstrap=True) and the sampling
# of the features to consider when looking for the best split at each node
random_forest = RF(max_features=3,# max # of features at each split
                   n_estimators=100,# the number of trees in the forest
                   max_depth=5,
                   random_state=100).fit(X_train , y_train)

y_hat = random_forest.predict(X_test)

print("test MSE:",np.mean((y_test - y_hat)**2))
print("R2:",1-np.mean((y_test - y_hat)**2)/TSS_test)


param_grid = {
        'n_estimators': [10], # Number of trees
        'random_state': [100], # Seed to initiate random sampling
        'max_features': [1, 3],  # Maximum # of features to sample
        'max_depth': [3],   # Maximum depth of the tree
        }

best_model, best_params = tree_models(RF(), param_grid, kf, X_train, y_train)
 
print("test MSE:",np.mean((y_test - best_model.predict(X_test))**2))
print("R2:",1-np.mean((y_test - best_model.predict(X_test))**2)/TSS_test)

# Create a df that stores purity-based feature importance values:
# ----------------------------------------------------------------------------
## Purity measures how "pure" each of these leaf nodes is with respect to
## the target variable. A pure leaf is one where all data points belong to
## the same class (in classification) or have similar values (in regression).

## The feature that causes the largest decrease in impurity at each split
## is considered the most important.
feature_imp = pd.DataFrame(
    {'importance':best_model.feature_importances_},
    index=ranked_columns)

# Sort feature based on importance:
feature_imp.sort_values(by='importance', ascending=True, inplace=True)

# Plot the feature importance values:
ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('Purity-Based Feature Importance')

plt.show()

# Define a function that computes the R2-based feature importance (see GKX2020):
# ----------------------------------------------------------------------------
## quantifies the contribution of each feature to improving the 𝑅2 score
## of the model (i.e., the proportion of variance in the target variable that
## is explained by the model)

## The feature whose removal leads to the largest drop in 𝑅2 is considered
## the most important.    
def marginal_r2(model, r2_full, X, y):
    feature_imp = pd.DataFrame(columns=['Marginal_R2'])
    for var in ranked_columns:
        x_new = X.copy()
        # This allows the function to evaluate the impact of "removing" that
        # feature on the model's performance i.e., set up as 0
        x_new[var] = 0
        y_hat = best_model.predict(x_new)
        r2 = 1-np.mean((y - y_hat)**2)/np.var(y)      
        feature_imp.loc[var] = max(r2_full - r2,0)  
    feature_imp['Marginal_R2'] = feature_imp.iloc[:,0]/ \
                                            feature_imp.iloc[:,0].sum()
    return(feature_imp)

r2_full = best_model.score(X_train,y_train)
feature_imp = marginal_r2(best_model, r2_full, X_train, y_train)
feature_imp.sort_values(by='Marginal_R2', ascending=True, inplace=True)
     
ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('R2-Based Feature Importance')

plt.show()

####################################################
# Custom cv object to be used in gridsearch:
    
class YearSplit(skm.BaseCrossValidator):
    
    '''
    A custom skm cross-validator class to be used in skm.GridsearchCV.
    
    The new CV splits the data at a certain year.
    
    '''
    def __init__(self, split_year, years = train['year'], n_splits=1):
        self.split_year = split_year
        self.n_splits = n_splits
        self.years = years

    def split(self, X, y=None, groups=None):
        train_index = np.where(self.years < self.split_year)[0]
        test_index = np.where(self.years >= self.split_year)[0]                      
        yield train_index, test_index
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
#################################################################### 

# all data before 2015 will be used for training, 
# and data from 2015 onward will be used for testing   
year_split_val = YearSplit(split_year=2015, years=train['year'])

param_grid = {
        'n_estimators': [100],
        'random_state': [100],
        'max_features': [1, 3],  # Maximum # of features to sample
        'max_depth': [1, 3],   # Maximum depth of the tree
        }

best_model, best_params = tree_models(RF(), param_grid, year_split_val,
                                      X_train, y_train) 
 
print("test MSE:",np.mean((y_test - best_model.predict(X_test))**2))
print("R2:",1-np.mean((y_test - best_model.predict(X_test))**2)/TSS_test)

r2_full = best_model.score(X_train,y_train)

feature_imp = marginal_r2(best_model, r2_full, X_train, y_train)
feature_imp.sort_values(by='Marginal_R2', ascending=True, inplace=True)

ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('R2-Based Feature Importance')

plt.show()

################################################################
#################### BOOSTED TREES ############################

# Gradient boosting is highly sensitive to the choice of learning rate,
# and including two boundary values ensures the model has an opportunity to
# find the most optimal setting for the given data.
param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1, 0.01],  # low values ensure slower, more precise learning
        'max_depth': [3],   # Maximum depth of the tree
        }

best_model, best_params = tree_models(GBR(), param_grid, year_split_val,
                                      X_train, y_train) 
print("test MSE:",np.mean((y_test - best_model.predict(X_test))**2))
print("R2:",1-np.mean((y_test - best_model.predict(X_test))**2)/TSS_test)

feature_imp = marginal_r2(best_model, r2_full, X_train, y_train)
feature_imp = feature_imp.sort_values(by='Marginal_R2', ascending=True)

ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('R2-Based Feature Importance')

plt.show()

#################################################################

# Define a function that performs validation using a single split 
# based on year:
    
def recursive_train(model, param_grid, test_start_year, length_val,
                    length_test=1):
    
    idx_trn = gkx['year'] < test_start_year
    train = gkx[idx_trn].reset_index()
    X_train = train[ranked_columns]
    y_train = train['RET_lead']
    idx_tst = (gkx['year'] >= test_start_year) & \
              (gkx['year'] < test_start_year+length_test)
    test = gkx[idx_tst].reset_index()
    X_test = test[ranked_columns]
    y_test = test['RET_lead']
    
    val_start_year = test_start_year - length_val
    
    year_split_val = YearSplit(split_year=val_start_year, years=train['year'])
    
    best_model, best_params = tree_models(model, param_grid, year_split_val,
                                          X_train, y_train)
    
    #TSS_test = np.mean((y_test - np.mean(y_train))**2)
    r2_full = best_model.score(X_train,y_train)
    
    test_sqr_error_model = (y_test - best_model.predict(X_test))**2
    test_sqr_error_yavg = (y_test - np.mean(y_train))**2
    
    feature_imp = marginal_r2(best_model, r2_full, X_train, y_train)
    #feature_imp.sort_values(by='Marginal_R2', ascending=True,
     #                                    inplace=True)
   
    return((best_model, best_params, test_sqr_error_model, test_sqr_error_yavg, 
            feature_imp))
                                         
######################################################################
 
param_grid = {
        'n_estimators': [10],
        'random_state': [100],
        'max_features': [1, 3],  # Maximum # of features to sample
        'max_depth': [1, 3],   # Maximum depth of the tree
        }
    
best_model, best_params,\
test_sqr_error_model, test_sqr_error_yavg, feature_imp = recursive_train(RF(),
                             param_grid, test_start_year=2018, length_val=3)

print(np.mean(test_sqr_error_model))
print(1-np.mean(test_sqr_error_model)/np.mean(test_sqr_error_yavg))

feature_imp.sort_values(by='Marginal_R2', ascending=True,
                                      inplace=True)

ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('R2-Based Feature Importance')

plt.show()

#########################################################################

# Perform recursive training as in GKX2020: incrase the length of training
# set by one year each time, keep the length of validation set constant,
# use a 1-year test set. Compute the average test MSE, test R2, and 
# feature importance over all trainings.

feature_imp_all = pd.DataFrame()
test_sqr_error_model_all = []
test_sqr_error_yavg_all = []

for year in range(2018,2020):
    
    best_model, best_params,\
    test_sqr_error_model, test_sqr_error_yavg,\
    feature_imp = recursive_train(RF(), param_grid,
                                      test_start_year=year, length_val=3)
    
    test_sqr_error_model_all.append(test_sqr_error_model)
    test_sqr_error_yavg_all.append(test_sqr_error_yavg)
    
    feature_imp_all[f'Train_{year}'] = feature_imp['Marginal_R2']
    
test_mse = np.mean(pd.concat(test_sqr_error_model_all, ignore_index=True))
test_r2 = 1 - test_mse/np.mean(pd.concat(test_sqr_error_yavg_all,
                                         ignore_index=True))

feature_imp = feature_imp_all.mean(axis=1)
feature_imp = pd.DataFrame(feature_imp, columns = ['Marginal_R2'])

feature_imp.sort_values(by='Marginal_R2', ascending=True,
                                      inplace=True)

ax = feature_imp.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('R2-Based Feature Importance')

plt.show()


