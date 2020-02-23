# Data Preprocessing

# MI band
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  

# Importing Datasets
dataset_1 = pd.read_csv('ACTIVITY_s.csv').drop_duplicates()
dataset_1['date'] = dataset_1.date.astype('datetime64[ns]')
dataset_2 = pd.read_csv('SLEEP_s.csv').drop_duplicates()
# dataset_9 = pd.read_csv('USER_1580696400122.csv')

# Average heartbeat calculation
date_hb = pd.read_csv('HEARTRATE_s.csv').drop(['time'], axis = 1)
date_hb.to_csv('HEARTRATE.csv', index = False)
hb = pd.read_csv('HEARTRATE.csv', parse_dates = ['date'], index_col = 'date')
avg_hb = hb.resample('D').mean()
avg_hb = avg_hb.reset_index()

# Merge average heartbeat to main dataset
merged = dataset_1.merge(avg_hb, on = 'date')
merged['lastSyncTime'] = pd.to_datetime(merged['lastSyncTime'],unit='s')
merged = merged.drop(['lastSyncTime'], axis = 1)
merged.to_csv("merge.csv")

# Merging sleep cycle to main dataset
dataset_2['date'] = dataset_2['date'].astype('datetime64[ns]')
deep_sleep_merge = merged.merge(dataset_2, on = 'date').drop('lastSyncTime', axis =1)
deep_sleep_merge['start'] = pd.to_datetime(deep_sleep_merge['start'], unit='s')
deep_sleep_merge['stop'] = pd.to_datetime(deep_sleep_merge['stop'], unit='s')

"""
# For missing Values ( 0 and nan)
X = deep_sleep_merge.iloc[:,1:5].values
imputer_zero = SimpleImputer(missing_values = 0, strategy = "mean")
imputer_zero = imputer_zero.fit(X)
X = (imputer_zero.transform(X))
X = X.astype('int64')
X = np.array(X)

Z = deep_sleep_merge.iloc[:,5:6].values
imputer_nan = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_nan = imputer_nan.fit(Z[:,:])
Z = (imputer_nan.transform(Z[:,:]))
Z = Z.astype('int64')
Z = np.array(Z)

final_dataset = np.concatenate((X,Z), axis=1)
final_dataset = pd.DataFrame(final_dataset)
"""



# Machine Learning

X = deep_sleep_merge.iloc[:, 1:4].values # Independent variable matrix
Y = deep_sleep_merge.iloc[:, 4].values # Dependent variable martix

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

# Fitting Multiple Linear Regression into Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.astype('int64')

# Backward Elimination Algorithm
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)  # first column must have value 1 for b0 constant

# Removing independent variables that are statistically insignificant
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Most Significant Feature
X_opt = X[:,[1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Visualising the regression results
plt.scatter(X_opt, Y, color = 'red')
plt.plot(X_opt, regressor_OLS.predict(X_opt), color = 'blue')
plt.title('Fitness Analysis')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()


Y_prediction = regressor_OLS.predict([9500])
# Fitbit 
"""
dataset_3 = pd.read_csv('fitbit.csv')
dataset_3['Date'] = dataset_3.Date.astype('datetime64[ns]')

X2 = dataset_3.iloc[:, 1:2].values # Independent variable matrix
Y2 = dataset_3.iloc[:, 3].values # Dependent variable martix








"""