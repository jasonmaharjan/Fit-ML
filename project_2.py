# COMPARITIVE ANALYSIS OF FITNESS BANDS





# MI band

# Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer  

# Importing Datasets
dataset_activity = pd.read_csv('ACTIVITY_a.csv').drop_duplicates()
dataset_activity['date'] = dataset_activity.date.astype('datetime64[ns]')
dataset_sleep = pd.read_csv('SLEEP_a.csv').drop_duplicates()

# Average heartbeat calculation
date_hb = pd.read_csv('HEARTRATE_a.csv').drop(['time'], axis = 1)
date_hb.to_csv('HEARTRATE.csv', index = False)
hb = pd.read_csv('HEARTRATE.csv', parse_dates = ['date'], index_col = 'date')
avg_hb = hb.resample('D').mean()    
avg_hb = avg_hb.reset_index()

# Merge average heartbeat/day to main dataset
hb_merged = dataset_activity.merge(avg_hb, on = 'date')
hb_merged['lastSyncTime'] = pd.to_datetime(hb_merged['lastSyncTime'],unit='s')
hb_merged = hb_merged.drop(['lastSyncTime'], axis = 1)
hb_merged.to_csv("hb_merge.csv")

# Merging dataset_sleep to dataset_activity
dataset_sleep['date'] = dataset_sleep['date'].astype('datetime64[ns]')
dataset_merged = hb_merged.merge(dataset_sleep, on = 'date').drop('lastSyncTime', axis =1)
dataset_merged['start'] = pd.to_datetime(dataset_merged['start'], unit='s')
dataset_merged['stop'] = pd.to_datetime(dataset_merged['stop'], unit='s')


# For 0 values in hb_merged (activity columns)
x = hb_merged.iloc[:,1:5].values
imputer_zero = SimpleImputer(missing_values = 0, strategy = "mean")
imputer_zero = imputer_zero.fit(x)
x = (imputer_zero.transform(x))
x = x.astype('int64')
x = np.array(x)

# For nan values in hb_merged (heartbeat column)
y = hb_merged.iloc[:,5:6].values
imputer_nan = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_nan = imputer_nan.fit(y[:,:])
y = (imputer_nan.transform(y[:,:]))
y = y.astype('int64')
y = np.array(y)

# Dataframe with 0 and nan values removed
z = np.concatenate((x,y), axis=1)
z = pd.DataFrame(z)

# Machine Learning
X = z.iloc[:, :3].values # Independent variable matrix
Y = z.iloc[:, 3].values # Dependent variable vector

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

# Most Significant Feature

X_opt = X[:,[2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()






# Fitbit 


# Data Preprocessing

fitbit_dataset = pd.read_csv('fitbit_activity_a.csv')
fitbit_dataset['Date'] = fitbit_dataset.Date.astype('datetime64[ns]')
fitbit_heartbeat = pd.read_csv('fitbit_heartbeat_a.csv')
fitbit_heartbeat['Date'] = fitbit_heartbeat.Date.astype('datetime64[ns]')

fitbit_dataset = fitbit_dataset.merge(fitbit_heartbeat, on = 'Date')
#fitbit_dataset['Steps'] = fitbit_dataset["Steps"].str.replace(",","").astype(float)
#fitbit_dataset['Calories'] = fitbit_dataset["Calories"].str.replace(",","").astype(float)
#fitbit_dataset['Distance'] = fitbit_dataset['Distance'] * 1000


# Machine Learning
Xf = fitbit_dataset.iloc[:, 1:3].values # Independent variable matrix
Yf = fitbit_dataset.iloc[:, 3].values # Dependent variable martix

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xf_train, Xf_test, Yf_train, Yf_test = train_test_split(Xf, Yf, test_size = 0.2, random_state = 0) 

# Fitting Multiple Linear Regression into Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xf_train, Yf_train)

# Predicting the test set results
Yf_pred = regressor.predict(Xf_test)
Yf_pred = Yf_pred.astype('int64')

# Backward Elimination Algorithm
import statsmodels.formula.api as sm
Xf = np.append(arr = np.ones((Xf.shape[0], 1)).astype(int), values = Xf, axis = 1)  # first column must have value 1 for b0 constant

# Removing independent variables that are statistically insignificant
Xf_opt = Xf[:,[0,1,2]]
regressor_OLS_f = sm.OLS(endog = Yf, exog = Xf_opt).fit() 
regressor_OLS_f.summary()

Xf_opt = Xf[:,[0,1]]
regressor_OLS_f = sm.OLS(endog = Yf, exog = Xf_opt).fit()
regressor_OLS_f.summary()

# Most Significant Feature
Xf_opt = Xf[:,[1]]
regressor_OLS_f = sm.OLS(endog = Yf, exog = Xf_opt).fit()
regressor_OLS_f.summary()


prediction = regressor_OLS.predict(4486)



# GRAPHS

# Visualising the regression results for MI Band
# Steps is most significant for MI
# Distance VS Calories 
plt.scatter(X[:,2], Y, color = 'red')
plt.plot(X[:,2], regressor_OLS.predict(X[:,2]), color = 'blue')
plt.title('Fitness Analysis of Mi Smart Band 4')
plt.xlabel('Distance')
plt.ylabel('Calories')
plt.show()

# Steps VS Calories
regressor_OLS = sm.OLS(endog = Y, exog = X[:,[1]]).fit()
plt.scatter(X[:,1], Y, color = 'red')
plt.plot(X[:,1], regressor_OLS.predict(X[:,1]), color = 'blue')
plt.title('Fitness Analysis of Mi Smart Band 4')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()


# Visualising the regression results for Fitbit band
# Distance is most significant for MI
# Distance VS Calories
plt.scatter(Xf[:,1], Yf, color = 'red')
plt.plot(Xf[:,1], regressor_OLS_f.predict(Xf[:,1]), color = 'blue')
plt.title('Fitness Analysis of Fitbit Charge 4')
plt.xlabel('Distance')
plt.ylabel('Calories')
plt.show()

# Steps VS Calories
regressor_OLS_f = sm.OLS(endog = Yf, exog = Xf[:,2]).fit()
plt.scatter(Xf[:,2], Yf, color = 'red')
plt.plot(Xf[:,2], regressor_OLS_f.predict(Xf[:,2]), color = 'blue')
plt.title('Fitness Analysis of Fitbit Charge 4')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()


# Combined Dataset 
fitbit_dataset = fitbit_dataset.rename(str.lower, axis = 'columns')
final_dataset = dataset_merged.merge(fitbit_dataset, on = 'date')

difference_in_steps = ((final_dataset['steps_x'] - final_dataset['steps_y']) / final_dataset['steps_x'])*100
difference_in_steps = np.array(difference_in_steps)

difference_in_distance = ((final_dataset['distance_x'] - final_dataset['distance_y']) / final_dataset['distance_x'])*100
difference_in_distance = np.array(difference_in_distance)

difference_in_calories = ((final_dataset['calories_x'] - final_dataset['calories_y']) / final_dataset['calories_x'])*100



# for nan values for heartbeat for MI
hb_nan = final_dataset.iloc[:,5].values
imputer_nan = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_nan = imputer_nan.fit(hb_nan.reshape(-1,1))
hb_nan = (imputer_nan.transform(hb_nan.reshape(-1,1)))
hb_nan = np.array(hb_nan)

# for nan values for heartbeat for Fitbit
hb_nan_f = final_dataset.iloc[:,-1].values
imputer_nan_f = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_nan_f = imputer_nan_f.fit(hb_nan_f.reshape(-1,1))
hb_nan_f = (imputer_nan_f.transform(hb_nan_f.reshape(-1,1)))
hb_nan_f = np.array(hb_nan_f)

difference_in_hb = ((hb_nan[:,0] - final_dataset['heartrate']) / hb_nan[:,0])*100
difference_in_hb = np.array(difference_in_hb)

# Heartbeat Graph for MI
plt.figure(figsize=(18,5))
plt.plot(final_dataset.iloc[:,0].values, hb_nan, color = 'b')
plt.title('MI Heartbeat Graph')
plt.xlabel('Date')
plt.ylabel('Heartbeat')
plt.show()

# Heartbeat Graph for Fitbit
plt.figure(figsize=(18,5))
plt.plot(final_dataset.iloc[:,0].values, hb_nan_f, color = 'g')
plt.title('Fitbit Heartbeat Graph')
plt.xlabel('Date')
plt.ylabel('Heartbeat')
plt.show()

# Comparison between HeartRates 
plt.figure(figsize=(18,5))
plt.plot(final_dataset.iloc[:,0].values, hb_nan, color = 'b')
plt.plot(final_dataset.iloc[:,0].values, hb_nan_f, color = 'g')
plt.title('MI Heartbeat Graph')
plt.xlabel('Date')
plt.ylabel('Heartbeat')
plt.show()


# BOSS GRAPH

# Comparitive Analysis Graph for Steps and Distance
plt.figure(figsize=(15,7))

plt.plot(final_dataset.iloc[:,0].values, difference_in_steps, linewidth= 3, label = 'Steps difference', color = 'b')
plt.plot(final_dataset.iloc[:,0].values, difference_in_distance , '--',linewidth = 2, label = 'Distance difference', color = 'g')
plt.plot(final_dataset.iloc[:,0].values, difference_in_calories , '--',linewidth = 2, label = 'Calories difference', color = 'r')
#plt.plot(final_dataset.iloc[:,0].values, difference_in_hb,':', linewidth= 3, label = 'Heartbeat difference', color = 'r')

plt.title('Comparative Analysis for MI vs Fitbit Charge 2')
plt.xlabel('Date')
plt.ylabel('Relative % Difference')
plt.legend()
plt.show()





