import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  

dataset_1 = pd.read_csv('ACTIVITY_s.csv').drop_duplicates()
dataset_1['date'] = dataset_1.date.astype('datetime64[ns]')
dataset_2 = pd.read_csv('SLEEP_s.csv').drop_duplicates()
# dataset_9 = pd.read_csv('USER_1580696400122.csv')


# Average calculation

date_hb = pd.read_csv('HEARTRATE_s.csv').drop(['time'], axis = 1)
date_hb.to_csv('HEARTRATE.csv', index = False)

df = pd.read_csv('HEARTRATE.csv', parse_dates = ['date'], index_col = 'date')
avg = df.resample('D').mean()
avg = avg.reset_index()


# Merge datasets

merged = dataset_1.merge(avg, on = 'date')
merged['lastSyncTime'] = pd.to_datetime(merged['lastSyncTime'],unit='s')
merged.to_csv("merge.csv")

# Merging sleep cycle

dataset_2['date'] = dataset_2['date'].astype('datetime64[ns]')
deep_sleep_merge = merged.merge(dataset_2, on = 'date').drop('lastSyncTime_x', axis = 1).drop('lastSyncTime_y', axis = 1)


# For missing Values (nan)

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




