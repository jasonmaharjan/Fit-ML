import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  

dataset_1 = pd.read_csv('ACTIVITY_1577030100133.csv').drop_duplicates()
dataset_1['date'] = dataset_1.date.astype('datetime64[ns]')
dataset_9 = pd.read_csv('USER_1577030100069.csv')


# Average calculation

date_hb = pd.read_csv('HEARTRATE_AUTO_1577030100366.csv').drop(['time'], axis = 1)
date_hb.to_csv('HEARTRATE.csv', index = False)

df = pd.read_csv('HEARTRATE.csv', parse_dates = ['date'], index_col = 'date')
avg = df.resample('D').mean()
avg = avg.reset_index()


# Merge datasets

merge = dataset_1.merge(avg, on = 'date')
merge['lastSyncTime'] = pd.to_datetime(merge['lastSyncTime'],unit='s')
merge.to_csv("merge.csv")


# For missing Values (nan)
Z = merge.iloc[:, 2:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(Z[:,:])
Z = (imputer.transform(Z[:,:]))








