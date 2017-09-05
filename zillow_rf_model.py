"""
This is one component I have been integrating into my model for predicting the the zestimate error (I'm just outside top 100).
On its own the random forest is not the greatest of predictive models (~0.0648) and it is greatly outperformed by
a standalone XGBoost model or standalone LightGBM model (both found in other kernels already posted). I have been blending multiple 
models together and I have had success with integrating the random forest regression outputs from this script at a low weight.
I hope some other people can integrate this small snippet of code into their solutions and improve their results as well. 

Sorry I'm not showing my whole model... but this is a competition after all!
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder


print('reading train and test data')
#id and 57 predictor columns such as number of bedrooms and stufff like that
properties = pd.read_csv('../input/properties_2016.csv')
#train has the logerror, a parcelid and a transaction date
train = pd.read_csv('../input/train_2016_v2.csv')
#read in the test_ids needed for submission
test_ids = pd.read_csv('../input/sample_submission.csv')

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
    properties[c].fillna(properties[c].median(skipna=True), inplace = True)
#fill -1 for categorical columns
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == object:
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#merge the training data with the properties
train_df = train.merge(properties, how='left', on='parcelid')
#drop the id from the test dataframe
x_test = properties.drop(['parcelid'], axis=1)
#drop id, data and logerror(response) from the df to make the training df
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
#retrieve just the logerror of the training set
y_train = train_df["logerror"].values.astype(np.float32)


#######
# turn the general pandas dataframes into numpy arrays for use in the random forest regressions
#######
x_train = x_train.values.astype(np.float32, copy=False)
x_test = x_test.values.astype(np.float32, copy=False)
z_rf = RandomForestRegressor(max_depth= 8, n_estimators = 100)

z_rf.fit(x_train, y_train)

rf_y_test = z_rf.predict(x_test)

"""
#uncomment this section and run locally in order to optimize the paramaters, you can add others as well.
param_grid = {
                 'n_estimators': [500, 1000, 1500, 2000],
                 'max_depth': [5, 7, 9]
             }

grid = GridSearchCV(z_rf, param_grid, cv=10)

grid.fit(x_train, y_train)

print(grid.best_score_)
print(grid.best_params_)
"""


test_columns = ['201610','201611','201612','201710','201711','201712']

for i in test_columns:
    test_ids[i] = [float(format(house, '.4f')) for house in rf_y_test]
    
test_ids.to_csv('cam_rf_model_component.csv', index=False)
