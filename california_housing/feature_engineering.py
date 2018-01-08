
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import gc
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

"""
cal_cities_lat_long.csv modified from: http://52.26.186.219/internships/useit/content/cities-california-latitude-and-longitude

cal_populations.csv modified from: http://www.dof.ca.gov/Reports/Demographic_Reports/documents/2010-1850_STCO_IncCities-FINAL.xls


"""

#######
# read in the data
######
housing = pd.read_csv('housing.csv')
housing.head()


# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#look a the categories
housing["income_cat"].hist()

#make a stratified split of the data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]

for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)

gc.collect()

#####
# plot data 
#####


train_set.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
              s=train_set['population']/100, label='population', figsize=(10,7),
              c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend() 


attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,8))



#####
# Alter existing features
#####

# total rooms --> rooms_per_household
# total bedrooms --> bedrooms per household

def housing_data_clean(input_df):
    input_df['rooms_per_household'] = input_df['total_rooms']/input_df['households']
    input_df['bedrooms_per_household'] = input_df['total_bedrooms']/input_df['households']
    input_df['bedrooms_per_room'] = input_df['total_bedrooms']/input_df['total_rooms']
    input_df['population_per_household'] = input_df['population']/input_df['households']
    input_df = input_df.drop(['total_bedrooms','total_rooms'], axis=1)
    return input_df

train_set = housing_data_clean(train_set)
train_set.head()
#do the same to the test set at the same time so they remain consistent with one another!
test_set = housing_data_clean(test_set)

X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value'].values.astype(float)

X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].values.astype(float)



#####
# fill numerical values
#####

def fill_median(dataframe, cols):
    """impute the mean for a list of columns in the dataframe"""
    for i in cols:
        dataframe[i].fillna(dataframe[i].median(skipna=True), inplace = True)
    return dataframe

def cols_with_missing_values(dataframe):
    """ query a dataframe and find the columns that have missing values"""
    return list(dataframe.columns[dataframe.isnull().any()])

def fill_value(dataframe, col, val):
    """impute the value for a list column in the dataframe"""
    """ use this to impute the median of the train into the test"""
    dataframe[i].fillna(val, inplace = True)
    return dataframe


missing_vals = cols_with_missing_values(X_train)
X_train = fill_median(X_train, missing_vals)

for i in missing_vals:
    X_test = fill_value(X_test, i, X_train[i].median(skipna=True))

#####
# One hot encode the categoricals
#####


encoder = LabelBinarizer()

encoded_ocean_train_1hot = encoder.fit_transform(X_train['ocean_proximity'])
#I'm using just transform below to ensure that the categories are sorted and used the same as in the train fit.
encoded_ocean_test_1hot = encoder.transform(X_test['ocean_proximity'])


train_cat_df = pd.DataFrame(encoded_ocean_train_1hot, index = X_train.index, columns = encoder.classes_ )
test_cat_df = pd.DataFrame(encoded_ocean_test_1hot,index = X_test.index, columns = encoder.classes_ )


###
# Combine and scale the dfs
###

X_train.drop('ocean_proximity', axis=1, inplace=True)
X_test.drop('ocean_proximity', axis=1, inplace=True)


X_train = pd.concat([X_train, train_cat_df], axis=1)
X_test = pd.concat([X_test, test_cat_df], axis=1)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


"""

>>> from sklearn.preprocessing import StandardScaler
>>>
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler(copy=True, with_mean=True, with_std=True)
>>> print(scaler.mean_)
[ 0.5  0.5]
>>> print(scaler.transform(data))
"""


#bind the encoded cols to the scaled train and test. 
#then proceed with more feature engineering
