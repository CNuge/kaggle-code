
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import gc
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from geopy.distance import vincenty

"""
cal_cities_lat_long.csv modified from: http://52.26.186.219/internships/useit/content/cities-california-latitude-and-longitude

both cal_populations.csv modified from: http://www.dof.ca.gov/Reports/Demographic_Reports/documents/2010-1850_STCO_IncCities-FINAL.xls


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





#########
# Engineer more features here prior to 
# passing data in for imputation and one hot encoding
#########

city_lat_long = pd.read_csv('cal_cities_lat_long.csv')
city_pop_data = pd.read_csv('cal_populations_city.csv')
county_pop_data = pd.read_csv('cal_populations_county.csv')


"""
original, had to change because we only want to deal with cities we have
both location and population data on.

city_coords = {}
for dat in city_lat_long.iterrows():
    row = dat[1]
    city_coords[row['Name']] = (float(row['Latitude']), float(row['Longitude']))

#how we deiscovered the need for the change
present = []
absent = []
for city in city_coords.keys():
    if city in city_pop_data['City'].values:
        present.append(city)
    else:
        absent.append(city)
len(present)
len(absent)
absent
"""

city_coords = {}

for dat in city_lat_long.iterrows():
    row = dat[1]
    if row['Name'] not in city_pop_data['City'].values:   
        continue           
    else: 
        city_coords[row['Name']] = (float(row['Latitude']), float(row['Longitude']))


#clean pop
#fill in the missing 1980s values with avg rate of change
#make a dictonary of cities lat/long pass in a tuple of lat/longs
#for a given point and do the comparison

#two functions
#1. take two lat long tuples as input
	#return the distance between the two
    #vincenty(tuple1, tuple2)


#example below
newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
x = vincenty(newport_ri, cleveland_oh)
x #distance stored in km, see units on printing
print(x)
type(x.kilometers)

#2. take a dict[city] = (lat, long) of locations and a tuple of lat long
	# run number 1 for each comparison and return a tuple with
	#the closest city's key + value and the distance between the points

def closest_point(location, location_dict):
    """ take a tuple of latitude and longitude and 
        compare to a dictonary of locations where
        key = location name and value = (lat, long)
        returns tuple of (closest_location , distance) """
    closest_location = None
    for city in location_dict.keys():
        distance = vincenty(location, location_dict[city]).kilometers
        if closest_location is None:
            closest_location = (city, distance)
        elif distance < closest_location[1]:
            closest_location = (city, distance)
    return closest_location

#test = (39.524325, -122.293592) #likely 'Willows'
#closest_point(test, city_coords)


#run number 2 to determine both the nearest city, and then
	#also the nearest city with 1million people (subset the original dict)

city_pop_dict = {}
for dat in city_pop_data.iterrows():
    row = dat[1]
    city_pop_dict[row['City']] =  row['pop_april_1990']


big_cities = {}

for key, value in city_coords.items():
    if city_pop_dict[key] > 500000:
        big_cities[key] = value


#do df.apply(closest_point(point,  closest city))

#then do df.apply(closest_point(point, to closest big city))


#######
# adding closest city data to dataframes
#######

housing['close_city'] = housing.apply(lambda x: 
							closest_point((x['latitude'],x['longitude']),city_coords), axis = 1)
housing['close_city_name'] = [x[0] for x in housing['close_city'].values]
housing['close_city_dist'] = [x[1] for x in housing['close_city'].values]

housing = housing.drop('close_city', axis=1)
housing.head()


#add the data relating to the points to the closest big city
housing['big_city'] = housing.apply(lambda x: 
							closest_point((x['latitude'],x['longitude']),big_cities), axis = 1)
housing['big_city_name'] = [x[0] for x in housing['big_city'].values]
housing['big_city_dist'] = [x[1] for x in housing['big_city'].values]

housing = housing.drop('big_city', axis=1)


#I moved the feature engineering before the train/test split so that we can get
#and even stratified shuffle split of the closest cities into the two sets



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
plt.show()

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


####
#
#If more categoricals created in engineering, add them to this step
#
#
#
####
encoder = LabelBinarizer()

encoded_ocean_train_1hot = encoder.fit_transform(X_train[['ocean_proximity', 
															'close_city_name',
															'big_city_name']])
#I'm using just transform below to ensure that the categories are sorted and used the same as in the train fit.
encoded_ocean_test_1hot = encoder.transform(X_test[['ocean_proximity', 
													'close_city_name',
													'big_city_name']])


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



#from here lets go in to an XGB model first,
#then build up a little (4-5 layer) neural network in keras
#I suspect XGB will do well, but we can try out both
#once a final rmse is arrived at, compare it to the previous results from
#the other kernels I made on the dataset.
