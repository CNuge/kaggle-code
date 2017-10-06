
#this follows the XGBoost tutorial on kaggle, I'll skip the replication 
#of the figures (as these will be the same) and focus on the curation,
#and subsequent XGBoost training. Once comfortable I will go through
#and try to tailor the model for more accurate predictions.

#all imports, pull from here as we use them
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]


#used
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np


#load the train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
train.columns.values.tolist()
train.info()
train.describe()
#serious variance in the repsonse column, likely some errors.
"""
only 11 columns including a unique id and a response. so 9 predictors
['id',
 'vendor_id',
 'pickup_datetime',
 'dropoff_datetime',
 'passenger_count',
 'pickup_longitude',
 'pickup_latitude',
 'dropoff_longitude',
 'dropoff_latitude',
 'store_and_fwd_flag',
 'trip_duration']
 """
#datetime is important to account for time of day and day of the week!


#clean the trip duration column
#remove those outside 2 standard deviations of the mean.
#sd trim
mean_trip = np.mean(train['trip_duration'])
sd_trip = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= mean_trip + 4*sd_trip]
train = train[train['trip_duration'] >= mean_trip - 4*sd_trip]




#limit model to only the borders of new york city
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]


#change the date and time formatting.

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)


#log transform of the trip duration to get a normal distribution
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)


import warnings
warnings.filterwarnings("ignore")
plot_vendor = train.groupby('vendor_id')['trip_duration'].mean()

snwflag = train.groupby('store_and_fwd_flag')['trip_duration'].mean()
pc = train.groupby('passenger_count')['trip_duration'].mean()


# determine the distance and direction of a specific trip based 
# on the pickup and dropoff coordinates

def haversine_array(lat1, lng1, lat2, lng2):
	lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
	AVG_EARTH_RADIUS = 6371  # in km
	lat = lat2 - lat1
	lng = lng2 - lng1
	d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
	h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
	return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
	a = haversine_array(lat1, lng1, lat1, lng2)
	b = haversine_array(lat1, lng1, lat2, lng1)
	return a + b

def bearing_array(lat1, lng1, lat2, lng2):
	AVG_EARTH_RADIUS = 6371  # in km
	lng_delta_rad = np.radians(lng2 - lng1)
	lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
	y = np.sin(lng_delta_rad) * np.cos(lat2)
	x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
	return np.degrees(np.arctan2(y, x))

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)	
	
train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

#Applying these functions to both the test and train data, we can 
#calculate the haversine distance which is the great-circle distance 
#between two points on a sphere given their longitudes and latitudes. 
#We can then calculate the summed distance traveled in Manhattan. 
#And finally we calculate (through some handy trigonometry) the direction (
#or bearing) of the distance traveled. These calculations are stored as 
#variables in the separate data sets. The next step I decided to take is 
#to create neighourhods, like Soho, or the Upper East Side, from the data 
#and display this.



#create the "Neighborhoods" using k means clustering
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
					train[['dropoff_latitude', 'dropoff_longitude']].values))


sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

#extract data from the datetime columns
#check that the train and test match for each.
#Extracting Month
train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month

train.groupby('Month').size(),test.groupby('Month').size()

# six months in each of the datasets, can make dummy variables
train['DayofMonth'] = train['pickup_datetime'].dt.day
test['DayofMonth'] = test['pickup_datetime'].dt.day
len(train.groupby('DayofMonth').size()),len(test.groupby('DayofMonth').size())

train['Hour'] = train['pickup_datetime'].dt.hour
test['Hour'] = test['pickup_datetime'].dt.hour
len(train.groupby('Hour').size()),len(test.groupby('Hour').size())

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
len(train.groupby('dayofweek').size()),len(test.groupby('dayofweek').size())

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)
# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]

#data from here:
#https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
# added to the data. It contains info on the fastest route
#between any two points in the train and test sets
#note that only the numeric columns are being appropriated.



fr1 = pd.read_csv('fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('fastest_routes_test.csv',
							   usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')


#check the shape of the dataframes to make sure the additions have worked
train.shape, test.shape


##########################
##########################
#build the dummy variables
##########################
##########################

#note the great use of the pd.get_dummies function here,
#makes this a ten line task instead of way more painful!


vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')
passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')

month_train = pd.get_dummies(train['Month'], prefix='m', prefix_sep='_')
month_test = pd.get_dummies(test['Month'], prefix='m', prefix_sep='_')
dom_train = pd.get_dummies(train['DayofMonth'], prefix='dom', prefix_sep='_')
dom_test = pd.get_dummies(test['DayofMonth'], prefix='dom', prefix_sep='_')
hour_train = pd.get_dummies(train['Hour'], prefix='h', prefix_sep='_')
hour_test = pd.get_dummies(test['Hour'], prefix='h', prefix_sep='_')
dow_train = pd.get_dummies(train['dayofweek'], prefix='dow', prefix_sep='_')
dow_test = pd.get_dummies(test['dayofweek'], prefix='dow', prefix_sep='_')

#check all the dummies pass muster
vendor_train.shape,vendor_test.shape
passenger_count_train.shape,passenger_count_test.shape
store_and_fwd_flag_train.shape,store_and_fwd_flag_test.shape
cluster_pickup_train.shape,cluster_pickup_test.shape
cluster_dropoff_train.shape,cluster_dropoff_test.shape
month_train.shape,month_test.shape
dom_train.shape,dom_test.shape
hour_train.shape,hour_test.shape
dow_train.shape,dow_test.shape

#drop the outlier passanger number of 9 column.
#not that many seats in a taxi so these are an error!
passenger_count_test = passenger_count_test.drop('pc_9', axis = 1)


#drop the categorical columns that were replaced with dummies

train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
				   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
Test_id = test['id']
test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
				   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)

train = train.drop(['dropoff_datetime','avg_speed_h','avg_speed_m','pickup_lat_bin','pickup_long_bin','trip_duration'], axis = 1)

#add indicator variables to the dataset

Train_Master = pd.concat([train,
						  vendor_train,
						  passenger_count_train,
						  store_and_fwd_flag_train,
						  cluster_pickup_train,
						  cluster_dropoff_train,
						 month_train,
						 dom_train,
						  hour_test,
						  dow_train
						 ], axis=1)


Test_master = pd.concat([test, 
						 vendor_test,
						 passenger_count_test,
						 store_and_fwd_flag_test,
						 cluster_pickup_test,
						 cluster_dropoff_test,
						 month_test,
						 dom_test,
						  hour_test,
						  dow_test], axis=1)


#these data are kept in other columns
Train_Master = Train_Master.drop(['pickup_datetime','pickup_date'],axis = 1)
Test_master = Test_master.drop(['pickup_datetime','pickup_date'],axis = 1)

#make sure there is one more column in train than in test
Train_Master.shape,Test_master.shape

#((1446345, 285), (625134, 284))


#The next step is to split the training set into sub-training and sub-testing sets. 
#The reason for this is to be able to tweak model parameters to increase accuracy 
#(i.e. decrease the root mean square error [RSME] value) without creating bias towards the test set.


#Splitting Train Master into 80-20 train test -- For the sake of this kernel I am limiting the data to 100 000 entries

#I altered this to Train_Master from the original Train_Master[0:100000]
#using all million rows instead of just the 100,000
Train, Test = train_test_split(Train_Master, test_size = 0.3)


X_train = Train.drop(['log_trip_duration'], axis=1)
Y_train = Train["log_trip_duration"]
X_test = Test.drop(['log_trip_duration'], axis=1)
Y_test = Test["log_trip_duration"]

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)



dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(Test_master)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



##############
# The XGBoost code for model training - with iterative tweaks
##############

md = [5,6,7,8]
lr = [0.1, 0.2, 0.3]
mcw = [20,25,30]
for m in md:
	for l in lr:
		for n in mcw:
			t0 = datetime.now()
			xgb_pars = {'min_child_weight': n, 'eta': l, 'colsample_bytree': 0.9, 
						'max_depth': m,
			'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
			'eval_metric': 'rmse', 'objective': 'reg:linear'}
			model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,
				  maximize=False, verbose_eval=1)
			print('Modeling iteration %s, RMSLE %.5f' % (xgb_pars, model.best_score))

#pick the lowest RMSE from above, train and make predictions using below



"""

#current top score on submission: 0.45585

#model vals:

Modeling iteration {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.41339
Modeling iteration {'min_child_weight': 25, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.41369
Modeling iteration {'min_child_weight': 30, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.41336
Modeling iteration {'min_child_weight': 20, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40251
Modeling iteration {'min_child_weight': 25, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40283
Modeling iteration {'min_child_weight': 30, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40342
Modeling iteration {'min_child_weight': 20, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39897
Modeling iteration {'min_child_weight': 25, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39904
Modeling iteration {'min_child_weight': 30, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 5, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39962
Modeling iteration {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40828
Modeling iteration {'min_child_weight': 25, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40794
Modeling iteration {'min_child_weight': 30, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40815
Modeling iteration {'min_child_weight': 20, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39881
Modeling iteration {'min_child_weight': 25, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39831
Modeling iteration {'min_child_weight': 30, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39932
Modeling iteration {'min_child_weight': 20, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39587
Modeling iteration {'min_child_weight': 25, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39664
Modeling iteration {'min_child_weight': 30, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 6, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39571
Modeling iteration {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40387
Modeling iteration {'min_child_weight': 25, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40368
Modeling iteration {'min_child_weight': 30, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40402
Modeling iteration {'min_child_weight': 20, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39596
Modeling iteration {'min_child_weight': 25, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39557
Modeling iteration {'min_child_weight': 30, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39590
Modeling iteration {'min_child_weight': 20, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39341
Modeling iteration {'min_child_weight': 25, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39304
Modeling iteration {'min_child_weight': 30, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 7, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39373
Modeling iteration {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40078
Modeling iteration {'min_child_weight': 25, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40041
Modeling iteration {'min_child_weight': 30, 'eta': 0.1, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.40026
Modeling iteration {'min_child_weight': 20, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39323
Modeling iteration {'min_child_weight': 25, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39368
Modeling iteration {'min_child_weight': 30, 'eta': 0.2, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39316
Modeling iteration {'min_child_weight': 20, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39133
Modeling iteration {'min_child_weight': 25, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39165
Modeling iteration {'min_child_weight': 30, 'eta': 0.3, 'colsample_bytree': 0.9, 'max_depth': 8, 'subsample': 0.9, 'lambda': 1.0, 'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'eval_metric': 'rmse', 'objective': 'reg:linear'}, RMSLE 0.39172
"""



##############
# The XGBoost code for model training
##############


#here I have upped the iterations to 100, to see if we can go beyond the best
#trained model that was found when we ran with only 50 as the max.
n=20
l=0.2
m=8
xgb_pars = {'min_child_weight': n, 'eta': l, 'colsample_bytree': 0.9, 
      'max_depth': m,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=10,
    maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)

"""Modeling iteration , RMSLE 0.39133
#Modeling RMSLE 0.38903 for 100 training rounds and params:
n=20
l=0.3
m=8
the extra 100 iterations provided ~ a 0.002 decrease in the RMSE

next:try to retrain the model with sd=4 included! also double the rounds.
Modeling RMSLE 0.38982 for sd 4 and same params as above.
the RMSLE on the validation set was slightly higher but this model generalized to
the test data better.
0.43661

#original run

xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 6,'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)
#Feature importance for model

pred = model.predict(dtest)
pred = np.exp(pred) - 1
"""

pred = model.predict(dtest)
pred = np.exp(pred) - 1
# get data to submission format
submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id','trip_duration']
submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)
submission.to_csv("submission_cam_full_trainXGBoost_rounds200_4sd_eta2.csv", index=False)








