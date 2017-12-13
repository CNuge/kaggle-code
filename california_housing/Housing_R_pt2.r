setwd('/Users/Cam/Desktop/Code/bin/kaggle_code/california_housing')
library(tidyverse)
library(reshape2)
library(boot)
library(randomForest)

housing = read.csv('./housing.csv')

housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms , na.rm = TRUE)

housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households

drops = c('total_bedrooms', 'total_rooms')

housing = housing[ , !(names(housing) %in% drops)]

categories = unique(housing$ocean_proximity)
#split the categories off
cat_housing = data.frame(ocean_proximity = housing$ocean_proximity)


for(cat in categories){
    cat_housing[,cat] = rep(0, times= nrow(cat_housing))
}
head(cat_housing) #see the new columns on the right

for(i in 1:length(cat_housing$ocean_proximity)){
    cat = as.character(cat_housing$ocean_proximity[i])
    cat_housing[,cat][i] = 1
}


cat_columns = names(cat_housing)
keep_columns = cat_columns[cat_columns != 'ocean_proximity']
cat_housing = select(cat_housing,one_of(keep_columns))
drops = c('ocean_proximity','median_house_value')
housing_num =  housing[ , !(names(housing) %in% drops)]


scaled_housing_num = scale(housing_num)

cleaned_housing = cbind(cat_housing, scaled_housing_num, median_house_value=housing$median_house_value)


set.seed(19) # Set a random seed so that same sample can be reproduced in future runs

sample = sample.int(n = nrow(cleaned_housing), size = floor(.8*nrow(cleaned_housing)), replace = F)
train = cleaned_housing[sample, ] #just the samples
test  = cleaned_housing[-sample, ] #everything but the samples


train_y = train[,'median_house_value']
train_x = train[, names(train) !='median_house_value']

test_y = test[,'median_house_value']
test_x = test[, names(test) !='median_house_value']


########
# Random Forest Model
########
rf_model = randomForest(train_x, y = train_y , ntree = 500, importance = TRUE)

names(rf_model) #these are all the different things you can call from the model.

importance_dat = rf_model$importance
importance_dat

sorted_predictors = sort(importance_dat[,1], decreasing=TRUE)
sorted_predictors

oob_prediction = predict(rf_model) #leaving out a data source forces OOB predictions

#you may have noticed that this is avaliable using the $mse in the model options.
#but this way we learn stuff!
train_mse = mean(as.numeric((oob_prediction - train_y)^2))
oob_rmse = sqrt(train_mse)
oob_rmse


y_pred_rf = predict(rf_model , test_x)
test_mse = mean(((y_pred_rf - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse # ~48620




######
# XG Boost
######
# http://cran.fhcrc.org/web/packages/xgboost/vignettes/xgboost.pdf
library(xgboost)

#put into the xgb matrix format
dtrain = xgb.DMatrix(data =  as.matrix(train_x), label = train_y )
dtest = xgb.DMatrix(data =  as.matrix(test_x), label = test_y)

# these are the datasets the rmse is evaluated for at each iteration
watchlist = list(train=dtrain, test=dtest)

# try 1 -off a set of paramaters I know work pretty well generally

bst = xgb.train(data = dtrain, 
				max.depth = 8, 
				eta = 0.3, 
				nthread = 2, 
				nround = 1000, 
				watchlist = watchlist, 
				objective = "reg:linear", 
				early_stopping_rounds = 50)

# try a 'slower learning' model. The up and down weights for each iteration are smaller
# we also use more iterations

bst_slow = xgb.train(data = dtrain, 
						max.depth=6, 
						eta = 0.01, 
						nthread = 2, 
						nround = 10000, 
						watchlist = watchlist, 
						objective = "reg:linear", 
						early_stopping_rounds = 50)

# note the best iteration is not the last iteration. 
XGBoost_importance = xgb.importance(feature_names = names(train_x), model = bst_slow)
XGBoost_importance[1:10]

# rmse: 45225.968750 # max.depth=5
#last week: $48620 with random forest
# an improvement of ~$3400 in average error. Wait! What we have done here is fit to the training set (leading to model overfit). Need to work with a validation set, then only at the end evaluate the model performance against the test set.


####
# Proper use - validation set
####

#validation set - Another subset of our data that is witheld from the training algorithm, but compared against at each iteration to see how

#make validation set

set.seed(19) # Set a random seed so that same sample can be reproduced in future runs

sample = sample.int(n = nrow(train), size = floor(.8*nrow(train)), replace = F)

train_t = train[sample, ] #just the samples
valid  = train[-sample, ] #everything but the samples

train_y = train_t[,'median_house_value']
train_x = train_t[, names(train) !='median_house_value']

valid_y = valid[,'median_house_value']
valid_x = valid[, names(test) !='median_house_value']

gb_train = xgb.DMatrix(data = as.matrix(train_x), label = train_y )
gb_valid = xgb.DMatrix(data = as.matrix(valid_x), label = valid_y)

# train xgb, evaluating against the validation
watchlist = list(train = gb_train, valid = gb_valid)

bst_slow = xgb.train(data= gb_train, 
						max.depth = 10, 
						eta = 0.01, 
						nthread = 2, 
						nround = 10000, 
						watchlist = watchlist, 
						objective = "reg:linear", 
						early_stopping_rounds = 50)

# error, need the matrix format
y_hat = predict(bst_slow, test_x)

# recall we ran the following to get the test data in the right format:
# dtest = xgb.DMatrix(data =  as.matrix(test_x), label = test_y)
# here I have it with the label taken off, just to remind us its external data xgb would ignore the label though during predictions
dtest = xgb.DMatrix(data =  as.matrix(test_x))

#test the model on truly external data

y_hat_valid = predict(bst_slow, dtest)

test_mse = mean(((y_hat_valid - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse 
# ~47507.09 This is higher then on the first run through, but we can be confident that the improved score is not due to overfit thanks to our use of a validation set! point out that this is evidence of how a lower rmse isn't necessarily better, as we now have more confidence in external predictions.2.3% improvement over a basic random forest... is it worth the effort? The answer to this question always depends on the purpose of the model.


###
# Grid search first principles 
###

max.depths = c(3, 5, 7, 9)
etas = c(0.01, 0.001, 0.0001)

best_params = 0
best_score = 0

count = 1
for( depth in max.depths ){
	for( num in etas){

		bst_grid = xgb.train(data = gb_train, 
								max.depth = depth, 
								eta=num, 
								nthread = 2, 
								nround = 10000, 
								watchlist = watchlist, 
								objective = "reg:linear", 
								early_stopping_rounds = 50, 
								verbose=0)

		if(count == 1){
			best_params = bst_grid$params
			best_score = bst_grid$best_score
			count = count + 1
			}
		else if( bst_grid$best_score < best_score){
			best_params = bst_grid$params
			best_score = bst_grid$best_score
		}	
	}
}

best_params
best_score
#valid-rmse: 47033.28

# max_depth of 9, eta of 0.01
bst_tuned = xgb.train( data = gb_train, 
						max.depth = 9, 
						eta = 0.01, 
						nthread = 2, 
						nround = 10000, 
						watchlist = watchlist, 
						objective = "reg:linear", 
						early_stopping_rounds = 50)

y_hat_xgb_grid = predict(bst_tuned, dtest)

test_mse = mean(((y_hat_xgb_grid - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse # test-rmse: 46675
# By tuning the hyperparamaters we have moved to a 4% improvement over random forest



#######
# tweak the hyperparamaters using a grid search
# The caret package (short for classification and regression training)
######

library(caret) 

# look up the model we are running to see the paramaters
modelLookup("xgbLinear")
 
# set up all the pairwise combinations

xgb_grid_1 = expand.grid(nrounds = c(1000,2000,3000,4000) ,
							eta = c(0.01, 0.001, 0.0001),
							lambda = 1,
							alpha = 0)
xgb_grid_1


#here we do one better then a validation set, we use cross validation to 
#expand the amount of info we have!
xgb_trcontrol_1 = trainControl(method = "cv",
								number = 5,
								verboseIter = TRUE,
								returnData = FALSE,
								returnResamp = "all", 
								allowParallel = TRUE)
								

######
#below a grid-search, cross-validation xgboost model in caret
######
# train the model for each parameter combination in the grid, using CV to evaluate on multiple folds. Make sure your laptop is plugged in or RIP battery.

# note how this is now the caret train function
?train

xgb_train_1 = train(x = as.matrix(train_x),
					y = train_y,
					trControl = xgb_trcontrol_1,
					tuneGrid = xgb_grid_1,
					method = "xgbLinear",
					maximize = "RMSE",
					early_stopping_rounds = 50,
					max.depth = 5)

names(xgb_train_1)
xgb_train_1$bestTune
xgb_train_1$method
summary(xgb_train_1)


#alternatively, you can 'narrow in' on the best paramaters by taking a range of options around the best values found and seeing if high resolution tweaks can provide even further improvements.

xgb_cv_yhat = predict(xgb_train_1 , as.matrix(test_x))


test_mse = mean(((xgb_cv_yhat - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse # 48450... not really an improvement!



########
# Ensemble the models together, 
# strategy for when accuracy is more important then knowing the best predictors
########


y_pred_rf
y_hat_valid
xgb_cv_yhat
y_hat_xgb_grid
length(y_hat_xgb_grid)


blend_pred = (y_hat * .25) + (y_pred_rf * .25) + (xgb_cv_yhat * .25) + (y_hat_xgb_grid * .25)
length(blend_pred)

blend_test_mse = mean(((blend_pred - test_y)^2))
blend_test_rmse = sqrt(blend_test_mse)
blend_test_rmse # 45907 by averaging just 4 predictors we have dropped the rmse ~1.5% lower then the best scoring of the 4 models. This does come at a cost though, we now can't make accurate inferrences about the best predictors!

#next step - you can grid search the weights of the ensemble to try and drop the rmse further!



#######
# Unrelated note
#######
# install.packages("devtools")
# devtools::install_github("brooke-watson/BRRR")


##  [1] "twochainz"  "twochainz1" "bigboi"     "biggie"     "bigsean"   
##  [6] "bigsean1"   "bigsean2"   "bigsean3"   "bigsean4"   "bigsean5"  
## [11] "bigshaq"    "birdman"    "birdman1"   "birdman2"   "busta"     
## [16] "chance"     "desiigner"  "diddy"      "drake"      "drake1"    
## [21] "drummaboy"  "fetty"      "flava"      "future"     "gucci"     
## [26] "gucci1"     "gucci2"     "jayz"       "jayz1.wav"  "kendrick"  
## [31] "khaled"     "khaled1"    "khaled2"    "khaled3"    "liljon"    
## [36] "liljon1"    "nicki"      "pitbull"    "ross"       "ross1"     
## [41] "schoolboy"  "snoop"      "soulja"     "takeoff"    "tpain.wav" 
## [46] "traviscott" "treysongz"  "trick"      "waka"       "weezy"     
## [51] "yg"


skrrrahh(0) #random
skrrrahh('tpain')
skrrrahh('ross')
skrrrahh('future')
skrrrahh('traviscott')
skrrrahh('desiigner')
skrrrahh('bigshaq')
skrrrahh('twochainz1')
skrrrahh('bigsean3')

#demo the machine learning benefit by running after an xgboost training line.