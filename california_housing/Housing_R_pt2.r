setwd('/Users/Cam/Desktop/Code/bin/kaggle_code/california_housing')
library(tidyverse)
library(reshape2)
library(boot)
library(randomForest)

housing = read.csv('./housing.csv')

head(housing)

summary(housing)

colnames(housing)

par(mfrow=c(2,5))

ggplot(data = melt(housing), mapping = aes(x = value)) + 
    geom_histogram(bins = 30) + facet_wrap(~variable, scales = 'free_x')

housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms , na.rm = TRUE)

housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households

drops = c('total_bedrooms', 'total_rooms')

housing = housing[ , !(names(housing) %in% drops)]

head(housing)

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


#some people like weird r format like this... I find it causes headaches
#rf_model = randomForest(median_house_value~. , data = train, ntree =500, importance = TRUE)

######
# XG Boost
######
# http://cran.fhcrc.org/web/packages/xgboost/vignettes/xgboost.pdf
library(xgboost)


dtrain = xgb.DMatrix(data =  as.matrix(train_x), label = train_y )
dtest = xgb.DMatrix(data =  as.matrix(test_x), label = test_y)


watchlist = list(train=dtrain, test=dtest)
bst = xgb.train(data=dtrain, max.depth=8, eta=0.3, nthread = 2, nround=1000, watchlist=watchlist, objective = "reg:linear", early_stopping_rounds = 50)


#make the above into a graid serch

XGBoost_importance = xgb.importance(feature_names = names(train_x), model = bst)
XGBoost_importance[1:10]


#########
# try adding one more model
#########



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


y_pred = predict(rf_model , test_x)
test_mse = mean(((y_pred - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse




########
# Ensemble the 3 models together, see if better predictions
########
