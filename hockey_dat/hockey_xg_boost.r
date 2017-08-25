

#set where we are working
setwd('/Users/Cam/Desktop/ds_practice/hockey_dat')

library('plyr')
library('stringr')
library('tidyverse')
library('magrittr')
library('scatterplot3d')
library('xgboost')
library('dummies')


save.image('hockey_salary')
#load in the data we generated in the random forest tutorial
#data is now successfully loaded in to dtest and dtrain
load('hockey_salary')

head(all_data)

#make sure the variables we need were brought over.
head(test_dat)
head(test_dat)
head(test_in)
predictor_columns
y_column


?xgboost

#need to make these into dummy variables before passing into xgb.DMatrix

train_in = select(train_df,one_of(predictor_columns))
test_in = select(test_df,one_of(predictor_columns))

names(train_df)[4:length(names(train_df))]
head(train_in)

#change undrafted to 0 and 1
train_in$undrafted = as.numeric(train_in$undrafted)
test_in$undrafted = as.numeric(test_in$undrafted)
#change the hand to two booleans
train_in = cbind(train_in ,dummy(train_in$Hand))
test_in = cbind(test_in ,dummy(test_in$Hand))


# Pr.St check if the pr.st are same in each 
levels(train_in$Pr.St) == levels(test_in$Pr.St)
#same. therefore we can conduct the dummy creation
train_in = cbind(train_in ,dummy(train_in$Pr.St))
test_in = cbind(test_in ,dummy(test_in$Pr.St))


#drop the pre dummies
train_in = train_in[, !(colnames(train_in) %in% c("Hand","Pr.St"))]
test_in = test_in[, !(colnames(test_in) %in% c("Hand","Pr.St"))]

tail(test_in)
#now there are no string to worry about and we can load in the numeric matrix
#note label args below different, see if syntax for one or both work.
dtrain = xgb.DMatrix(data =  as.matrix(train_in), label = train_dat$Salary)
dtest = xgb.DMatrix(data =  as.matrix(test_in), label = test_dat[,y_column])



watchlist = list(train=dtrain, test=dtest)
bst = xgb.train(data=dtrain, max.depth=8, eta=0.3, nthread = 2, nround=1000, watchlist=watchlist, objective = "reg:linear", early_stopping_rounds = 50)
#48rounds on standard
xgb.importnace(bst)
? xgb.train

XGBoost_importance = xgb.importance(feature_names = names(train_in), model = bst)
XGBoost_importance

#top predictors == xGF and DrftYr





