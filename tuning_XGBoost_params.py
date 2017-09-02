
"""
General Approach for Parameter Tuning

We will use an approach similar to that of GBM here. The various steps to be performed are:

Choose a relatively high learning rate. Generally a learning rate of 0.1 works but somewhere 
between 0.05 to 0.3 should work for different problems. Determine the optimum number of trees 
for this learning rate. XGBoost has a very useful function called as “cv” which performs cross-validation 
at each boosting iteration and thus returns the optimum number of trees required.
Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) 
for decided learning rate and number of trees. Note that we can choose different parameters to 
define a tree and I’ll take up an example here.
Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity 
and enhance performance.
Lower the learning rate and decide the optimal parameters.
Let us look at a more detailed step by step approach.
"""


### below may be helpful


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

"""

Step 1: Fix learning rate and number of estimators for tuning tree-based parameters

In order to decide on boosting parameters, we need to set some initial values of other parameters. 
Lets take the following values:

max_depth = 5 : This should be between 3-10. I’ve started with 5 but you can choose a different 
number as well. 4-6 can be good starting points.
min_child_weight = 1 : A smaller value is chosen because it is a highly imbalanced class problem 
and leaf nodes can have smaller size groups.
gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
subsample, colsample_bytree = 0.8 : This is a commonly used used start value. Typical values range between
0.5-0.9.
scale_pos_weight = 1: Because of high class imbalance.
Please note that all the above are just initial estimates and will be tuned later. 
Lets take the default learning rate of 0.1 here and check the optimum number of trees 
using cv function of xgboost. The function defined above will do it for us.
"""
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

"""
Step 2: Tune max_depth and min_child_weight
This is in lieu of the for loop that I employed on the initial XGBoost taxi data run.
Spits out a big matrix of paramater combinations and you select the best ones, based on the value


We tune these first as they will have the highest impact on model outcome. To start with, 
let’s set wider ranges and then we will perform another iteration for smaller ranges.

Important Note: I’ll be doing some heavy-duty grid searched in this section which can take 15-30 mins 
or even more time to run depending on your system. You can vary the number of values you are testing 
based on what your system can handle.
"""
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
"""
Here, we have run 12 combinations with wider intervals between values. 
The ideal values are 5 for max_depth and 5 for min_child_weight. 
Lets go one step deeper and look for optimum values. We’ll search for values 1 above and 
below the optimum values because we took an interval of two.
"""
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
"""

Here, we get the optimum values as 4 for max_depth and 6 for min_child_weight. 
Also, we can see the CV score increasing slightly. Note that as the model performance increases, 
it becomes exponentially difficult to achieve even marginal gains in performance. 
You would have noticed that here we got 6 as optimum value for min_child_weight 
but we haven’t tried values more than 6. We can do that as follow:.

"""
param_test2b = {
 'min_child_weight':[6,8,10,12]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2b.fit(train[predictors],train[target])
modelfit(gsearch3.best_estimator_, train, predictors)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

"""
We see 6 as the optimal value.
 
"""

"""

Step 3: Tune gamma

Now lets tune gamma value using the parameters already tuned above. 
Gamma can take various values but I’ll check for 5 values here. 
You can go into more precise values as.

"""
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

"""
This shows that our original value of gamma, i.e. 0 is the optimum one. 
Before proceeding, a good idea would be to re-calibrate the number of boosting rounds for the 
updated parameters."""


xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, train, predictors)


"""
Here, we can see the improvement in score. So the final parameters are:

max_depth: 4
min_child_weight: 6
gamma: 0"""

"""Step 4: Tune subsample and colsample_bytree

The next step would be try different subsample and colsample_bytree values. 
Lets do this in 2 stages as well and take values 0.6,0.7,0.8,0.9 for both to start with."""


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_





param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])


"""
Step 5: Tuning Regularization Parameters

Next step is to apply regularization to reduce overfitting.
Though many people don’t use this parameters much as gamma provides a substantial way of controlling 
complexity. But we should always try it. I’ll tune ‘reg_alpha’ value here and leave it upto you to 
try different values of ‘reg_lambda’."""

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


"""We can see that the CV score is less than the previous case. 
But the values tried are very widespread, we should try values closer to the optimum here (0.01) 
to see if we get something better."""

param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


#You can see that we got a better CV. 
#Now we can apply this regularization in the model and look at the impact:

xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictors)



"""Step 6: Reducing Learning Rate

Lastly, we should lower the learning rate and add more trees.
 Lets use the cv function of XGBoost to do the job again."""

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)


"""
Now we can see a significant boost in performance and the effect of parameter tuning is clearer.

As we come to the end, I would like to share 2 key thoughts:

It is difficult to get a very big leap in performance by just using parameter tuning or slightly 
better models. The max score for GBM was 0.8487 while XGBoost gave 0.8494. 
This is a decent improvement but not something very substantial.
A significant jump can be obtained by other methods like feature engineering, 
creating ensemble of models, stacking, etc
You can also download the iPython notebook with all these model codes from my GitHub account. 
For codes in R, you can refer to this article."""

















