import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

test_dat = pd.read_csv('test.csv')
train_dat = pd.read_csv('train.csv')
submission = pd.read_csv('sample_submission.csv')

#no missing values, mostly integers with some float
train_dat.info()
train_dat.describe()


train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)



merged_dat = pd.concat([train_x, test_dat],axis=0)


cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
for column in cat_features:
	temp=pd.get_dummies(pd.Series(combine[column]))
	combine=pd.concat([combine,temp],axis=1)
	combine=combine.drop([column],axis=1)

train_x = merged_dat[:train_x.shape[0]]
test_dat = merged_dat[train_x.shape[0]:]



# train an initial rf classifier model


rf_model = RandomForestClassifier(n_estimators = 1000, n_jobs= -1)

cross_val_score(rf_model, train_x, train_y, cv=5, scoring="accuracy")


y_train_pred = cross_val_predict(rf_model, train_x, train_y, cv=5)
confusion_matrix(train_y, y_train_pred)

#get a very large number 21,000 false positives, and only 29 true negatives!
#need to alter the threshold needed for a positive prediction, or else
#we could just ignore this as the goal is a probability

#use this to get probabilities instead of just classes
rf_model.predict_proba()


#remember for the below we need to predict probabilities using	.predict_proba()
#KNeighborsClassifier

KNN_params ={
	'n_neighbors': [3,5,7],
	'weights': ['uniform', 'distance']
}

KNN_class = KNeighborsClassifier()

knn_grid = GridSearchCV(KNN_class, KNN_params, cv = 5 , n_jobs = -1)

knn_grid.fit(train_x, train_y)

#get the top tuned hyperparamaters
knn_grid.best_params_
#best C values was 10, so we should continue search further in that direction
knn_grid.best_estimator_

"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
					 metric_params=None, n_jobs=1, n_neighbors=7, p=2,
					 weights='uniform')
"""

#AdaBoostClassifier

ada_params = {
	'learning_rate' : [1.,2.,3.],
	'n_estimators': [200,500,1000,1500]
}

ada_class = AdaBoostClassifier()

ada_grid = GridSearchCV(ada_class,ada_params, cv = 5, n_jobs=-1)

ada_grid.fit(train_x, train_y)

ada_grid.best_estimator_

ada_grid.feature_importance_

"""
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
					learning_rate=1.0, n_estimators=200, random_state=None)
"""

#from sklearn.ensemble import GradientBoostingClassifier

gb_params = {
	'n_estimators' : [100,200,300],
	'learning_rate' : [.1,.2,.3],
	'max_depth' : [3,5,7]
}

gb_class = GradientBoostingClassifier()

gb_grid = GridSearchCV(gb_class, gb_params, cv = 5, n_jobs=-1)
gb_grid.fit(train_x, train_y)

gb_grid.best_estimator_

gb_grid.feature_importance_


"""

GradientBoostingClassifier(criterion='friedman_mse', init=None,
							learning_rate=0.1, loss='deviance', max_depth=3,
							max_features=None, max_leaf_nodes=None,
							min_impurity_decrease=0.0, min_impurity_split=None,
							min_samples_leaf=1, min_samples_split=2,
							min_weight_fraction_leaf=0.0, n_estimators=100,
							presort='auto', random_state=None, subsample=1.0, verbose=0,
							warm_start=False)
"""


#######
# Train the cross validated models & make predictions
#######

test_x = test_dat.drop(['id'], axis = 1)


knn_opt = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
					 metric_params=None, n_jobs=-1, n_neighbors=7, p=2,
					 weights='uniform')

knn_opt.fit(train_x, train_y)
test_y_knn = knn_opt.predict_proba(test_x)

knn_out = submission
knn_out['target'] = test_y_knn

knn_out['target'] = 1-knn_out['target']
knn_out.to_csv('knn_predictions1.csv', index=False, float_format='%.4f')



ada_opt = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
					learning_rate=1.0, n_estimators=200, random_state=None)

ada_opt.fit(train_x, train_y)
test_y_ada = ada_opt.predict_proba(test_x)

ada_out = submission
ada_out['target'] = test_y_ada
ada_out['target'] = 1-ada_out['target']


ada_out.to_csv('ada_predictions1.csv', index=False, float_format='%.4f')




gb_opt = GradientBoostingClassifier(criterion='friedman_mse', init=None,
							learning_rate=0.1, loss='deviance', max_depth=3,
							max_features=None, max_leaf_nodes=None, min_impurity_split=None,
							min_samples_leaf=1, min_samples_split=2,
							min_weight_fraction_leaf=0.0, n_estimators=100,
							presort='auto', random_state=None, subsample=1.0, verbose=0,
							warm_start=False)


gb_opt.fit(train_x, train_y)
test_y_gb = gb_opt.predict_proba(test_x)

gb_out = submission
gb_out['target'] = test_y_gb

gb_out['target'] = 1-gb_out['target']
gb_out.to_csv('gb_predictions1.csv', index=False, float_format='%.4f')





















