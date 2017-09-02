setwd('/Users/Cam/Desktop/ds_practice/hockey_dat')
getwd()
ls()
options(prompt='R> ')
options(continue = '\t')

save.image('hockey_salary')
load('hockey_salary')
install.packages('ggplot2')
# load libraries
library('randomForest')
library('plyr')
library('stringr')
library('tidyverse')
library('magrittr')
library('scatterplot3d')
library('xgboost')

#3 Read in the data
train = read.csv('train.csv')
head(train)
test_x = read.csv('test.csv')
head(test_x)
test_y = read.csv('test_salaries.csv')
head(test_y)


#4 impute missing data and fix problem categorical columns
#to do this we merge the train and test data into a single set

#######
# add a column that labels data as members of the train or test populations.
# then work with the all_data for this section, and split after step 4.
######

#add train/test column
test_x$TrainTest = "test"
train$TrainTest =  "train"

test = cbind(test_y, test_x)
all_data = rbind(train,test)

#where are there missing values?

all_missing_list=  colnames(all_data)[colSums(is.na(all_data)) > 0]


#4aDONE
# make new column for undrafted

all_data$undrafted = is.na(all_data$DftRd)


#4cDONE fill the Pr.St column with 'INT' for international players

all_data$Pr.St = mapvalues(all_data$Pr.St, from = "", to="INT")
?mapvalues

#4dDONE Make team boolean columns
#get the unique list of team acronymns
teams = c()
for( i in levels(all_data$Team)){
	x = strsplit(i, "/")
	for(y in x){
		teams = c(teams, y)
	}
}
teams = unique(teams)

# add columns with the team names as the header and 0 as values
for(team in teams){
	all_data[,team] = 0
}

#iterate through and record the teams for each player
for(i in 1:length(all_data$Team)){
	teams_of_person = strsplit(as.character(all_data$Team[i]), "/")[[1]]
	for(x in teams_of_person){
		all_data[,x][i] = 1	
	}
}


#4e Make position boolean columns
pos = c()
for( i in levels(all_data$Position)){
	x = strsplit(i, "/")
	for(y in x){
		pos = c(pos, y)
	}
}
pos = unique(pos)

# add columns with the pos names as the header and 0 as values
for(position in pos){
	all_data[,position] = 0
}

#iterate through and record the position(s) for each player
for(i in 1:length(all_data$Position)){
	pos_of_person = strsplit(as.character(all_data$Position[i]), "/")[[1]]
	for(x in pos_of_person){
		all_data[,x][i] = 1	
	}
}



#4fDONE turn the born column into 
# an age column 
# 3 integer columns year:month:date

bday_parts = str_split_fixed(all_data$Born, "-",3)

#adjust year column to account for missing digits
table(birth_year)
birth_year = c()
for(year in bday_parts[,1]){
	if(as.numeric(year) < 10){
		yr = paste("20", year, sep="")
		birth_year = c(birth_year, yr)
	}else{
		yr = paste("19",year, sep="")
		birth_year = c(birth_year, yr)
	}
}

all_data$birth_year = as.numeric(birth_year)
all_data$birth_month = as.numeric(bday_parts[,2])
all_data$birth_day = as.numeric(bday_parts[,3])



#4gDONE split Cntry and Nat to boolean columns

birth_country = levels(all_data$Cntry)
# add columns with the country of birth options
# note the Estonia for Uncle Leo
for(country in birth_country){
	c = paste("born", country, sep="_")

	all_data[,c] = 0
}

#iterate through and record the birth country of each player
for(i in 1:length(all_data$Cntry)){
	birth_country = all_data$Cntry[i]
	c = paste("born", birth_country, sep="_")
	all_data[,c][i] = 1	
}


nationality = levels(all_data$Nat)
for(country in nationality){
	c = paste("nation", country, sep="_")
	all_data[,c] = 0
}

#iterate through and record the birth country of each player
for(i in 1:length(all_data$Nat)){
	nationality = all_data$Nat[i]
	c = paste("nation", nationality, sep="_")
	all_data[,c][i] = 1	
}



#4hDONE impute the missing value's median for numerical columns

#fill median values
#loop through the dataframe, filling each column with the median of 
#the existing values for the entire dataset
#where are there still missing values?

all_missing_list =  colnames(all_data)[colSums(is.na(all_data)) > 0]
length(all_missing_list) == 0
#if above true all values are imputed!

for( i in 1:length(all_missing_list)){
	#get the global median
	median_all = median(all_data[,all_missing_list[i]], na.rm =TRUE)
	#imput the missing values with the column's median
	all_data[,all_missing_list[i]][is.na(all_data[,all_missing_list[i]])] = median_all
}




#5 Graphical analyses & fun hypotheses

#how many players from each country?
#Canadians are over half of the NHL!
barplot(sort(table(all_data$Nat),decreasing=TRUE), horiz=TRUE, las=1,col=c("red","blue4","blue","red3","skyblue"))

#age distribution
table(all_data$birth_year)
#behold the beaitiful outlier that is Jagr, note there are 15 nhl players born
#in 2000 or later! This makes me feel old
hist(all_data$birth_year, breaks=28, col="skyblue", xlab='Year of birth', main='Distribution of NHL player Age;\nAKA Jaromir Jagr the ancient outlier')

#jagr aside, there is some left skew to the plot, father time comes for us all but some seem better at delaying the effects!

#median birth year is 1992
median(all_data$birth_year)

#correlation of age and salary
#range of players making between 575,000 and 14,000,000
#that is quite the distribution!
summary(all_data$Salary)
hist(all_data$Salary)


#Does scoring more directly lead to getting more money?
plot(all_data$G, all_data$Salary, xlab='goals scored', ylab='money earned')
#few players with lots of goals and not much financial compensation

plot(all_data$G, all_data$Salary, xlab='goals scored', ylab='money earned', main="who are the outliers?")
text(all_data$G, all_data$Salary, labels=all_data$Last.Name, cex=0.7, pos = 3)
# big outliers in the bottom right are Auston Matthews and Patrick Laine. These were the first and second overall picks in the previous years entry draft. One could argue that they provide the most cost effective scoring in the league. This highlights the value of the entry draft in the salary cap era, it provides high quality players at rock bottom prices. There are clearly multiple factors that are affecting how much a player makes, and things like rookie contracts cause there to be strong interaction effects between on ice metrics of success (such as goals, corsi, +/-, time on ice, etc.) and other considerations such as the players age.


matthews_goal_price =925000/40
matthews_goal_price #only $23,125 a goal! what a deal!

crosby_goal_price = 10900000/44
crosby_goal_price 
#Crosby was paid $247,727  a goal. The median house price in Pittsburgh is $123,500 so Sid could by two houses for every goal he scores. 
Crosby_homes = 44*4
pittsburgh_households = 143739
Crosby_market_share = Crosby_homes/pittsburgh_households
Crosby_market_share
#So over his current 12 year contract he could easily by up 1% of the total pittsburgh housing market(before taxes)


##############
# 3-d plots
##############

color.gradient <- function(x, colors=c("green", "yellow", "red"), colsteps=100) {
  return( colorRampPalette(colors) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}

sd3 = scatterplot3d(all_data$G, all_data$birth_year,  all_data$Salary, # x y and z 
                 pch=19, 
                 type="h", 
			cex.axis=0.5,
			las=1,
			lty.hplot=2,           
                	color=color.gradient(all_data$Salary,c("black","salmon")), 
			main="Interaction of age, goals and salary",
                 zlab="Salary",
                xlab="Goals",
			ylab="Birth Year",
			grid=TRUE)	
	
sd3.coords = sd3$xyz.convert(all_data$G, all_data$birth_year,  all_data$Salary) # convert 3D coords to 2D projection
text(sd3.coords$x, sd3.coords$y,labels=all_data$Last.Name,cex=.5, pos=4)  

#Few interesting things we can pull from the noise here, besides the already noted Laine-Matthews proximity look at the proximity of Malkin and Ovechkin in the middle. These were the #1 and #2 draft picks in the 2004 draft! So as one may expect, if Laine and Matthews don't get horribly injured then their contracts and on ice productivity may remain highly correlated as they age ( this makes sense when you consider contract negotiation would invovle agents and teams putting large weight on the contracts of 'comparable players')

#note how close Subban and Weber are in 3-d space (top left) these are two players that were traded for one another in the previous off season

#Also, Towes and Kane (the twin towers of Chicago's salary cap hell) stick far above everyone else! Kopitar is also looking pretty overpaid considering his numbers line him up with the defencemen and not the goal scorers on the graph!

# two main classes of players seem to stick out from the glob in the middle, vetran defenceman in the top left (Seabrook, Suter, Weber, Suban) and young goal scorers in the bottom right (McDavid, Laine, Matthews, Kucherov). I suspect the random forest regression will be able to accomplish what I just did by eye, using the large number of on ice metrics we have (not just goals!) and developing bins that we could in plain english think of as 'vetran defenceman', 'young goal scorer', 'aging star', 'goon' etc. With these player profiles modelled. I suspect we will be able to get some extremly accurate salary predictions.



score_3d = scatterplot3d(all_data$TOI, all_data$SCF,  all_data$Salary, # x y and z 
                 pch=19, 
                 type="h", 
			cex.axis=0.5,
			las=1,
			lty.hplot=2,           
                	color=color.gradient(all_data$Salary, colors=c("black","skyblue")), 
			main="More linear relationships:\nSalary, time on ice, and scoring chances",
                 zlab="Salary",
                xlab="Time on ice(s)",
			ylab="scoring chances while player on ice",
			grid=TRUE)	
	
score_3d.coords = score_3d$xyz.convert(all_data$TOI, all_data$SCF,  all_data$Salary) # convert 3D coords to 2D projection
text(score_3d.coords$x, score_3d.coords$y,labels=all_data$Last.Name,cex=.5, pos=4)  

#here we see a much more strong relationship, guys spending more time on the ice, and helping their teams generate scoring chances are getting paid more. Apologies to Anze Kopitar for my earlier dig, looks like he provides more value than the goals analysis alone indicated!
 
#Toronto Maple leafs fans look at Nikita Zaitsev lurking at a low salary in the high minutes/high scoring chances region of the graph! That is a young defencemen giving quality minutes on a level that is approaching some of the top vetran d men in the league.



#6 Split train/test into two dfs again

#Remove the columns we replaced in step 4 (don't want 'Sidney' to be best predictor of salary), city for same reason, Cole Harbour NS and sililar small towns could cause overfit
#write the whole adjusted dataframe to a file
write.csv(all_data,'all_data_manipulated.csv')
all_data = all_data[, !(colnames(all_data) %in% c("Last.Name","First.Name","Cntry","Nat","Born","Team","City","Position"))]
head(all_data)

train_dat = all_data[all_data$TrainTest == "train",]

test_dat = all_data[all_data$TrainTest == "test",]
#we lose anyone?
length(test_dat$TrainTest) + length(train_dat$TrainTest) == length(all_data$TrainTest)


#drop the train/test split columns
train_dat = train_dat[, !(colnames(train_dat) %in% c("TrainTest"))]
test_dat = test_dat[, !(colnames(test_dat) %in% c("TrainTest"))]


#7 feature selection
#now lets run an iterative rf regression

############
# iterative randomforest drop
############
#this function will perform a random forest regression on the data, drop the lowest
#predictor, and then repeat the process until the best predictor is identified
#it will return a dataframe with the pseudo-rsquared
#note the ntree parameter in the loop, tune this to the desired size


y_column = c("Salary")
all_columns = names(train_dat)
predictor_columns = all_columns[all_columns != y_column]
predictor_columns


# % Var explained: 64.19 by all predictors on first run
rf_result = randomForest(Salary~., data = train_dat, ntree = 1000, importance = TRUE)
first_importance_list = importance(rf_result)
first_importance_list  = sort(first_importance_list[,1], decreasing=TRUE)
first_importance_list



#####################
# R random forest cross validation for feature selection
#####################

#This function shows the cross-validated prediction performance of models with sequentially reduced number of predictors (ranked by variable importance) via a nested cross-validation procedure.

?rfcv
rf_cv_feature_selection =  rfcv(train_dat[,predictor_columns],train_dat[,y_column], cv.fold=5,scale='log', step=.25, recursive=TRUE ,ntree=1000)


#this provides a good range of where we should be looking for the optimal MSE
#with these data we can then target the range  my feature selection below.
#starting with x number of top predictors and iteratively dropping them.
rf_cv_feature_selection$n.var
rf_cv_feature_selection$error.cv

# step=10 , recursive=TRUE)
#step can be a fraction if scale='log' or a number to drop if scale!='log'




#####################
# my feature selection through continued reduction
#####################

#function to run random forest, then drop worst predictor and repeat

rf_train_test_feature_selection = function(train_df, test_df, y_column , predictor_columns){
	outdf = data.frame(size=numeric(),t_mse=numeric(),lowest_predictor=character())
	cols_to_use = c(y_column, predictor_columns)
	rf_input=select(train_df,one_of(cols_to_use))	
	while(length(names(rf_input)) > 2){
		#train the model
		#I can't find the cause of the bug but I have to hard code in salary :/
		rf_result = randomForest(Salary~., data = rf_input, ntree = 10000, importance = TRUE)
		test_prediction	= predict(rf_result, test_df[,predictor_columns])	
		test_mse = mean((test_prediction - test_df[, y_column])^2)
		
		#find the worst predictor, add its name to the dataframe and drop column
		importance_dat = rf_result$importance
		sorted_predictors = sort(importance_dat[,1], decreasing=TRUE)
		worst_pred = names(sorted_predictors[length(sorted_predictors)])
		out_line = data.frame(size=length(sorted_predictors), t_mse=test_mse, lowest_predictor=worst_pred)
		print(out_line)
		outdf = rbind(outdf,out_line)
		rf_input = rf_input[, !colnames(rf_input) %in% worst_pred]
		}
	return(outdf)
	}


#run the iterative dropfunction, performing a rf regression for each feature set
model_selection = rf_train_test_feature_selection(train_dat, test_dat, y_column , predictor_columns)
save.image('hockey_salary')
#look at the plot, which number of features gives the best mse on the test data?
plot(model_selection$size, model_selection$t_mse)
#improve this plot
min(model_selection$t_mse)
#2.476437e+12

# Optimal model at size of row:
#67   153 2.476437e+12       nation_DEU
#select these features down and retrain the model to get the list of best predictors and the model produced.

## 2nd place the team's unblocked shot attempts (Fenwick, USAT) while this player was on the ice!
# be sure tho not lose the top predictor

'%!in%' = function(x,y)!('%in%'(x,y))
all_columns[all_columns %!in% model_selection$lowest_predictor]
#last predictor standing was SF



# final model predictors
x = model_selection$lowest_predictor[67:length(model_selection$lowest_predictor)]
write.csv(x,"temp.txt")
all_top_preds= append(as.character(x),'SF' )

?randomForest
?rfcv
test_final_input=select(test_df,one_of(all_top_preds))
trian_final_input=select(train_df,one_of("Salary",all_top_preds))

#final rf run
final_rf_result = randomForest(Salary~., data = trian_final_input, ntree = 10000, importance = TRUE)
final_rf_result
x=sqrt(1.75511e+12)
x
final_importance_list  = sort(importance(final_rf_result)[,1], decreasing=TRUE)
final_importance_list

test_prediction	= predict(final_rf_result, test_df[,predictor_columns])

test_rmse = sqrt(mean((test_prediction - test_df[, y_column])^2))
test_rmse