

#set where we are working
setwd('/Users/Cam/Desktop/ds_practice/hockey_dat')

library('plyr')
library('stringr')
library('tidyverse')
library('magrittr')
library('scatterplot3d')
library('xgboost')
library('dummies')
library('randomForest')



save.image('hockey_salary')
#load in the data we generated in the random forest tutorial
#data is now successfully loaded in to dtest and dtrain
load('hockey_salary')


# Read in the data
train = read.csv('train.csv')
head(train)
test_x = read.csv('test.csv')
head(test_x)
test_y = read.csv('test_salaries.csv')
head(test_y)

#data cleaning
#impute missing data and fix problem categorical columns
#to do this we merge the train and test data into a single set

#add train/test column
test_x$TrainTest = "test"
train$TrainTest =  "train"

test = cbind(test_y, test_x)
all_data = rbind(train,test)

# make new column for undrafted
all_data$undrafted = is.na(all_data$DftRd)

#fill the Pr.St column with 'INT' for international players
all_data$Pr.St = mapvalues(all_data$Pr.St, from = "", to="INT")

#Make team boolean columns
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

#Make position boolean columns
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



#turn the born column into 
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



#split Cntry and Nat to boolean columns

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



# impute the missing value's median for numerical columns

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


all_data = all_data[, !(colnames(all_data) %in% c("Last.Name","First.Name","Cntry","Nat","Born","Team","City","Position"))]
head(all_data)

train_dat = all_data[all_data$TrainTest == "train",]

test_dat = all_data[all_data$TrainTest == "test",]
#we lose anyone?
length(test_dat$TrainTest) + length(train_dat$TrainTest) == length(all_data$TrainTest)


#drop the train/test split columns
train_dat = train_dat[, !(colnames(train_dat) %in% c("TrainTest"))]
test_dat = test_dat[, !(colnames(test_dat) %in% c("TrainTest"))]


y_column = c("Salary")
all_columns = names(train_dat)
predictor_columns = all_columns[all_columns != y_column]
predictor_columns


###################################
###################################
#New XGBoost code
###################################
###################################
#Additional XGBoost Cleaning
#need to make these into dummy variables before passing into xgb.DMatrix

train_in = select(train_dat,one_of(predictor_columns))
test_in = select(test_dat,one_of(predictor_columns))

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

XGBoost_importance = xgb.importance(feature_names = names(train_in), model = bst)
XGBoost_importance[1:10]

#top predictors == xGF and DrftYr


#graph the XGboost top predictors in 2 and 3 dimensions, compare and contrast with the random forest top predictors

color.gradient <- function(x, colors=c("green", "yellow", "red"), colsteps=100) {
  return( colorRampPalette(colors) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}

sd3 = scatterplot3d(graph_all_data$xGF, graph_all_data$DftYr,  graph_all_data$Salary, # x y and z 
                 pch=19, 
                 type="h", 
			cex.axis=0.5,
			las=1,
			lty.hplot=2,           
                	color=color.gradient(all_data$Salary,c("black","salmon")), 
			main="Interaction of age, goals and salary",
                 zlab="Salary",
                xlab="xGF:",
			ylab="Draft Year",
			grid=TRUE)	
	
sd3.coords = sd3$xyz.convert(graph_all_data$xGF, graph_all_data$DftYr,  graph_all_data$Salary) # convert 3D coords to 2D projection
text(sd3.coords$x, sd3.coords$y,labels= graph_all_data$Last.Name,cex=.5, pos=4)  

