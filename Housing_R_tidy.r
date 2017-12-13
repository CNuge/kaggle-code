##############################
## tidyverse version of Housing_R.r
##
## Karl Cottenie
##
## 2017-12-11
##
##############################

library(tidyverse)
library(viridis)
# + scale_color/fill_viridis(discrete = T/F)
theme_set(theme_light())

# Startup ends here

# KC: how fast does this code run?
# select from this line to the next proc.time statement, and execute the code
start_time_tidy = proc.time()

housing.tidy = read_csv('housing.csv')

housing.tidy # KC: but not really necessary, bc you can just double click on it in line 17

summary(housing.tidy)

# par(mfrow=c(2,5))
# KC: I don't think par influences ggplot statements

# colnames(housing), not really needed in tidyverse
# KC: housing will give you all the column names
# piping will recognize the column names anyway

housing.tidy %>% gather(longitude:median_house_value, key = "variable", value = "value") %>%
  ggplot(aes(x = value)) + 
  geom_histogram(bins = 30) + facet_wrap(~ variable, scales = 'free_x')

housing.tidy %>% ggplot(aes(x = ocean_proximity)) +
  geom_bar() # KC: missing from Housing_R.r

housing.tidy %>% mutate(total_bedrooms_c = ifelse(is.na(total_bedrooms), 
                                                  median(total_bedrooms, na.rm = T),
                                                  total_bedrooms)) %>%
  summary()
# KC: this is just to show how I would check whether it did it correctly

housing.tidy = housing.tidy %>% 
  mutate(total_bedrooms = ifelse(is.na(total_bedrooms), 
                                 median(total_bedrooms, na.rm = T),
                                 total_bedrooms),
         mean_bedrooms = total_bedrooms/households,
         mean_rooms = total_rooms/households) %>%
  select(-c(total_rooms, total_bedrooms))

# KC: this is one fast way to do it, but relies on some knowledge of an obscure function
# cat_housing.tidy = housing.tidy %>%
#   model.matrix( ~ ocean_proximity - 1, data = .) %>%
#   as.tibble()
# 
# names(cat_housing.tidy) = names(cat_housing.tidy) %>% # cleaning up the column names
#   strsplit("ocean_proximity") %>% map_chr(function(x) x[2])

# KC: this would be a second option, using a very general approach
categories = unique(housing.tidy$ocean_proximity) # all categories
cat_housing.tidy = categories %>% # compare the full vector against each category consecutively
  lapply(function(x) as.numeric(housing.tidy$ocean_proximity == x)) %>% # convert to numeric
  do.call("cbind", .) %>% as.tibble() # clean up
colnames(cat_housing.tidy) = categories # make nice column names

cleaned_housing.tidy = housing.tidy %>% 
  select(-c(ocean_proximity, median_house_value)) %>%
  scale() %>% as.tibble() %>%
  bind_cols(cat_housing.tidy) %>%
  add_column(median_house_value = housing.tidy$median_house_value)

cleaned_housing.tidy %>% summary()

running_time_tidy = proc.time() - start_time_tidy
# KC: this is the end of the timing event
# 3 times faster compared to base R code

set.seed(19) # Set a random seed so that same sample can be reproduced in future runs

sample = sample.int(n = nrow(cleaned_housing.tidy), size = floor(.8*nrow(cleaned_housing.tidy)), replace = F)
train = cleaned_housing.tidy[sample, ] #just the samples
test  = cleaned_housing.tidy[-sample, ] #everything but the samples

head(train)

nrow(train) + nrow(test) == nrow(cleaned_housing.tidy)

library('boot')

?cv.glm # note the K option for K fold cross validation

glm_house = glm(median_house_value~median_income+mean_rooms+population, data=cleaned_housing.tidy)
k_fold_cv_error = cv.glm(cleaned_housing.tidy , glm_house, K=5)

k_fold_cv_error$delta

glm_cv_rmse = sqrt(k_fold_cv_error$delta)[1]
glm_cv_rmse #off by about $83,000... it is a start

names(glm_house) #what parts of the model are callable?

glm_house$coefficients 

library('randomForest')

?randomForest

names(train)

set.seed(1738)

train_y = train[,'median_house_value']
train_x = train[, names(train) !='median_house_value']

head(train_y)
head(train_x)


#some people like weird r format like this... I find it causes headaches
#rf_model = randomForest(median_house_value~. , data = as.matrix(train), ntree =50, importance = TRUE)
rf_model = randomForest(train_x, y = as.matrix(train_y) , ntree = 50, importance = TRUE)
# KC: this is the only problem, need to convert to matrix before you can feed it to random forest

names(rf_model) #these are all the different things you can call from the model.

rf_model$importance

oob_prediction = predict(rf_model) #leaving out a data source forces OOB predictions

#you may have noticed that this is avaliable using the $mse in the model options.
#but this way we learn stuff!
train_mse = mean(as.numeric((oob_prediction - train_y)^2))
oob_rmse = sqrt(train_mse)
oob_rmse

test_y = test[,'median_house_value']
test_x = test[, names(test) !='median_house_value']


y_pred = predict(rf_model , test_x)
test_mse = mean(((y_pred - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse
