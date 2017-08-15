setwd('/Users/Cam/Desktop/ds_practice/faa-laser-incident-reports')
getwd()
ls()
options(prompt='R> ')
options(continue = '\t')


laser_dat = read.csv('adjusted_laser_data.csv')

head(laser_dat)

?table

laser_dat$day_of_week = factor(laser_dat$day_of_week,c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
day_breakdown = table(laser_dat$day_of_week)
day_breakdown
barplot(day_breakdown,ylim=c(0,3000))

results = chisq.test(day_breakdown)
summary(results)
results
results$observed
results$expected

holiday_breakdown = table(laser_dat$holidays)
holiday_breakdown
barplot(holiday_breakdown)