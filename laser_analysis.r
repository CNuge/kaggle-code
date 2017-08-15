setwd('/Users/Cam/Desktop/ds_practice/faa-laser-incident-reports')
getwd()
ls()
options(prompt='R> ')
options(continue = '\t')


laser_dat = read.csv('adjusted_laser_data.csv')

head(laser_dat)


day_breakdown = table(laser_dat$day_of_week)
barplot(day_breakdown)



holiday_breakdown = table(laser_dat$holidays)
holiday_breakdown
barplot(holiday_breakdown)