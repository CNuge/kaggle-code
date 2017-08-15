from datetime import date, datetime
import calendar
import pandas as pd
from pandas import Series, DataFrame
import holidays
import ggplot as *

""" Looking at the FAA laser incident reports data, I have the
	following two alternative hypotheses I wish to explore.
	1. There will be a higher number of laser incidents on the weekend 
		(friday night - early sunday morning)
	2. There will be a higher number of laser incidents on holidays. """

# First read in the data

laser_dat = pd.read_csv('all_laser_dat.csv')
laser_dat.head()

#drop the 4 rows that do not have a time associated with them
laser_dat = laser_dat[laser_dat['TIME (UTC)'] != 'UNKN']

#turn the dates to datetimes.
laser_dat['TIME (UTC)']

laser_dat['hour'] = laser_dat.apply(lambda x: x['TIME (UTC)'][:-2], axis=1)
laser_dat['min'] = laser_dat.apply(lambda x: x['TIME (UTC)'][-2:], axis=1)

laser_dat['min'].fillna(0, inplace=True)
laser_dat['hour'].fillna(0, inplace=True)

#account for lack of zeros
min_changed = []
for i in laser_dat['min']:
	if len(i) == 0:
		min_changed.append('00')
	elif len(i) == 1:
		min_changed.append('0'+i)
	else:
		min_changed.append(i)

hr_changed = []
for i in laser_dat['hour']:
	if len(i) == 0:
		hr_changed.append('00')
	elif len(i) == 1:
		hr_changed.append('0'+i)
	else:
		hr_changed.append(i)


laser_dat['min_adj'] = min_changed
laser_dat['hr_adj'] = hr_changed

laser_dat['time'] = laser_dat.apply(lambda x: '%s:%s:%s' % (x['DATE'] , x['hr_adj'],  x['min_adj'] ), axis=1)

laser_dat['date_time'] = laser_dat.apply(lambda x: datetime.strptime(x['time'], '%d-%b-%y:%H:%M'), axis=1)

#drop the making of datetime columns, except for the 'hour' column
laser_dat = laser_dat.drop(['time','hour','min', 'min_adj','TIME (UTC)','DATE'], axis=1)

# add a column with the day of the week

laser_dat['day_of_week'] = laser_dat.apply(lambda x:  calendar.day_name[x['date_time'].weekday()] , axis = 1)


# add a column with holiday/no holidays

us_holidays = holidays.UnitedStates()  # or holidays.US()

holiday_tf = []
for date in laser_dat['date_time']:
	if date in us_holidays:
		holiday_tf.append(True)
	elif date not in us_holidays:
		holiday_tf.append(False)


laser_dat['holidays'] = holiday_tf

laser_dat['holidays'].value_counts()


laser_dat.to_csv('adjusted_laser_data.csv')

