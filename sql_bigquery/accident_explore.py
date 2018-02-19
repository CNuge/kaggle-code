import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



a_v_2016 = pd.read_csv('vehicle_and_accident_data_2016.csv',index_col=0)

a_v_2016.head()
a_v_2016.tail()

a_v_2016.columns


#first look at the lat and longitude data

a_v_2016.plot(kind='scatter', x='longitude', y='latitude')
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()


#there are some erroneous values in the data,
# this will remove them

a_v_2016 = a_v_2016.drop(a_v_2016[ a_v_2016['longitude'] > 0].index)

#plotting this, we still ate a little squished on account of alaska and hawaii,
# I'm going to remove these in order to focus on the continentual united states
a_v_2016.plot(kind='scatter', x='longitude', y='latitude',
			 alpha=0.4,figsize=(10,7), c='black')
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()

a_v_2016 = a_v_2016.drop(a_v_2016[ a_v_2016['longitude'] < -130].index)


#distribution of fatal car accidents in the continential united states
#show in a mercator projection
m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,\
            llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
#m.drawstates()
# draw parallels and meridians.
parallels = np.arange(-90., 91., 5.)
# Label the meridians and parallels
m.drawparallels(parallels, labels=[False,True,True,False])
# Draw Meridians and Labels
meridians = np.arange(-180., 181., 10.)
m.drawmeridians(meridians, labels=[True, False, False, True])
m.drawmapboundary(fill_color = 'white')
plt.title('Fatal car accidents in the continential United States, 2016')
x,y = m(a_v_2016['longitude'].values, a_v_2016['latitude'].values) #transform to projection
m.plot(x,y, 'bo', markersize = 0.5)
plt.show()



#isolate just the drunk driving incidents

drunk_driving = a_v_2016[a_v_2016['driver_drinking'] == 'Drinking']


m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,\
            llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
#m.drawstates()
# draw parallels and meridians.
parallels = np.arange(-90., 91., 5.)
# Label the meridians and parallels
m.drawparallels(parallels, labels=[False,True,True,False])
# Draw Meridians and Labels
meridians = np.arange(-180., 181., 10.)
m.drawmeridians(meridians, labels=[True, False, False, True])
m.drawmapboundary(fill_color = 'white')
plt.title('Fatal car accidents Involving a drunk driver, 2016')
x,y = m(drunk_driving['longitude'].values, drunk_driving['latitude'].values) #transform to projection
m.plot(x,y, 'bo', markersize = 0.5)
plt.show()


# Concentration appears to mirror the location of major cities in the United states


#what kind of cars are involved in the accidents?

a_v_2016.vehicle_make_name.unique()

"""
The category breakdown there is a little ridiculous

Note this entry:
  'Other Domestic\nAvanti\nChecker\nDeSoto\nExcalibur\nHudson\nPackard\nPanoz\nSaleen\nStudebaker\nStutz\nTesla (Since 2014)'
  Hudson ceased to exist in 1954, Packard died in 1956 and studebaker went defunct in 1967... yet these are grouped with Teslas?

"""

a_v_2016.vehicle_model_year

a_v_2016.body_type_name.unique()
