import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



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




"""

housing_plot.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
				s=housing_plot['population']/100, label='population', figsize=(10,7),
				c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)


plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend() 
plt.show()

"""