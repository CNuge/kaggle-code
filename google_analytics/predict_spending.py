import pandas as pd
import numpy as np


all_train = pd.read_csv('./data/train_cleaned.csv')

all_train.head() #there was a json one I missed, double back time :(


#need to go through and clean the columns
all_train.describe()