from datetime import datetime
import pandas as pd
from pandas import DataFrame
import pandas_datareader.data as web


if __name__ == '__main__':

	""" set the stock data for the past year """
	now_time = datetime.now()

	start_time = datetime(now_time.year - 5, now_time.month , now_time.day)

	s_and_p_companies = pd.read_csv('s_and_p.csv')

	s_and_p = list(s_and_p_companies['Ticker'].values)

	bad_names =[]
	#undone =[]
	for i, stock in enumerate(s_and_p):
		try:
			print(stock)
			stock_df = web.DataReader(stock,'iex', start_time, now_time)
			stock_df['Name'] = stock
			output_name = stock + '_data.csv'
			stock_df.to_csv(output_name)
			if i == 0:
				big_df = stock_df
			else:
				big_df = big_df.append(stock_df)
		except:
			bad_names.append(stock)
			print('bad: %s' % (stock))
	
	print(bad_names)
	big_df.to_csv('all_s_and_p_data.csv')

	""" Save failed queries to a text file to retry """
	if len(bad_names) > 0:
		with open('failed_queries.txt','w') as outfile:
			for name in bad_names:
				outfile.write(name+'\n')

