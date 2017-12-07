from datetime import datetime
from pandas import DataFrame
import pandas_datareader as web

def get_american_stock_dat(stock_of_interest, start_time, now_time):
	""" get a dataframe for an american stock of interest """
	f_dat = web.DataReader(stock_of_interest, 'yahoo', start_time, now_time)
	return f_dat


if __name__ == '__main__':


	""" set the stock data for the past year """
	now_time = datetime.now()

	start_time = datetime(now_time.year - 5, now_time.month , now_time.day)

	s_and_p = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES',
	'AET','AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
	'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP','AXP',
	'AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV','ANSS',
	'ANTM','AON','AOS','APA','AIV','AAPL','AMAT','ADM','ARNC','AJG','AIZ','T',
	'ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BK','BCR','BAX','BBT',
	'BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX','BHF','BMY',
	'AVGO','BF.B','CHRW','CA','COG','CPB','COF','CAH','CBOE','KMX','CCL','CAT',
	'CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW','CHTR','CHK','CVX',
	'CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG','CTXS','CLX',
	'CME','CMS','COH','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP','ED',
	'STZ','COO','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI','DHR',
	'DRI','DVA','DE','DLPH','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
	'DG','DLTR','D','DOV','DOW','DPS','DTE','DRE','DD','DUK','DXC','ETFC','EMN',
	'ETN','EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX',
	'EQIX','EQR','ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM',
	'FFIV','FB','FAST','FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR',
	'FMC','FL','F','FTV','FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP',
	'GIS','GM','GPC','GILD','GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG',
	'HAS','HCA','HCP','HP','HSIC','HSY','HES','HPE','HLT','HOLX','HD','HON',
	'HRL','HST','HPQ','HUM','HBAN','IDXX','INFO','ITW','ILMN','IR','INTC','ICE',
	'IBM','INCY','IP','IPG','IFF','INTU','ISRG','IVZ','IRM','JEC','JBHT','SJM',
	'JNJ','JCI','JPM','JNPR','KSU','K','KEY','KMB','KIM','KMI','KLAC','KSS','KHC',
	'KR','LB','LLL','LH','LRCX','LEG','LEN','LVLT','LUK','LLY','LNC','LKQ','NYSE:LMT',
	'L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM','MAS','MA','MAT',
	'MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU','MSFT','MAA',
	'MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ','NOV','NAVI',
	'NTAP','NFLX','NYSE:NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI','NYSE:NBL','JWN',
	'NSC','NTRS','NOC','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE','ORCL','PCAR',
	'PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE','PCG','PM',
	'PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR','PLD','PRU',
	'PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RRC','RJF','RTN','O','RHT','REG',
	'REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SCG','SLB','SNI',
	'STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV','SPGI','SWK','SPLS',
	'SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TGT','TEL','FTI',
	'TXN','TXT','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRV','TRIP','FOXA',
	'FOX','TSN','UDR','ULTA','USB','UA','UAA','UNP','UAL','UNH','UPS','URI','UTX','UHS',
	'UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO','VMC','WMT',
	'WBA','DIS','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WFM','WMB',
	'WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']
	
	undone =['ALLE', 'AMGN', 'AON', 'AOS', 'AVB', 'BAC', 'BRK.B', 'BA', 'BF.B', 'CI', 'COH', 'CMCSA', 'STZ', 'DVN', 'DOW', 'DD', 'EFX', 'FDX', 'LB', 'NYSE:LMT', 'MAT', 'NYSE:NWL', 'NYSE:NBL', 'OMC', 'RHT', 'SPLS', 'WFM']
	bad_names =[]
	for i, stock in enumerate(undone):
		try:
			print(stock)
			stock_df = get_american_stock_dat(stock, start_time, now_time)
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

