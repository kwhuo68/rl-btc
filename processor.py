import numpy as np 
import pandas as pd
import math
import requests, json, time 
from datetime import datetime

class Processor:
	def __init__(self, history_length):
		self.history_length = history_length

	def fetchHistoricalDataForTicker(self, fsym, tsym, lim):
		df_cols = ['time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
		curr_ts = str(int(time.time()))
		limit = str(lim)
		histURL = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + fsym + '&tsym=' + tsym + '&limit=' + limit + '&toTs=' + curr_ts + '&aggregate=1' + '&e=Coinbase' #CCCAGG for aggregated
		resp = requests.get(histURL)
		resp_json = json.loads(resp.content.decode('utf-8'))	
		df = pd.DataFrame(columns = df_cols)
		for i in range(0, lim):
			data = []
			for count, val in enumerate(df_cols):
				entry = resp_json['Data'][i][val]
				data.append(entry)
			row = pd.Series(data, df_cols)
			df = df.append(row, ignore_index = True)
		if(df.empty):
			return
		df = df.rename(index=str, columns={"time": "ts"})
		df.index = pd.to_datetime(df.ts, unit = 's')
		df = df.drop('ts', axis = 1)
		matrix_df = df.as_matrix()
		return matrix_df

	def fetchData(self):
		data = self.fetchHistoricalDataForTicker('ETH', 'USD', 2000)
		#Arbitrary 1500/500 split
		train_data = data[:1500, :]
		test_data = data[1501:, :]
		return {'train': train_data, 'test': test_data}




