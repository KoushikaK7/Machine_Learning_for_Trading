""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Koushika Kesavan (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: koush3 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903847815 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""
import datetime as dt  		  	   		  	  		  		  		    	 		 		   		 		  
import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt		  	   		  	  		  		  		    	 		 		   		 		  
from util import get_data, plot_data

def author():
	return 'koush3'	 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def get_bolinger_ind(prices_df, prices_loopback_df, lookback, start_date, end_date):
	# Calculate SMA
	sma_df = prices_loopback_df.rolling(lookback).mean()
	# Calculate Bolinger Bands
	std = prices_loopback_df.rolling(lookback).std()
	upper_df = sma_df + (2*std)
	lower_df = sma_df - (2*std)
	Bolinger_percentage_df = (prices_loopback_df-lower_df)/(upper_df-lower_df)

	sma_df.rename(columns={sma_df.columns[0] : 'sma'}, inplace=True)
	upper_df.rename(columns={upper_df.columns[0] : 'upper_band'}, inplace=True)
	lower_df.rename(columns={lower_df.columns[0] : 'lower_band'}, inplace=True)
	Bolinger_percentage_df.rename(columns={Bolinger_percentage_df.columns[0] : 'Bolinger %'}, inplace=True)

	return Bolinger_percentage_df
	
	#sma_df['mean'] = min_max_scaling(sma_df['mean'])
	#upper_df['upper_band'] = min_max_scaling(upper_df['upper_band'])
	#lower_df['lower_band'] = min_max_scaling(lower_df['lower_band'])
	#normalized_price_df = get_normalized_price(prices_df)
	# ax = prices_df.reset_index().plot(x='index', y='JPM')
	# sma_df.reset_index().plot(ax=ax, x='index', y='sma')
	# upper_df.reset_index().plot(ax=ax, x='index', y='upper_band')
	# lower_df.reset_index().plot(ax=ax, x='index', y='lower_band')
	# plt.xlim(start_date, end_date)
	# plt.title("Bollinger Bands Indicator for JPM")
	# plt.xlabel("Date")
	# plt.ylabel("Price")
	# plt.grid()
	# plt.savefig("Bollinger_Indicator.png")
	# plt.clf()

	#ax = prices_df.reset_index().plot(x='index', y='JPM')
	# Bolinger_percentage_df.reset_index().plot(x='index', y='Bolinger %')
	# plt.xlim(start_date, end_date)
	# plt.title("Bollinger % for JPM")
	# plt.axhline(y=0.0, color='r', linestyle='-')
	# plt.axhline(y=1.0, color='r', linestyle='-')
	# plt.xlabel("Date")
	# plt.ylabel("Bollinger Percentage")
	# plt.grid()
	# plt.savefig("Bollinger_Percentage_Indicator.png")
	# plt.clf()
	# #plt.show()
	# return((sma_df,upper_df,lower_df))

def get_RSI(prices_df, prices_loopback_df, lookback, symbol, start_date, end_date):
	rsi_df = prices_loopback_df.mask(prices_loopback_df>0, 0)
	for day in range(prices_loopback_df.shape[0]):
		up_gain = 0
		down_loss = 0
		for prev_day in range (lookback):
			delta = prices_loopback_df[symbol][day-prev_day] - prices_loopback_df[symbol][day-prev_day-1]
			if delta>0:
				up_gain+=delta
			else:
				down_loss+=-1*delta
		if down_loss==0:
			rs = 0
		else:
			rs = (up_gain/lookback)/(down_loss/lookback)
		rsi_df[symbol][day] = 100 - (100/(1+rs))
	
	
	rsi_df.rename(columns={rsi_df.columns[0] : 'RSI'}, inplace=True)

	# ax = prices_df.reset_index().plot(x='index', y='JPM')
	# rsi_df.reset_index().plot(ax=ax, x='index', y='RSI')
	# plt.xlim(start_date, end_date)
	# plt.title("RSI Indicator for JPM")
	# plt.xlabel("Date")
	# plt.ylabel("RSI Value")
	# plt.grid()
	# plt.axhline(y=30.0, color='r', linestyle='-')
	# plt.axhline(y=70.0, color='r', linestyle='-')
	# plt.savefig("RSI_Indicator.png")
	# plt.clf()
	#plt.show()

	return(rsi_df)
	   		  	  		  		  		       	

def get_ema(prices_df, no_days):
	price_df = prices_df.copy()
	ema_df = price_df.ewm(com=((no_days-1)/2)).mean()

	return(ema_df)	  
  		

def get_MACD(prices_df, ema1_days, ema2_days, macd_signal_days, start_date, end_date):
	ema1_df = get_ema(prices_df, ema1_days)
	ema2_df = get_ema(prices_df, ema2_days)
    
	macd_df = ema1_df-ema2_df
	macd_signal_df = get_ema(macd_df, macd_signal_days)
	macd_df['macd_signal'] = macd_signal_df[:]

	#macd_signal_df.rename(columns={macd_signal_df.columns[0] : 'macd_signal'}, inplace=True)
	macd_df.rename(columns={macd_df.columns[0] : 'macd'}, inplace=True)
	#
	# #ax = prices_df.reset_index().plot(x='index', y='JPM')
	# ax = macd_signal_df.reset_index().plot(x='index', y='macd_signal')
	# macd_df.reset_index().plot(ax=ax, x='index', y='macd')
	# plt.xlim(start_date, end_date)
	# plt.title("MACD Indicator for JPM")
	# plt.xlabel("Date")
	# plt.ylabel("Price Value")
	# plt.grid()
	# plt.savefig("MACD_Indicator.png")
	# plt.clf()

	return(macd_df)

# def get_stoc_osc(prices_df, prices_loopback_df, k_window, d_window):
# 	high = prices_loopback_df.rolling(window=k_window).max()
# 	low = prices_loopback_df.rolling(window=k_window).min()
# 	K_df = 100*(prices_loopback_df-low)/(high-low)
# 	D_df = K_df.rolling(window=d_window).mean()
#
# 	K_df.rename(columns={K_df.columns[0] : 'K'}, inplace=True)
# 	D_df.rename(columns={D_df.columns[0] : 'D'}, inplace=True)
#
# 	# ax = prices_df.reset_index().plot(x='index', y='JPM')
# 	# K_df.reset_index().plot(ax=ax, x='index', y='K')
# 	# D_df.reset_index().plot(ax=ax, x='index', y='D')
# 	# plt.xlim(start_date, end_date)
# 	# plt.ylim(-20, 120)
# 	# plt.title("RSI Indicator for JPM")
# 	# plt.xlabel("Date")
# 	# plt.ylabel("RSI Value")
# 	# plt.grid()
# 	# plt.show()
# 	return(K_df,D_df)


def get_momentum_ind(prices_df, no_days, start_date, end_date):
	momentum_df = prices_df.copy()
	momentum_df[no_days:] = (momentum_df[no_days:] / momentum_df[:-no_days].values)-1
	momentum_df[0:no_days] = np.nan

	momentum_df.rename(columns={momentum_df.columns[0] : 'Momentum'}, inplace=True)
	momentum_df['Momentum'] = momentum_df['Momentum']*100
	# momentum_df.reset_index().plot(x='index', y='Momentum')
	# plt.xlim(start_date, end_date)
	# plt.title("ROC-12 Indicator for JPM")
	# plt.xlabel("Date")
	# plt.ylabel("Percentage change in Price")
	# plt.axhline(y=0.0, color='r', linestyle='-')
	# plt.grid()
	# plt.savefig("Momentum_Indicator.png")
	# plt.clf()

	return(momentum_df)

# def tsi(sd, ed, symbol, plot = False):
#
#     # look up history to calculate the ema for the 24 days
#     # since the max ema windows size is 20, we can say 50 is safe
#     delta = dt.timedelta(70)
#     extedned_sd = sd - delta
#
#     df_price = get_data([symbol], pd.date_range(extedned_sd, ed))
#     df_price = df_price[[symbol]]
#     df_price = df_price.ffill().bfill()
#
#     # calculate, smoothing and double smoothing price change
#     diff = df_price - df_price.shift(1)
#     ema_25 = diff.ewm(span=25, adjust=False).mean()
#     ema_13 = ema_25.ewm(span=13, adjust=False).mean()
#
#     # calculate, smoothing and double smoothing absolute price change
#     abs_diff = abs(diff)
#     abs_ema_25 = abs_diff.ewm(span=25, adjust=False).mean()
#     abs_ema_13 = abs_ema_25.ewm(span=13, adjust=False).mean()
#
#     df_tsi = ema_13 / abs_ema_13
#
#     # remove history price
#     df_tsi = df_tsi.truncate(before=sd)
#
#     return df_tsi


def get_normalized_price(price_df):
	normalized_price_df = price_df['JPM'] / price_df['JPM'][0]
	return normalized_price_df  		  		  		    	 		 		   		 		  
  		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":
	start_date = dt.datetime(2008,1,1)
	end_date = dt.datetime(2009,12,31)
	lookback = 20
	symbol = 'JPM'
	# Get the price Table
	prices_df = get_data([symbol], pd.date_range(start_date, end_date), addSPY=True)
	del prices_df['SPY'] # Remove SPY column
	back_days = dt.timedelta(days = 2*lookback)
	prices_loopback_df = get_data([symbol], pd.date_range(start_date-back_days, end_date), addSPY=True)
	del prices_loopback_df['SPY'] # Remove SPY column
	get_bolinger_ind(prices_df, prices_loopback_df, lookback)
	get_RSI(prices_df, prices_loopback_df, lookback, symbol)
	ema_10 = get_ema(prices_df, 10)
	ema_50 = get_ema(prices_df, 50)
	#ema_diff = ema_24 - ema_12
	#ema_diff.rename(columns={ema_diff.columns[0] : 'emadiff'}, inplace=True)
	#ema_diff['emadiff'] = min_max_scaling(ema_diff['emadiff'])
	ema_10.rename(columns={ema_10.columns[0] : 'ema_10'}, inplace=True)
	ema_50.rename(columns={ema_50.columns[0] : 'ema_50'}, inplace=True)
	ax = prices_df.reset_index().plot(x='index', y='JPM')
	ema_10.reset_index().plot(ax=ax, x='index', y='ema_10')
	ema_50.reset_index().plot(ax=ax, x='index', y='ema_50')
	# plt.xlim(start_date, end_date)
	# plt.title("EMA Indicator for JPM")
	# plt.xlabel("Date")
	# plt.ylabel("Price Value")
	# plt.grid()
	get_MACD(prices_df, 12, 26, 9)
	#get_stoc_osc(prices_df, prices_loopback_df, 14, 3)
	get_momentum_ind(prices_loopback_df, 12)


