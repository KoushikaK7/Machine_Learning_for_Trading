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
from util import get_data, plot_data
#import TheoreticallyOptimalStrategy as tos
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def compute_portvals(df_trade_action, start_val=1000000, commission=9.95, impact=0.005):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		  	  		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		  	  		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  	  		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  	  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		  	  		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		  	  		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		  	  		  		  		    	 		 		   		 		  
    # TODO: Your code here

    # Trade Execution Table - Read Orders File Data
    pd.options.mode.chained_assignment = None  # default='warn'
    start_date = df_trade_action.index[0]
    end_date = df_trade_action.index[-1]
    symbol = df_trade_action.columns[0]
    # print(type(symbol))
    #df.loc[df["Order"] == "SELL", 'Shares'] = df.loc[df["Order"] == "SELL", 'Shares']*-1


    # Prices Table - Read Adjusted Prices for Symbols present in Orders

    portvals = pd.DataFrame(get_data([symbol], pd.date_range(start_date, end_date), addSPY=False))
    portvals.fillna(method='ffill', inplace=True)
    portvals.fillna(method='bfill', inplace=True)

    # Final Prices Table - Add Cash Value to Prices Table
    col_length = portvals.shape[0]
    portvals['Cash'] = 0
    df_trade_action['Cash'] = 0

    for index, row in df_trade_action.iterrows():
        if row[symbol] != 0:
            trade_value = row[symbol] * portvals[symbol][index]
            if row[symbol] < 0:
                df_trade_action['Cash'][index] -= trade_value + (trade_value * impact) + commission
            else:
                df_trade_action['Cash'][index] -= trade_value + (trade_value * impact) + commission

    # Holdings Table
    df_trade_action.iloc[0, -1] = df_trade_action.iloc[0, -1] + start_val
    df_trade_action = df_trade_action.cumsum(axis=0)


    #print(df_trade_action)

    # Final Value Table

    df_trade_action[symbol] = df_trade_action[symbol] * portvals[symbol]
    final_value = pd.DataFrame(df_trade_action.sum(axis=1))
    #print(final_value)
    return final_value

def author():
    return "koush3"
#
#
if __name__ == "__main__":
    df = tos.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    #print(compute_portvals(df, start_val=100000, commission=0, impact=0))

