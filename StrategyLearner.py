""""""
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  

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
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util  		   	  			  	 		  		  		    	 		 		   		 		  
import random  		   
import QLearner as ql	  	
import indicators	
from marketsimcode import compute_portvals  
import matplotlib.pyplot as plt		  		    	 		 		   		 		  

class StrategyLearner(object):  		   	  			  	    	  			  	 		  		  		   	 		   		 		  
    # StrategyLearner constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0, commission=0.0):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose   	  			  	 		  		  		    	 		 		   		 		  
        self.impact = impact  
        self.commission = commission
        random.seed(903847815)	 

        self.learner = ql.QLearner(num_states=96,\
            num_actions = 3, \
            alpha = 0.2, \
            gamma = 0.9, \
            rar = 0.9, \
            radr = 0.99, \
            dyna = 200, \
            verbose=False)  	

   	  			  	 		  		  		    	 		 		   		 		  
    def add_evidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000):  	
        
        # getting indicator data
        ema_20, macd, rsi, bollinger, momentum = get_discretized_indicators(sd, ed, symbol)

        # getting Market price data
        df_adj_price, df_trades = get_prices_df(sd, ed, symbol)
        dates = df_adj_price.index

        # start learning
        current_position = 0
        current_cash = sv
        prev_position = 0
        prev_cash = sv

        for i in range(1, len(dates)):
            today = dates[i]
            yesterday = dates[i - 1]
            s_prime = compute_current_state(current_position, ema_20.loc[today], 
                macd.loc[today], rsi.loc[today], bollinger.loc[today], momentum.loc[today])

            r = current_position * df_adj_price.loc[today].loc[symbol] + current_cash - prev_position * df_adj_price.loc[yesterday].loc[symbol] - prev_cash

            # {0: SHORT, 1: CASH, 2: LONG}
            next_action = self.learner.query(s_prime, r)
            if next_action == 0:
                trade = -1000 - current_position
            elif next_action == 1:
                trade = -current_position
            else:
                trade = 1000 - current_position
            
            if self.verbose:
                print(today)
                print("previous day position: {}".format(prev_position))
                print("previous day cash: {}".format(prev_cash))
                print("today position: {}".format(current_position))
                print("today cash: {}".format(current_cash))
                print("Price today: " + str(df_adj_price.loc[today].loc[symbol]))
                print("Last trade reward: " + str(r))
                print("Trade: {}".format(trade))
                print()

            prev_position = current_position
            current_position += trade
            df_trades.loc[today].loc[symbol] = trade

            if trade > 0:
                impact = self.impact
            else:
                impact = -self.impact
            
            prev_cash = current_cash
            current_cash += -df_adj_price.loc[today].loc[symbol] * (1 + impact) * trade
        
        if self.verbose:
            print("[{} in benchmark]".format(symbol))
            print(get_benchmark(sd, ed, sv, self.commission, self.impact).tail())
            print()
            print("[{} in training performance]".format(symbol))
            print(compute_portvals(df_trades, start_val = sv, commission = self.commission, impact = self.impact).tail())
            print()

 	 		  		  		    	 		 		   		 		  
    # Method to use previous learning and test new set of data  		   	  			  	 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "JPM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 10000, \
        label = 'in_sample'):  		   	  			  	 		  		  		    	 		 		   		 		  

        # getting indicator data
        ema_20, macd, rsi, bollinger, momentum = get_discretized_indicators(sd, ed, symbol)

        # getting Market price data
        df_adj_price, df_trades = get_prices_df(sd, ed, symbol)
        dates = df_adj_price.index
	  	 		  		  		    	 		 		   		 		  			  	 		  		  		    	 		 		   		 		  
        current_position = 0

        # train the learner
        for i in range(1, len(dates)):
            today = dates[i]
            yesterday = dates[i - 1]

            s_prime = compute_current_state(current_position, ema_20.loc[today], 
                macd.loc[today], rsi.loc[today], bollinger.loc[today], momentum.loc[today])

            # {0: SHORT, 1: CASH, 2: LONG}
            next_action = self.learner.querysetstate(s_prime)
            if next_action == 0:
                trade = -1000 - current_position
            elif next_action == 1:
                trade = -current_position
            else:
                trade = 1000 - current_position

            current_position += trade
            df_trades.loc[today].loc[symbol] = trade
     
        #del df_trades['Cash']
        return df_trades

    def author(self):
        return 'koush3'

def get_prices_df(sd, ed, symbol):
    df_adj_price = util.get_data([symbol], pd.date_range(sd, ed), addSPY=True)
    del df_adj_price['SPY']
    df_adj_price.fillna(method='ffill', inplace=True)
    df_adj_price.fillna(method='bfill', inplace=True)
    #normalized_df_adj_price = df_adj_price[symbol]/df_adj_price[symbol][0]
    df_trade_holdings = df_adj_price.copy()
    df_trade_holdings[:] = 0		   	  			  	 
    return df_adj_price, df_trade_holdings

def get_discretized_indicators(sd, ed, symbol):
    prices, _ = get_prices_df(sd, ed, symbol)

    ema_20 = indicators.get_ema(prices, 20)

    macd_raw = indicators.get_MACD(prices, 12, 26, 9, sd, ed)
    macd = macd_raw.copy()
    del macd['macd_signal']
    bollinger1 = macd.copy()
    momentum1 = macd.copy()
    bollinger1.rename(columns={'macd': 'bollinger'}, inplace=True)
    momentum1.rename(columns={'macd': 'momentum'}, inplace=True)

    lookback = 20
    back_days = dt.timedelta(days=2 * lookback)
    prices_lookback_df = util.get_data([symbol], pd.date_range(sd - back_days, ed), addSPY=True)
    del prices_lookback_df['SPY']
    rsi = indicators.get_RSI(prices, prices_lookback_df, lookback, symbol, sd, ed)
    bollinger = indicators.get_bolinger_ind(prices, prices_lookback_df, lookback, start_date=sd, end_date=ed)
    momentum = indicators.get_momentum_ind(prices, no_days=12, start_date=sd, end_date=ed)

    trade_position = 0
    trade_dates = prices.index

    for i in range(len(trade_dates)):
        price_today = prices.loc[trade_dates[i]][0]
        # EMA20 has 3 States: Price <= EMA: 0, Price > EMA: 1
        ema20_today = ema_20.loc[trade_dates[i]][0]
        if (price_today > ema20_today):
            ema_20[symbol][trade_dates[i]] = 1
        elif (price_today < ema20_today):
            ema_20[symbol][trade_dates[i]] = -1
        else:
            ema_20[symbol][trade_dates[i]] = 0

        # MACD has 3 States: MACD < Signal: 1, MACD = Signal: 0, MACD > Signal: -1
        macd_raw_today = macd_raw.loc[trade_dates[i]]['macd']
        macd_signal_today = macd_raw.loc[trade_dates[i]]['macd_signal']
        if (macd_raw_today < macd_signal_today):
            macd['macd'][trade_dates[i]] = 1
        elif (macd_raw_today > macd_signal_today):
            macd['macd'][trade_dates[i]] = -1
        else:
            macd['macd'][trade_dates[i]] = 0

        # RSI has 3 States: RSI < 30: 1, RSI > 70: -1, else: 0
        rsi_today = rsi.loc[trade_dates[i]][0]
        if (rsi_today < 30):
            rsi['RSI'][trade_dates[i]] = 1
        elif (rsi_today > 70):
            rsi['RSI'][trade_dates[i]] = -1
        else:
            rsi['RSI'][trade_dates[i]] = 0

        # Bollinger has 3 States: Bolinger % < 0: 1, Bolinger % > 1: -1, else: 0
        bollinger_today = bollinger.loc[trade_dates[i]]
        if bollinger_today['Bolinger %'] < 0:
            bollinger1['bollinger'][trade_dates[i]] = 1
        elif bollinger_today['Bolinger %'] > 1:
            bollinger1['bollinger'][trade_dates[i]] = -1
        else:
            bollinger1['bollinger'][trade_dates[i]] = 0

        # momentum has 3 states: Momentum > 0: 1, Momentum < 0: -1, else: 0
        momentum_today = momentum.loc[trade_dates[i]]
        if momentum_today['Momentum'] > 0:
            momentum1['momentum'][trade_dates[i]] = 1
        elif momentum_today['Momentum'] < 0:
            momentum1['momentum'][trade_dates[i]] = -1
        else:
            momentum1['momentum'][trade_dates[i]] = 0


    return ema_20, macd, rsi, bollinger1, momentum1

def compute_current_state(position, ema_20, macd, rsi, bollinger, momentum):
    # 96 states in total, each permutation of indicators + position return between 0 and 95
    # position: -1000, ema_20: 0, macd: 0, rsi:0, bollinger:0, momentum:0 => 0
    # position: 1000, ema_20: 1, macd: 1, rsi:1, bollinger:1, momentum:1 => 95
    state = 0
    if position == 0:
        state += 32
    elif position == 1000:
        state += 64
    state += ema_20.iloc[-1] * 9 + macd.iloc[-1] * 2 + rsi.iloc[-1] * 2 + bollinger.iloc[-1] * 3 + momentum.iloc[-1] * 12
    return int(state)

def get_benchmark(sd, ed, sv, commission, impact):
    df_trades = util.get_data(['JPM'], pd.date_range(sd, ed), addSPY=True)
    del df_trades['SPY']
    df_trades[:] = 0
    df_trades.loc[df_trades.index[0]] = 1000
    portvals = compute_portvals(df_trades, sv, commission = commission, impact = impact)
    return portvals

def plot_graphs(df_trades, sd, ed, sv, commission, impact, label, symbol, verbose):
    benchmark_portvals = get_benchmark(sd, ed, sv, commission, impact)
    learner_portvals = compute_portvals(df_trades, start_val = sv, commission = commission, impact = impact)
    long = []
    short = []
    current = 0.0
    last_action = 'OUT'
    for date in df_trades.index:
        current += df_trades.loc[date][symbol]
        if current < 0:
            if last_action == 'OUT' or last_action == 'LONG':
                last_action = 'SHORT'
                short.append(date)
        elif current > 0:
            if last_action == 'OUT' or last_action == 'SHORT':
                last_action = 'LONG'
                long.append(date)
        else:
            last_action = 'OUT'
    # Normalize the portvals
    benchmark_portvals = benchmark_portvals / benchmark_portvals[0][1]
    learner_portvals = learner_portvals / learner_portvals[0][1]

    plt.figure(figsize=(14, 8))
    plt.title("Stragety learner on " + label)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.xticks(rotation=30)
    plt.grid()
    plt.plot(benchmark_portvals, label="benchmark", color="green")
    plt.plot(learner_portvals, label="learner", color="red")

    for date in short:
        plt.axvline(date, color="black")

    for date in long:
        plt.axvline(date, color="blue")
    plt.legend()
    plt.savefig("images/StrategyLearner_{}.png".format(label), bbox_inches='tight')
    #plt.show()
    plt.clf()
    if verbose:
       # if True:
           print("[{} test benchmark]".format(symbol))
           print(benchmark_portvals.tail())
           print()
           print("[{} testing performance]".format(symbol))
           print(learner_portvals.tail())
           print() 

def test():
    symbol = "JPM"
    impact = 0.005
    commission = 9.95
    verbose = False
    sd_in = dt.datetime(2008,1,1)
    ed_in = dt.datetime(2009,12,31)
    sd_out = dt.datetime(2010,1,1)
    ed_out = dt.datetime(2011,12,31)
    sv = 100000
    learner = StrategyLearner(verbose = verbose, impact=impact, commission=commission) # constructor
    learner.add_evidence(symbol = symbol, sd = sd_in, ed = ed_in, sv = sv)
    df_trades_in = learner.testPolicy(symbol = symbol, sd = sd_in, ed = ed_in, sv = sv, label = 'in_sample')
    plot_graphs(df_trades_in, sd = sd_in, ed = ed_in, sv =sv, commission = commission, impact = impact, label = 'in_sample', symbol = symbol, verbose = verbose)
    df_trades_out = learner.testPolicy(symbol = symbol, sd = sd_out, ed = ed_out, sv = sv, label = 'out_sample')
    plot_graphs(df_trades_out, sd = sd_out, ed = ed_out, sv = sv, commission = commission, impact = impact, label = 'out_sample', symbol = symbol, verbose = verbose)


if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    # print("One does not simply think up a strategy")  	
    test()
