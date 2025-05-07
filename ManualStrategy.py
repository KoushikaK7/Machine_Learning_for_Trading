import datetime as dt
import random
import pandas as pd
from util import get_data, plot_data
import indicators
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

class ManualStrategy:
    def testPolicy(self, symbol, sd, ed, sv):
        df_adj_price = get_data([symbol], pd.date_range(sd, ed))
        df_adj_price = df_adj_price.drop(['SPY'], axis=1)
        df_adj_price.fillna(method='ffill', inplace=True)
        df_adj_price.fillna(method='bfill', inplace=True)
        normalized_df_adj_price = df_adj_price[symbol]/df_adj_price[symbol][0]

        df_trade_holdings = df_adj_price.copy()
        df_trade_holdings[:] = 0
        #print(df_adj_price)

        lookback = 20

        back_days = dt.timedelta(days=2 * lookback)
        prices_lookback_df = get_data([symbol], pd.date_range(sd - back_days, ed))
        prices_lookback_df = prices_lookback_df.drop(['SPY'], axis=1)
        #print(prices_loopback_df)
        bollinger = indicators.get_bolinger_ind(df_adj_price, prices_lookback_df, lookback, start_date=sd, end_date=ed)
        momentum = indicators.get_momentum_ind(df_adj_price, no_days=12, start_date=sd, end_date=ed)

        ema_20_value = indicators.get_ema(df_adj_price, 20)
        normalize_ema_20 = ema_20_value[symbol]/ema_20_value[symbol][0]

        macd_raw = indicators.get_MACD(df_adj_price, 12, 26, 9, sd, ed)
        rsi = indicators.get_RSI(df_adj_price, prices_lookback_df, lookback, symbol, sd, ed)

        trade_position = 0
        trade_dates = df_trade_holdings.index
        last_action = 0

        for i in range(len(trade_dates)):
            last_action += 1
            normalized_df_adj_price_today = normalized_df_adj_price[trade_dates[i]]

            ema_20_value_today = normalize_ema_20.loc[trade_dates[i]]
            if normalized_df_adj_price_today > ema_20_value_today:
                ema_vote_value = 1
            elif normalized_df_adj_price_today < ema_20_value_today:
                ema_vote_value = -1
            else:
                ema_vote_value = 0

            macd_raw_today = macd_raw.loc[trade_dates[i]]['macd']
            macd_signal_today = macd_raw.loc[trade_dates[i]]['macd_signal']
            if macd_raw_today < macd_signal_today:
                macd_vote_value = 2
            elif macd_raw_today > macd_signal_today:
                macd_vote_value = -2
            else:
                macd_vote_value = 0

            # tsi_today = tsi.loc[trade_dates[i]]
            # # if tsi_today > 0.05:
            # if tsi_today[symbol] > 0.1:
            #     tsi_vote = 3
            # # elif tsi_today < -0.05:
            # elif tsi_today[symbol] < 0.1:
            #     tsi_vote = -1
            # else:
            #     tsi_vote = 0

            rsi_today_value = rsi.loc[trade_dates[i]]
            if np.all(rsi_today_value < 30):
                rsi_vote_value = 1
            elif np.all(rsi_today_value > 70):
                rsi_vote_value = -1
            else:
                rsi_vote_value = 0

            bollinger_today = bollinger.loc[trade_dates[i]]
            if bollinger_today['Bolinger %'] < 0:
                bollinger_vote = 2
            elif bollinger_today['Bolinger %'] > 1:
                bollinger_vote = -1
            else:
                bollinger_vote = 0

            momentum_today = momentum.loc[trade_dates[i]]
            if momentum_today['Momentum'] > 0:
                momentum_vote = 1
            elif momentum_today['Momentum'] < 0:
                momentum_vote = -1
            else:
                momentum_vote = 0

            vote_aggregate = macd_vote_value + bollinger_vote + momentum_vote + ema_vote_value +rsi_vote_value
            # print("Ema_Vote: %d", ema_vote_value)
            # print("MACD_Vote: %d", macd_vote_value)
            # print("TSI_Vote: %d", tsi_vote)
            # # print("RSI_Vote: %d", rsi_vote_value)
            # print("Vote Aggregate: %d", vote_aggregate)

            if vote_aggregate >= 2:
                trade_action = 1000 - trade_position
            elif vote_aggregate <= 0:
                trade_action = -1000 - trade_position
            else:
                trade_action = -trade_position

            if last_action >= 5:
                df_trade_holdings.loc[trade_dates[i]] = trade_action
                trade_position += trade_action
                last_action = 0
        #print(df_trade_holdings)

        return df_trade_holdings

    def author(self):
        return "koush3"

def benchmark_portval(sd, ed, sv):
    df_trades = get_data(['JPM'], pd.date_range(sd, ed))
    df_trades = df_trades.drop(['SPY'], axis=1)
    df_trades[:] = 0
    df_trades.ix[df_trades.index[0]] = 1000
    benchmark_portvals = compute_portvals(df_trades, sv, commission=9.95, impact=0.005)
    return benchmark_portvals

def statistics(benchmark_portvals, manual_portvals, verbose=False):
    # Cummulative Returns
    cr_benchmark = benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0] - 1
    cr_manual = manual_portvals.iloc[-1] / manual_portvals.iloc[0] - 1

    # Average Daily Returns
    dr_benchmark = (benchmark_portvals / benchmark_portvals.shift(1)) - 1
    dr_benchmark = dr_benchmark.iloc[1:]
    dr_manual = (manual_portvals / manual_portvals.shift(1)) - 1
    dr_manual = dr_manual.iloc[1:]
    ben_avg_drt = dr_benchmark.mean()
    man_avg_dr = dr_manual.mean()

    # Standard Deviation of Daily Returns
    ben_std_drt = dr_benchmark.std()
    man_std_drt = dr_manual.std()

    if verbose:
        print('\n')

        print("Benchmark Statistics")
        print("Cummulative Return:", end="")
        print('%.6f' % cr_benchmark)
        print("Average Daily Return:", end="")
        print('%.6f' % ben_avg_drt)
        print("Average Standard Deviation:", end="")
        print('%.6f' % ben_std_drt)

        print('\n')

        print("Manual Strategy Statistics")
        print("Cummulative Return:", end="")
        print('%.6f' % cr_manual)
        print("Average Daily Return:", end="")
        print('%.6f' % man_avg_dr)
        print("Average Standard Deviation:", end="")
        print('%.6f' % man_std_drt)


def plot_graphs(benchmark_portvals, manual_portvals, short, long, label):
    # Normalize
    benchmark_portvals = benchmark_portvals / benchmark_portvals.ix[benchmark_portvals.index[0]]
    manual_portvals = manual_portvals / manual_portvals.ix[manual_portvals.index[0]]

    plt.figure(figsize=(14, 8))
    plt.title("Benchmark vs Manual Strategy on " + label)
    plt.xlabel("Date")
    plt.ylabel(" Normalized Portfolio Value")
    plt.xticks(rotation=30)
    #plt.ylim(0, 3)
    plt.grid()
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.plot(manual_portvals, label="Manual_Strategy", color="red")

    for date in short:
        plt.axvline(date, color="black")

    for date in long:
        plt.axvline(date, color="blue")

    plt.legend()
    plt.savefig("images/BenchvsManual_{}.png".format(label), bbox_inches='tight')
    plt.clf()

def short_long_calculation(symbol, last_action, df_trades):
    long_value = []
    short_value = []
    current_value = 0.0
    # print(type(current))
    for date in df_trades.index:
        current_value += df_trades.loc[date].loc[symbol[0]]
        if current_value < 0:
            if last_action == 'OUT' or last_action == 'LONG':
                last_action = 'SHORT'
                short_value.append(date)
        elif current_value > 0:
            if last_action == 'OUT' or last_action == 'SHORT':
                last_action = 'LONG'
                long_value.append(date)
        else:
            last_action = 'OUT'
    return long_value, short_value

if __name__=="__main__":
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv=100000)
    df_trades_out = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000)
    benchmark_portvals = benchmark_portval(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv=100000)
    benchmark_portvals_out = benchmark_portval(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000)
    manual_portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    manual_portvals_out = compute_portvals(df_trades_out, start_val=100000, commission=9.95, impact=0.005)
    long_value, short_value = short_long_calculation(symbol=['JPM'], last_action='OUT', df_trades=df_trades)
    long_value_out, short_value_out = short_long_calculation(symbol=['JPM'], last_action='OUT', df_trades=df_trades_out)
    #plot_graphs(benchmark_portvals, manual_portvals, short_value, long_value, 'in_sample')
    #plot_graphs(benchmark_portvals_out, manual_portvals_out, short_value_out, long_value_out, 'out_sample')
    statistics(benchmark_portvals, manual_portvals, verbose=True)
    statistics(benchmark_portvals_out,manual_portvals_out, verbose=True)






