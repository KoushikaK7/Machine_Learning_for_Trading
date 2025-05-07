from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner
from StrategyLearner import test
from ManualStrategy import ManualStrategy
from ManualStrategy import short_long_calculation
from ManualStrategy import plot_graphs
from ManualStrategy import statistics
from ManualStrategy import benchmark_portval
import datetime as dt
import matplotlib.pyplot as plt
from experiment1 import experiment1
from experiment2 import experiment2
import indicators

def author():
    return 'koush3'

if __name__=="__main__":
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_out = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    benchmark_portvals = benchmark_portval(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    benchmark_portvals_out = benchmark_portval(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    manual_portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    manual_portvals_out = compute_portvals(df_trades_out, start_val=100000, commission=9.95, impact=0.005)
    long_value, short_value = short_long_calculation(symbol=['JPM'], last_action='OUT', df_trades=df_trades)
    long_value_out, short_value_out = short_long_calculation(symbol=['JPM'], last_action='OUT', df_trades=df_trades_out)
    plot_graphs(benchmark_portvals, manual_portvals, short_value, long_value, 'in_sample')
    plot_graphs(benchmark_portvals_out, manual_portvals_out, short_value_out, long_value_out, 'out_sample')
    experiment1()
    experiment2()
    statistics(benchmark_portvals, manual_portvals, verbose=False)
    statistics(benchmark_portvals_out, manual_portvals_out, verbose=False)
    test()


