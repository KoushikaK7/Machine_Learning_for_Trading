from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
from ManualStrategy import benchmark_portval
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'koush3'

def experiment1():
    symbol = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000

    ml = ManualStrategy()
    ml_trade_holdings = ml.testPolicy(symbol, sd=sd, ed=ed, sv=sv)
    ml_portval = compute_portvals(ml_trade_holdings, start_val=sv, commission=9.95, impact=0.005)
    bench_portval = benchmark_portval(sd, ed, sv)

    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv = sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_portval = compute_portvals(learner_trades, start_val = sv, commission=9.95, impact=0.005)
    plot_graph(ml_portval, learner_portval, bench_portval, label='in_sample')

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    ml_trade_holdings = ml.testPolicy(symbol, sd=sd, ed=ed, sv=sv)
    ml_portval = compute_portvals(ml_trade_holdings, start_val=sv, commission=9.95, impact=0.005)
    bench_portval = benchmark_portval(sd, ed, sv)

    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_portval = compute_portvals(learner_trades, start_val=sv, commission=9.95, impact=0.005)
    plot_graph(ml_portval, learner_portval, bench_portval, label='out_sample')

def plot_graph(manual_portvals, strategy_portvals, bench_portval, label):

    # normalize
    benchmark_portvals = bench_portval / bench_portval.ix[bench_portval.index[0]]
    manual_portvals = manual_portvals / manual_portvals.ix[manual_portvals.index[0]]
    strategy_portvals = strategy_portvals / strategy_portvals.ix[strategy_portvals.index[0]]

    plt.figure(figsize=(14,8))
    plt.title("Experiment 1: Benchmark vs Manual Strategy vs Q-Learning Strategy "+label)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.xticks(rotation=30)
    plt.grid()
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.plot(manual_portvals, label="Manual Strategy", color="red")
    plt.plot(strategy_portvals, label="Q-Learning Strategy Learner", color="green")

    plt.legend()
    plt.savefig("images/Experiment1_{}.png".format(label), bbox_inches='tight')
    # plt.show()
    plt.clf()

if __name__=="__main__":
    experiment1()
