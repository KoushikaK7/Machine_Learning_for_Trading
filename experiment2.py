import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util  		   	  			  	 		  		  		    	 		 		   		 		  
import random  		   
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

def experiment2():
    symbol = "JPM"
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    label = 'in_sample'

    strategylearner1 = StrategyLearner(verbose = False, impact=0.00)
    strategylearner1.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
    df_learner1 = strategylearner1.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, label = label)
    learner1_portvals = compute_portvals(df_learner1, start_val = sv, commission=0, impact=0.00)

    strategylearner2 = StrategyLearner(verbose = False, impact=0.025)
    strategylearner2.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
    df_learner2 = strategylearner2.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, label = label)
    learner2_portvals = compute_portvals(df_learner2, start_val = sv, commission=0, impact=0.005)

    strategylearner3 = StrategyLearner(verbose = False, impact=0.05)
    strategylearner3.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
    df_learner3 = strategylearner3.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv, label = label)
    learner3_portvals = compute_portvals(df_learner3, start_val = sv, commission=0, impact=0.01)


    # Plot the impact effect compare graphs
    plot_impact_effect(learner1_portvals, learner2_portvals, learner3_portvals, verbose=False)

def plot_impact_effect(learner1, learner2, learner3, verbose=False):
    # Normalize
    learner1_n = learner1 / learner1[0][1]
    learner2_n = learner2 / learner2[0][1]
    learner3_n = learner3 / learner3[0][1]

    learner1_cum_ret = learner1.iloc[-1]/learner1.iloc[0] - 1
    learner2_cum_ret = learner2.iloc[-1] / learner2.iloc[0] - 1
    learner3_cum_ret = learner3.iloc[-1] / learner3.iloc[0] - 1

    learner1_dr = (learner1 / learner1.shift(1)) - 1
    learner1_dr = learner1.iloc[1:]
    learner1_dr_mean = learner1_dr.mean()

    learner2_dr = (learner2 / learner2.shift(1)) - 1
    learner2_dr = learner2.iloc[1:]
    learner2_dr_mean = learner2_dr.mean()

    learner3_dr = (learner3 / learner3.shift(1)) - 1
    learner3_dr = learner3.iloc[1:]
    learner3_dr_mean = learner3_dr.mean()

    learner1_dr_std = learner1_dr.std()
    learner2_dr_std = learner2_dr.std()
    learner3_dr_std = learner3_dr.std()

    plt.figure(figsize=(14, 8))
    plt.title("Experiment 2: Effect of impact - Normalized Portfolio Return")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.xticks(rotation=30)
    plt.grid()
    plt.plot(learner1_n, label="impact: 0.00")
    plt.plot(learner2_n, label="impact: 0.025")
    plt.plot(learner3_n, label="impact: 0.05")
    plt.legend()
    plt.savefig("images/Experiment2.png", bbox_inches='tight')
    #plt.show()
    plt.clf()

    if verbose:

        print("Cumulative Return:")
        print("Learner1(impact 0.00):", end="")
        print('%.6f' % learner1_cum_ret)
        print("Learner2(impact 0.10):", end="")
        print('%.6f' % learner2_cum_ret)
        print("Learner3(impact 0.30):", end="")
        print('%.6f' % learner3_cum_ret)

        print("\n")

        print("Average Daily Return:")
        print("Learner1(impact 0.00):", end="")
        print('%.6f' % learner1_dr_mean)
        print("Learner2(impact 0.10):", end="")
        print('%.6f' % learner2_dr_mean)
        print("Learner3(impact 0.30):", end="")
        print('%.6f' % learner3_dr_mean)

        print("\n")

        print("Standard Deviation of Daily Returns:")
        print("Learner1(impact 0.00):", end="")
        print('%.6f' % learner1_dr_std)
        print("Learner2(impact 0.10):", end="")
        print('%.6f' % learner2_dr_std)
        print("Learner3(impact 0.30):", end="")
        print('%.6f' % learner3_dr_std)

def author():
    return 'koush3'

if __name__=="__main__":  		   	  			  	 		   	
    experiment2()



