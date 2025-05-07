# Machine Learning for Trading
Trading requires a lot of attention and sensitivity to the market. It is dependent upon many parameters such as Risk, Volatility, Momentum, Market news etc. These parameters can be interpreted as the Technical Indicators which when combined together can form a “Trading Strategy” that would help us in making Trading decisions. In this project, we will delve into a Human Developed Trading Strategy and a Machine Learning based Trading Strategy (Q-Learner) 

# Indicator Overview
The Technical Indicators used in this project for Manual Strategy and Strategy Learner are Moving average convergence/divergence(MACD), Bollinger Band 
Percent(B%), Momentum[ROC], Exponential Moving Average(EMA 20) and Relative Strength Index(RSI).
# Bollinger Band Percent (B %)
A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences
# Moving Average Convergence/Divergence (MACD) 
Moving average convergence/divergence (MACD, or MAC-D) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security’s price. The MACD line is calculated by subtracting the 26-period EMA from the 12-period EMA.  The result of that 
calculation is the MACD line. A nine-day EMA of the MACD line is called the signal line, which is then plotted on top of the MACD line, which can function as a trigger for buy or sell signals.
# Momentum (ROC) 
The Price Rate of Change (ROC) is a momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago. The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if price changes are to the upside, and moving into negative territory if price changes are to the downside.
# Exponential Moving Average (EMA 20) 
An exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance on the most recent data points. The exponential moving average is also referred to as the exponentially weighted moving average
# Relative Strength Index (RSI)
The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price 
changes to evaluate overvalued or undervalued conditions in the price of that security.

# Manual Strategy
The five Technical Indicators mentioned above are given a certain weightage each and they are pooled together (using a simple logical expression) for a final vote aggregate to decide on the trade action. The possible trade actions include {-2000, -1000, 0, 1000, 2000}. The final possible trade positions are {-1000, 0, 1000}.

# Strategy Learner
Q-Learner requires set of States, Actions and Rewards. Trading has perfect 3 Actions (BUY, SELL and NOTHING). States can be calculated using the market indicators. Using the five indicators with permutation and combination, I have arrived 96 states. Reward is calculated on daily basis based on the portfolio held and the market price on that day. 
