# CompProject3012
Bayesian Surprise Classifier – MA Crossover Strategy Performance

This repository contains visual performance summaries for the moving average (MA) crossover strategies evaluated in my undergraduate dissertation: Enhancing Systematic Trading Strategies with Bayesian Surprise Signals from Implied Volatility.
Overview

Each image in this repository displays the backtested performance of a specific MA crossover strategy, tested with two variations that incorporate Bayesian surprise signals derived from options market implied volatility.

The strategies are tested across multiple FX pairs and stock indices. Results are broken down by asset and crossover type (e.g. 5/13, 13/25, 20/50, 50/100), and include:

Cumulative returns over time
Comparative plots for baseline vs surprise-enhanced strategies
These visualisations support the quantitative analysis discussed in the dissertation, helping demonstrate the impact of volatility-driven surprise signals on trend-following systems.

Strategies Tested

Two variations of each MA crossover strategy were implemented to assess the effect of Bayesian surprise signals:

Surprise-Filtered Strategy:
Trades are only executed when a moving average crossover signal aligns with a detected Bayesian surprise in implied volatility. This acts as a filter, aiming to avoid trades in low-information environments.

Surprise-Weighted Position Sizing Strategy:
All crossover signals are traded, but the position size is doubled when a Bayesian surprise is detected. This variation tests whether surprise signals can improve performance by scaling exposure based on the perceived information content of market conditions.
