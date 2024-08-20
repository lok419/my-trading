# Systematic Trading Framework

## Background

- Framework that is designed to coorporate into multiple trading strategy

- The Framework is desgined to coorprate into <b>both discretionary trading and alogrithm tradnig (even more complex quantitative strategy)</b> such that we can both

    - Trader can put discretionary trades into certin stocks, simple algo helps to enter the position

    - Framework assists to attribute captial among differet strategy and optimize portfolio Sharpe Ratio

    - Modularize the strategy so we can develop new strategy independently

    - Regular rebalance based on performance / volatility of different strategy


## Components

1. Core Strategy

2. Trading subsystem

3. Portfolios

4. Executors


## Core Strategy

This is a strategy class which defines the position to trade, the class should have below function. <b>Ideally, I want the core class only deal with strategy logic without any other stuff. All returnability prediction or alpha generation should be done here.</b>

- Type: Algo / Discretionary

- Set position data (could be more than daily OHCL)

- Set Hyper-parameters (same strategy object could be build with different variations)

- Strategy function which generates <b>Matrix of daily position</b>

#### Example - Buy and Hold Strategy (Full Discretionary)

- Strategy which holds one stock, and generate position will always be 1.

- If you have more than one stocks to buy and hold, you should <b>create multiple Buy and Hold strategy</b> rather than put than into one single strategy, this is because <b>(3) Portfolios</b> would handle the captial attribute among different stocks

#### Example - Algorithm Trading (Half discretionary)

- Strategy defines one stock to trade. Trending followings strategy generate position between -1 and 1 based on confidence in forecast.

- Some hyperparameters needs to be defined

#### Example - Quantitative Trading

- Strategy generate outputs for a many stocks (e.g. SP500), complex logic to determine the positions. The stock universe could be quite dynamic


## Trading Subsystem

- One Trading subsystem contains similar strategy with different variation, it could be <b>the same strategy with different hyper-parameters</b>.

- The subsystem will collat all position from different strategy's variations and combine into one positions.

- <b>Backtest / performance / GetData / stock universe function are implemented here</b>

- Main implementation is the core logic to combine weights from the different strategy outputs (e.g. could be equal weighted)

    - <b>Volatility targeting</b> - The combined weights should align with your volatility targeting (e.g. risk tolerance),

    - <b>Diversification multiplier</b> [<i>NOT IMPLMENTED</i>] - Optionally, we need to scale up the combined weights, as the combined weights tends to make the position smaller than original volatiltiy targeting
    
    - Assume we trade 100% of the capial within this subsystem

## Portfolios

- Portfolios class allocate captial to different Trading subsystem. <b>It is NOT designed to generate alpha, but more to attribute and rebalance among strategies</b>

- It looks at backtest or realized PL of the subsystem and allocate the captial.

- <b>Ideally, I want to follow the MAB algorithm which good strategy will dominate, and bad strategy will be dumpped over time</b>, but we could also use MPT and treat each of the subsystem as "stock" and optimize the weights

- Current available portfolios:
    - <b>Mean Variance Optimization</b>: It looks at backtest return per strategy and maximize objective function $ret - gamma * risk$ via quadratic programming 

- Other functionaility:
    - <b>Backtesting</b>: This allows you to backtest the optimized portfolio performance
    - <b>Rebalance</b>: Class enables users to choose rebalance frequence (in cron). E.g. RebalancerIter('0 0 * * Fri', 2) => Rebalance on Friday every 2 weeks.
    - <b>Transaction Cost Model</b>: Portfolio can take transaction costs model into account for backtesting. E.g. TransactionCostFutu() which is coded at exact tiered fee from Futu official website


## Executors

- Executor will look at your current position, and target portfolios and execute accordingly. Ideally, all above logic doesn't tie to a single execution platform.

- Some trading cost optimization could be done here

    - Regular weekly or biweekly rebalance (or we can define a rebalance frequence for each sub-system)

    - position inertia - i.e. do not trade the delta in notional is samller than $x

- Different execution channels. (your portfolio could have cryto which should be executed in Binance)

- Supported channels: Futu

## Notes

- In my design, I put control of volatility target in sub-system, and portfolios allocation is more to optimize Sharpe Ratio and performance</b>

- I want all strategy to be deterministic, i.e. exact positions or on a date should be reproducible. So I avoid any intraday strategy for now.

## Persudo Codes

```python 
# My Ideal persudo codes of portfolios
capital = 1000000
vol_target = 0.2
date = datetime.today()

trading_subsystems = [
    TradingSubSystemBase(instruments='AAPL', vol_target=vol_target, strategy=[BuyAndHoldStrategy(position=1)]),
    TradingSubSystemBase(instruments='TSLA', vol_target=vol_target, strategy=[BuyAndHoldStrategy(position=1)]),      
    TradingSubSystemBase(instruments='NVDA', vol_target=vol_target, strategy=[BuyAndHoldStrategy(position=0.8)]),
    TradingSubSystemBase(instruments='SPX', vol_target=vol_target, strategy=[
        MonmentumTrends(window1=5, window=10)
        MonmentumTrends(window1=20, window=50)
        MonmentumTrends(window1=60, window=252)
    ]),        

    # I want this quant strategy to have samller volatility because of uncertainty of the my algo
    TradingSubSystemQuant(instruments=[.....], vol_target=vol_target * 0.8, strategy=[
        MeanReversion(threshold1=0.05, threshold2=0.05)        
        MeanReversion(threshold1=0.1, threshold2=0.1)        
        MeanReversion(threshold1=0.1, threshold2=0.05) 
        MeanReversion(threshold1=0.2, threshold2=0.4) 
    ]),
]

# Generate Target portfolios as of date, and ideally, all optimization should be deterministic 
portfolios = Portfolio(subsystems=trading_subsystems, date=date, capital=capital)
portfolios.optimize()
portfolios.save()

executor = Executor(portfolios)
executor.rebalance()
executor.trade()
```