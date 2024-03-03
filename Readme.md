# Welcome to my trading notebooks

# Background
Here is my own trading research on everything I interested. This involves lots of utility function from stats, exchange, logging, factor loading etc.... I have been trying out new things since half-year ago, mainly focus on US stocks and cryto

All of the codes here are not just reasearch, but it is designed to be actual executable strategy (might or might not profitable) with systematic risk control. Some of them are stil working, some of them are abandoned.

Here I have strategy v1, v2, v3 here. Each of them has a different trading framework designed for specific trading. Here I will briefly describe what is it doing.

# Strategy v1 (abandoned for now)

Strategy v1 mainly includes some intraday and interday mean reversion strategy

1. Interday Mean Reversion
- It looks for today's price and T-1's price and bet for mean reversion intraday
- This strategy is tested out that it suffers from high transaction costs if we trade then small volume ($20,000)
- The actual profit even exclude transaction costs doesn't look very promising

2. Post Earnings AnnouncementDrift
- It looks for post earnings anouncement price vs pre, and bet for mean reversion intraday
- This strategy is only applicable in earnings seasons, and the backtest profit is smaller than 1)

3. Interday Mean Reversions
- It searches for cointegrated paris (could be >2 stocks) and trade the spreads
- I haven't spent so much time on the strategy, stil some proof for improvement here
- Main challenging part is how to trade spreads (perhaps grid trading?)

# Strategy v2 (Systematic Trading Framework)

## Background

- Framework is mainly referenced to book <b>Systematic Trading A unique new method for designing trading and investing systems Robert Carver</b>

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

This is a strategy class which defines the position to trade, the class should have below function. <b>Ideally, i want strategy class only deal with strategy logic without any other stuff.</b>

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

- Main implementation is the core logic to combine weights from the different outputs (e.g. could be equal weighted)

    - <b>Volatility targeting</b> - The combined weights should align with your volatility targeting (e.g. risk tolerance),

    - <b>Diversification multiplier</b> - Optionally, we need to scale up the combined weights, as the combined weights tends to make the position smaller than original volatiltiy targeting
    
    - Assume we trade 100% of the capial within this subsystem

## Portfolios

- Portfolios class allocate captial to different Trading subsystem

- It looks at out of sample back-test or realized PL of the subsystem and allocate the captial.

- <b>Ideally, I want to follow the MAB algorithm which good strategy will dominate, and bad strategy will be dumpped over time</b>, but we could also use MPT and treat each of the subsystem as "stock" and optimize the weights

## Executors

- Executor will look at your current position, and target portfolios and execute accordingly. Ideally, all above logic doesn't tie to a single execution platform.

- Some trading cost optimization could be done here

    - Regular weekly or biweekly rebalance (or we can define a rebalance frequence for each sub-system)

    - position inertia - i.e. do not trade the delta in notional is samller than $x

- Different execution channels. (your portfolio could have cryto which should be executed in Binance)

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

# Strategy V3 (Grid Trading)

Strategy V3 here is more for trading crypto using <b>grid tradings</b>.

Utilmate goal is to build a grid trading bots which trades 24 hours on BTCUSD or ETHUSD which are highly volalite instruments

I expect the strategy could benefit from tradnig crypto

## Implementation

The implementation is differnt from strategy_v2 as grid strategy is based on orders and execution rather than pure position signals.

We need to build a strategy class which receive data from vendor (Binance/Coinbase), send order and execute via Executor

Executor will be a abstract class which has all features of place order , cancel orders , close position etc. we also need a backtest Executor to replicate all these function

Finally, if we expect to more than one more strategy and need some capital control, we also need to build a trading sub system similar to strategy v2.

## Execute Logic (Arithmetic)

### For each time interval, iterate thru below steps

1. check if status is idle (i.e. no outstanding grid orders) and hurst exponent to see if this indicates mean-reverting trends

2. If both yes for above, place grid orders via LIMIT ORDER

- num orders   = grid_size * 2
- grid spacing = historical volatility * vol scale
- stop loss    = historical volatility * vol scale * vol_stoploss_scale
- historical volatility = Average true range over Y days

3. fill the orders using high and low (backtest mode only)

4. check if status active neutral (i.e. have filled grid orders but position are neutral). If yes, cancel all orders

5. check if current price triggers stop-loss. If yes, cancel all orders and close out position via MARKET ORDER.

## Control Server

Strategy runs 247, and we can't stay in front of the computer all the days. Therefore, I have built a telegram bot which 

- monitor strategy performance (table, plots) over period of 2,4,6,12 hours.... (can be any)
- update strategy parameters. Strategy is not perfect, we will need to constantly adjust the parameters especailly the grid spacing
- interrupt strategy. We might need to termiante or pause the strategy in some case. (i.e. flatten all delta)
- risk update (not done yet). Alert if there are consecutive lossing trades are made

## Strategy logging

2024-02-27: Attempt to use rolling average metrics based on looking back 2 * half life interval. Replace vol with half-life vol and center price from spot to half-life SMA close

-  Tested BTCFDUSD on 15days. original cum returns is 6% whereas new change is 3% only

2024-02-28: Attempt to use momentum order when hurst exponent is >= 0.6 and use extra momentum filters (Spot > T-5 > T-10) to put the momentum grid orders.

-  Tested BTCFDUSD. this added more return on original strategy, because this is mutually exclusive with mean reverting orders, this enhance return during non-mean-reverting periods 

2024-02-29: Changed the momentum filters to be Spot > T-5 High and T-5 Low > T-10 High to be more conservative

- Exectue refresh time updated from 30s to 1m to avoid small price volatolity to trigger the stoploss

- For backtesting, shall we consider the interval high/low if this trigger stop-loss rather than just check close price???

- [BTCFDUSD] Realized returns are now 2% after fixing the fill price

- <b>Follow up: need to use STOP_LOSS_LIMIT for momentum orders</b>, because now all limit buy orders above market price are filled immediately, but we want to avoid that.

2024-03-01: Updated momentum orders to use STOP-LOSS-LIMIT to avoid LIMIT ORDER executed immediately

- Able to split the PnL from momentum orders and mean reverting orders.

- [BTCFDUSD] Realized returns are now 4%

- [SOLFDUSD] started to trade SOLFDUSD (need to change the quantity decision = 2)

- <b>we need to build a server in order to systematically runs for multiple strategies, meanwhile we can also interrupt the model parameters during runtime</b>

2024-03-02: Attempt to build telegram bot 

- trying to explore reduce grid spacing and lookback periods. In backtest, reducing lookback periods generally has better performance

2024-03-03: Still building telegram bot to control the model parameters on the fly by mobile phone

- the backtest filling logic is to aggressive. In reality, the orders aren't filled like what we assumed in backtest. we need to update the backtest filling logic 

2024-03-04: replace volatility from close std to average true return. This is because ATR has considered all interval high and low whereas close std is just a metrics on close price

- add interrupt function to strategy: RUN, PAUSE, TERMINATE, STOP

## TODO

- enhance backtest fill logic, high and low price are likely unfill-able in reality, we need to account for that

- enhance the center price logic for mean-revert order, need to use the mean price in previous periods instead of current price as center price

- enhance telegram to update a set of predefined model parameters (like vol_grid_scale to be 0.1,0.2,0.3....)

- enhance hurst exponent ratio to be shorter time frame (now is 100)








