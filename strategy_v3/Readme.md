# Strategy V3

Strategy V3 here is more for trading crypto using <b>grid tradings</b>.

Utilmate goal is to build a grid trading bots which trades 24 hours on BTCUSD or ETHUSD which are highly volalite instruments

I expect the strategy could benefit from tradnig crypto

# Implementation

The implementation is differnt from strategy_v2 as grid strategy is based on orders and execution rather than pure position signals.

We need to build a strategy class which receive data from vendor (Binance/Coinbase), send order and execute via Executor

Executor will be a abstract class which has all features of place order , cancel orders , close position etc. we also need a backtest Executor to replicate all these function

Finally, if we expect to more than one more strategy and need some capital control, we also need to build a trading sub system similar to strategy v2.

# Grid Trading Logic (Arithmetic)

### For each time interval, iterate thru below steps

1. check if status is idle (i.e. no outstanding grid orders) and hurst exponent to see if this indicates mean-reverting trends

2. If both yes for above, place grid orders via LIMIT ORDER

- num orders   = grid_size * 2
- grid spacing = historical volatility * vol scale
- stop loss    = historical volatility * vol scale * vol_stoploss_scale

3. fill the orders using high and low (backtest mode only)

4. check if status neutral (i.e. have filled grid orders but position are neutral). If yes, cancel all orders

5. check if current price triggers stop-loss. If yes, cancel all orders and close out position via MARKET ORDER.

### Notes

Improvement: we should be able to predict either mean-reverting or trending and place a sutiable grids

e.g. if the price is trending up, instead of placing a grid of 5-sells and 5-buys centered at current price, we can place a grid with 5-sells and 5-buys centered at current price + 1% to capture the momentum

