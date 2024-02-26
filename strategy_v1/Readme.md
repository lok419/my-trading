# Interday / Intraday Strategy

The module is more for intraday strategy, incldues

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