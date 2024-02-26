import pytz
from utils.data_helper import get_today
from utils.performance import *
from utils.data import *
from utils.logging import get_logger
from utils.Threads import CustomThreadPool
from pandas.tseries.offsets import BDay
from plotly.subplots import make_subplots
from datetime import datetime, time
import plotly.graph_objects as go
import plotly

class StrategyBase(object):            

    def __init__(self, strategy_name):        
        self.strategy_name = strategy_name
        self.capital = 20000
        self.is_backtest = True
        self.logger = get_logger(strategy_name)
        self.open_period_to_use_close = 20

        # mutliple threads for getting price
        self.is_multi_thread = True
        self.thread_num = 100

    def display_params(self):                
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        
        self.logger.info(f'capital:                     {self.capital}')
        self.logger.info(f'is_backtest:                 {self.is_backtest}')            
        self.logger.info(f'open_period_to_use_close:    {self.open_period_to_use_close}')
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        

    def set_stock_universe(self, stock_universe=get_sp500_tickers()):
        """
            Stock universe to be considered in the strategy
        """ 
        self.stock_universe = stock_universe

    def set_capital(self, v):
        """
            Capital allocated of the strategy
        """ 
        self.capital = v

    def set_is_backtest(self, v):
        """
            Preprocess or generate position might have different behavior
        """ 
        self.is_backtest = v

    def set_open_period_to_use_close(self, v):
        """
            First x mins of market open to use close price, the open price from Yahoo price api is not accurate during market open 
        """ 
        self.open_period_to_use_close = v

    @cache
    def load_price_data(self, start_date, end_date):
        """
            Function to load the pricing data, ideally this function should take a price API logic and we just need to feed in the price object
            But now we just uses Yahoo API
        """ 
        # Bug from yahoo API, it doesn't include price at end_date            
        px = yf.download(tickers=self.stock_universe, interval="1d",auto_adjust=True, start=start_date, end=end_date+BDay(10))

        if len(self.stock_universe) == 1:
            px.columns = pd.MultiIndex.from_product([px.columns, self.stock_universe])

        px = px[start_date:end_date]

        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz=tz)
        market_open_time = datetime.combine(now.date(), time(9,30), tzinfo=tz)     
        market_open_mins = ((now - market_open_time).seconds - 240) / 60

        if self.is_backtest == False and end_date == get_today() and market_open_mins <= self.open_period_to_use_close:
            self.logger.critical(f'Actual Trade mode: First {self.open_period_to_use_close} mins of market open, use close price instead of open price')            
            px.loc[end_date, 'Open'] = px.loc[end_date, 'Close'].values

        elif self.is_backtest == False:
            self.logger.critical(f'Actual Trade Mode: After {self.open_period_to_use_close} mins of market open, use open price')                        

        return px

    def set_start_date(self, date):
        """
            configure the backtest period
        """ 
        self.start_date = date
        return self

    def set_end_date(self, date):
        """
            configure the backtest period
        """  
        self.end_date = date
        return self
    
    def preprocess_data(self):
        """
            Core logic of the strategy, the function prepares all nescessary data to generate the daily positions

            The output from this function is also useful to analyize
        """    
        pass

    def generate_position(self):
        """
            Core logic of the strategy, the function generates positions data based on output from preprocess_data()
        """
        pass                      

    def actual_trade(self):
        """
            Return the actual position for trading
            There must be some environmental different from backtest, so we createa new function here
        """          
        pass

    # def execute_strategy(self, is_test=True):     
    #     """
    #         Function to execute the strategy
    #     """        
    #     pass
    
    # def exit_strategy(self, is_test=True):
    #     """
    #         Function to exit the strategy
    #     """   
    #     pass

    def backtest(self):
        """
            Function to run the backtest based on defined range            
        """        
        self.set_is_backtest(True)
        self.preprocess_data()
        self.generate_position()
        self.backtest_summary()  

    def generate_backtest_return(self):
        """
            Strategy should implement their own logic to generate return given the positions

            This function needs to define two self variables            
                self.port_ret:  daily portfolio level returns   (nx1)
                self.ret:       daily stock level return        (nxm)

            For example
                2023-11-09: 1
                2023-11-10: 1
                2023-11-11: 0

            An intraday strategy would define return as open to close daily 

                df['Close'] / df['Open'] - 1

            An interday stratety might define return as close to close daily

                df['Close'] / df['Close'].shift(1) - 1

            A strategy enters at open and exits at close, special logic to define the return

                position_entry = position[position.abs().diff() > 0].fillna(0)
                position_hold = position[position.abs().diff() == 0].fillna(0)

                c2c_ret = raw['Close'] / raw['Close'].shift(1) - 1
                o2c_ret = raw['Close'] / raw['Open'] - 1

                ret = position_entry * o2c_ret + position_hold * c2c_ret
                port_ret = ret.sum(axis=1)
        """  
       

    def backtest_summary(self, benchmark=['^SPX', '^IXIC'], trade_info=True):
        """
            Generate backtest summary given positions                            
        """ 
        self.generate_backtest_return()      
        port_ret = self.port_ret
        ret = self.ret
        position = self.position

        # performance summary
        performance_summary_plot(port_ret, self.strategy_name, benchmark=benchmark)

        if trade_info:            
            # % of trades in right directions
            correct_dirs = (np.sign(position) == np.sign(ret)) & (position != 0)
            correct_dirs = correct_dirs.sum(axis=1)
            total_trades = position[position != 0].count(axis=1)
            pct_correct = correct_dirs / total_trades

            # Total of Trades
            total_trades = position[position != 0].count(axis=1)
            total_long = position[position > 0].count(axis=1)
            total_short = position[position < 0].count(axis=1)

            colors = plotly.colors.DEFAULT_PLOTLY_COLORS
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{},{}], 
                    [{"colspan": 2}, None]],        
                subplot_titles=[
                    'Daily % of trades in correct direction',
                    '',
                    'Number of Trades',        
                ],    
                vertical_spacing=0.10,
            )
            fig.update_layout(width=1300, height=800, legend=dict(y=0.4), hovermode='x')

            fig.add_trace(go.Histogram(x=pct_correct, marker=dict(color=colors[0]), showlegend=False, nbinsx=40), row=1, col=1)
            fig.add_trace(go.Scatter(x=total_trades.index, y=total_trades, marker=dict(color=colors[0]), showlegend=True, name='# Trades'), row=2, col=1)
            fig.add_trace(go.Scatter(x=total_long.index, y=total_long, marker=dict(color=colors[1]), showlegend=True, name='# Total Long'), row=2, col=1)
            fig.add_trace(go.Scatter(x=total_short.index, y=total_short, marker=dict(color=colors[2]), showlegend=True, name='# Total Short'), row=2, col=1)
            fig.show()

                    

            
        