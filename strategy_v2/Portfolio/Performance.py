from plotly.subplots import make_subplots
from itertools import cycle
from IPython.display import display
from utils.performance import get_benchmark_return, performance_summary_table, cumulative_return, cumlative_log_return
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
import plotly
import numpy as np
import math

class Performance(object):
    '''
        Extended Class from Portfolios to consolidate all performance related functions
    '''
    def __init__():
        pass

    def performance(self, 
                    benchmark:list[str]=['^SPX'],
                    benchmark_equal_weighted:bool=True,
                    show_all_rets: bool=True,                    
        ):        
        '''
            benchmark:       Benchmarks to compare in performance plots
            show_all_rets:   True if you want to show all sub-system return. Otherwise, only show the portfolio returns
        '''        
        # avoid modifying the default benchmark args
        benchmark = benchmark.copy()
        all_rets = self.ret.copy()
        all_rets['Optimized Portfolio'] = self.port_ret           
        all_rets['Rebalanced Portfolio'] = self.port_ret_rebal        

        # function are copied from utils.performance libs but with extra plots added        
        start_date = None
        end_date = None
        for s in all_rets:            
            all_rets[s] = all_rets[s].fillna(0)
            start_date = min(all_rets[s].index.min(), start_date) if start_date is not None else all_rets[s].index.min()
            end_date = max(all_rets[s].index.max(), start_date) if start_date is not None else all_rets[s].index.max()      
        
        if len(benchmark) > 0:        
            for b in benchmark:                        
                all_rets[b] = get_benchmark_return(b, start_date, end_date)

        if benchmark_equal_weighted:               
            all_rets['Equal Weighted'] = get_benchmark_return(tuple(self.instruments), start_date, end_date)
            benchmark += ['Equal Weighted']

        systems_name = [str(s) for s in self.systems]

        # re-order the naming for convenience 
        top_names = ['Rebalanced Portfolio', 'Optimized Portfolio'] + benchmark        
        all_rets = all_rets[top_names + systems_name]

        display(performance_summary_table(all_rets))
        
        fig = make_subplots(
            rows=12, cols=1,        
            row_heights=[0.6, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            subplot_titles=[
                'Strategy Cumulative Log Return',                
                'Daily Return (%)',
                'Volatility 1m (%)',
                'Strategy Weights (%)',
                'Rebalanced Portfolio Leverage (%) (excl. MTM)',                
                'Rebalanced Portfolio Weights (%) (excl. MTM)',
                'Rebalanced Portfolio Shares',
                'Rebalanced Portfolio Value',
                'Instrument Price',                
                'Instrument Volatility 20D (%)',    
                'Instrument Return (%)',    
                'Transaction Cost',
            ],    
            vertical_spacing=0.03,
            shared_xaxes=True,                    
        )    
        fig.update_layout(
            width=1500, height=4000,
            xaxis_showticklabels=True, 
            xaxis2_showticklabels=True, 
            xaxis3_showticklabels=True,
            xaxis4_showticklabels=True, 
            xaxis5_showticklabels=True, 
            xaxis6_showticklabels=True, 
            xaxis7_showticklabels=True, 
            xaxis8_showticklabels=True, 
            xaxis9_showticklabels=True, 
            xaxis10_showticklabels=True, 
            xaxis11_showticklabels=True, 
            xaxis12_showticklabels=True, 
            hovermode='x',            
        )
        fig.update_layout(hoverlabel=dict(namelength=-1))

        colors = plotly.colors.qualitative.Plotly
        colors_iter = cycle(colors)          
        
        for s, r in all_rets.items():                              
            # do not show strategy returns when show_all_rets = False
            if show_all_rets or s not in systems_name: 
                c = colors_iter.__next__()     

                lw = 3 if s in top_names else 1
                fig.add_trace(go.Scatter(x=r.index, y=cumlative_log_return(r), name=s, legendgroup=s, marker=dict(color=c), line=dict(width=lw)), row=1, col=1)
                fig.add_trace(go.Scatter(x=r.index, y=100*r, name=s, showlegend=False, legendgroup=s, marker=dict(color=c)), row=2, col=1)            

                v20 = r.rolling(20).std()*100*np.sqrt(252)                
                fig.add_trace(go.Scatter(x=v20.index, y=v20, name=s, showlegend=False, legendgroup=s, marker=dict(color=c), line=dict(width=3)), row=3, col=1)                

                # Transaction Cost
                if s == 'Optimized Portfolio':
                    # exclude cash for leverage
                    leverage_opt = self.port_position.loc[:, self.port_position.columns != 'CASH'].sum(axis=1)                    
                    fig.add_trace(go.Scatter(x=leverage_opt.index, y=leverage_opt*100, name=s, showlegend=False, legendgroup=s, marker=dict(color=c), line=dict(width=lw)), row=5, col=1)            
                    fig.add_trace(go.Scatter(x=self.port_tc.index, y=self.port_tc.cumsum(), name=s, showlegend=False, legendgroup=s, marker=dict(color=c)), row=12, col=1)
                    

                elif s == 'Rebalanced Portfolio':
                    # exclude cash for leverage
                    leverage_rebal = self.port_position_rebal_strike.loc[:, self.port_position_rebal_strike.columns != 'CASH'].sum(axis=1)                                        
                    fig.add_trace(go.Scatter(x=leverage_rebal.index, y=leverage_rebal*100, name=s,showlegend=False, legendgroup=s, marker=dict(color=c), line=dict(width=lw)), row=5, col=1)        
                    fig.add_trace(go.Scatter(x=self.port_tc_rebal.index, y=self.port_tc_rebal.cumsum(), name=s, showlegend=False, legendgroup=s, marker=dict(color=c)), row=12, col=1)

                elif s not in benchmark:
                    # exclude cash for leverage
                    leverage_strat = self.position[s]
                    leverage_strat = leverage_strat.loc[:, leverage_strat.columns != 'CASH'].sum(axis=1)                    
                    fig.add_trace(go.Scatter(x=leverage_strat.index, y=leverage_strat*100, name=s, showlegend=False, legendgroup=s, marker=dict(color=c), line=dict(width=lw)), row=5, col=1)        

        colors_iter = cycle(colors)                
        for s in systems_name:     
            c = colors_iter.__next__()           
            w = self.port_w[s]            
            fig.add_trace(go.Scatter(x=w.index, y=100*w, name=s, showlegend=not show_all_rets, legendgroup=s, marker=dict(color=c)), row=4, col=1)                  

        instruments = self.instruments     
        colors_iter = cycle(colors)        

        for i in instruments:
            c = colors_iter.__next__()        

            pos = self.port_position_rebal_strike[i]
            pos_shs = self.port_position_shs_rebal[i]            
            pos_dp = self.port_position_dp_rebal[i]

            px = self.close_px[i]
            px_return = (1+px.pct_change().fillna(0)).cumprod()*100

            r = px / px.shift(1) - 1
            v20 = r.rolling(20).std()*100*np.sqrt(252)            

            # instrument weights in portfolio
            fig.add_trace(go.Scatter(x=pos.index, y=pos*100, name=i, showlegend=True, legendgroup=i, marker=dict(color=c)), row=6, col=1)     

            # instrument shares in portfolio
            fig.add_trace(go.Scatter(x=pos.index, y=pos_shs, name=i, showlegend=False, legendgroup=i, marker=dict(color=c)), row=7, col=1)     

            # instrument value in portfolio
            fig.add_trace(go.Scatter(x=pos_dp.index, y=pos_dp, name=i, showlegend=False, legendgroup=i, marker=dict(color=c)), row=8, col=1) 

            # instrument price
            fig.add_trace(go.Scatter(x=px.index, y=px, name=i, showlegend=False, legendgroup=i, marker=dict(color=c)), row=9, col=1)                 
            
            # instrument price volatiltiy
            fig.add_trace(go.Scatter(x=v20.index, y=v20, name=i, showlegend=False, legendgroup=i, marker=dict(color=c), line=dict(width=3)), row=10, col=1)            

            # instrument cum return
            fig.add_trace(go.Scatter(x=px.index, y=px_return, name=i, showlegend=False, legendgroup=i, marker=dict(color=c)), row=11, col=1)     

        fig['layout']['yaxis2']['title']= 'Return (%)'
        fig['layout']['yaxis3']['title']= 'Vol (%)'
        fig['layout']['yaxis4']['title']= 'Weight (%)'
        fig['layout']['yaxis6']['title']= 'Weight (%)'
        fig['layout']['yaxis7']['title']= 'Shares'
        fig['layout']['yaxis8']['title']= 'Value ($)'
        fig['layout']['yaxis9']['title']= 'Price ($)'        
        fig['layout']['yaxis10']['title']= 'Vol (%)'
        fig['layout']['yaxis11']['title']= 'Return (%)'
        fig['layout']['yaxis12']['title']= 'Cost ($)'

        fig.show()

    def portfolio_breakdown(self, lookback=20):
        '''
            Breaking each subsystem into instrument holdings
        '''
        end_date = self.end_date
        start_date = end_date - BDay(lookback)

        cols = 3 
        rows = math.ceil(len(self.systems) / cols)

        fig = make_subplots(            
            rows=rows, cols=cols,                    
            subplot_titles=[str(s) for s in self.systems],    
            vertical_spacing=0.15,     
            shared_xaxes=True,            
        )   

        fig.update_layout(
            title=f'Sub-Portfolios Breakdown (Last {lookback} BDays)',
            width=1500, height=170+250*rows,                   
            hovermode='x',            
        )
        fig.update_xaxes(showticklabels=True)
        fig.update_xaxes(matches='x')  
        fig.update_layout(hoverlabel=dict(namelength=-1))                   

        showlegend = {}
        colors = plotly.colors.qualitative.Plotly
        colors_iter = cycle(colors)  
        colorsmap = {}

        for idx, s in enumerate(self.systems):   
            for i in s.instruments:
                colorsmap[i] = colorsmap[i] if i in colorsmap else next(colors_iter)
                pos = s.position[str(s)][i][start_date:end_date]                
                fig.add_trace(go.Scatter(x=pos.index, y=pos*100, name=i, legendgroup=i, showlegend=showlegend.get(i, True), marker=dict(color=colorsmap[i])), row=idx//cols+1, col=idx%cols+1)     
                showlegend[i] = False

        fig.show()

    def instrument_breakdown(self, lookback=20):
        '''
            Breaking each instrument by holding in each system
        '''
        end_date = self.end_date
        start_date = end_date - BDay(lookback)

        # preserve the original orders while dropping duplicates
        instruments = self.instruments        

        cols = 3 
        rows = math.ceil(len(instruments) / cols)        

        fig = make_subplots(            
            rows=rows, cols=cols,                    
            subplot_titles=[str(i) for i in instruments],    
            vertical_spacing=0.05,     
            shared_xaxes=True,             
        )   

        fig.update_layout(
            title=f'Instrument Breakdown (Last {lookback} BDays)',
            width=1500, height=170+250*rows,                   
            hovermode='x', 
        )
        fig.update_xaxes(showticklabels=True)
        fig.update_xaxes(matches='x') 
        fig.update_layout(hoverlabel=dict(namelength=-1))

        showlegend = {}        
        colors = plotly.colors.qualitative.Plotly
        colors_iter = cycle(colors)  
        colorsmap = {}

        for idx, i in enumerate(instruments):
            for s in self.systems:
                pos = s.position[str(s)]
                if i in pos.columns:
                    colorsmap[s] = colorsmap[s] if s in colorsmap else next(colors_iter)
                    pos = pos[i][start_date:end_date]                    
                    fig.add_trace(go.Scatter(x=pos.index, y=pos*100, name=str(s), legendgroup=str(s), showlegend=showlegend.get(str(s), True), marker=dict(color=colorsmap[s])), row=idx//cols+1, col=idx%cols+1)                    
                    showlegend[str(s)] = False

        fig.show()
        