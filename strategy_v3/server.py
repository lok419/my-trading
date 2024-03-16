from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
from telegram.constants import ParseMode
from utils.credentials import TELEGRAM_BOT_API_KEY, NGROK_DOMAIN, NGROK_PORT_TUNNEL, NGROK_DOMAIN_URL
from strategy_v3.ExecuteSetup import ExecuteSetup, StrategyFactory
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.Executor import ExecutorBinance
from tabulate import tabulate
from utils.logging import get_logger
import json
import warnings
import tempfile
import subprocess
import traceback
import inspect
import numpy as np

warnings.filterwarnings('ignore')
logger = get_logger('Telegram Bot')

default_update_options = {
    'grid_size': [3,5,7,10],
    'vol_lookback': [10, 15, 20, 30, 40, 60],
    'vol_grid_scale': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3 ,0.4],
    'vol_stoploss_scale': [1,2,3,4,5],
    'position_size': [10, 50, 100, 300, 500, 1000],
    'refresh_interval': [10,20,30,60,180,300],
    'spread_adv_factor': [0.01,0.03,0.05,0.1],
    'hurst_exp_mr_threshold': [0, 0.4, 0.5, 0.6],
    'hurst_exp_mo_threshold': [0.6, 0.7, 0.8, 1],        
}

def handler_print(func):
    '''
        Decorator: Print the handle command here
    '''
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message:
            logger.info(f"[{update.message.from_user.username}] {update.message.text}")

        elif update.effective_chat and update.callback_query:
            logger.info(f"[{update.effective_chat.username}] {update.callback_query.data}")

        await func(update=update, context=context)

    return inner

def handler_expcetion(func):
    '''
        Decorator: Handle Exception for each handler to response back the error message to users
    '''
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await func(update=update, context=context)
        except Exception as e:
            traceback.print_exception(e)
            logger.error(e)
            msg = f'failed - {e}'
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    return inner

async def button_callback(update_: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
        This is the callback whenever user clicks the button
    '''
    data = update_.callback_query.data    
    command = data.split(" ")[0].replace('/', '')
    context.args = data.split(" ")[1:]        
    await context.bot.send_message(chat_id=update_.effective_chat.id, text=data)    
    await eval(command)(update_, context)

@handler_expcetion
@handler_print
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):            
    '''
        Print out all active strategy
    '''        
    config = ExecuteSetup.read_all() 
    button_list = []    
    for s in list(config.keys()):
        button_list.append([InlineKeyboardButton(text=s, callback_data=f'/action {s}')])
    button_list.append([InlineKeyboardButton(text='config', callback_data=f'/config')])
    markup = InlineKeyboardMarkup(button_list)    
    await context.bot.send_message(chat_id=update.effective_chat.id, text='List of strategy and global config:', reply_markup=markup)

@handler_expcetion
@handler_print
async def action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = context.args[0]
    button_list = []    
    button_list.append([InlineKeyboardButton(text='Back', callback_data=f'/start')]) 
    button_list.append([InlineKeyboardButton(text='Config', callback_data=f'/config {s}')])  
    button_list.append([InlineKeyboardButton(text='Update', callback_data=f'/update {s}')])  
    button_list.append([
        InlineKeyboardButton(text='Run', callback_data=f'/update {s} status RUN'),
        InlineKeyboardButton(text='Pause', callback_data=f'/update {s} status PAUSE'), 
        InlineKeyboardButton(text='Terminate', callback_data=f'/update {s} status TERMINATE'),
        InlineKeyboardButton(text='Stop', callback_data=f'/update {s} status STOP'),        
    ])

    times = ['2 Hours Ago', '4 Hours Ago', '12 Hours Ago', '1 Days Ago', '5 Days Ago', '10 Days Ago', '30 Days Ago', 'LTD']
    for t in times:
        button_list.append([InlineKeyboardButton(text=t, callback_data='/')])
        button_list.append([
            InlineKeyboardButton(text=f'PnL', callback_data=f'/pnl {s} {t}'),
            InlineKeyboardButton(text=f'Plot', callback_data=f'/plot {s} {t}'),
            InlineKeyboardButton(text=f'Summary', callback_data=f'/summary {s} {t}')
        ])                

    markup = InlineKeyboardMarkup(button_list)    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Actions for strategy {s}:', reply_markup=markup)

@handler_expcetion
@handler_print
async def update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = context.args[0]    
    key = context.args[1] if len(context.args) > 1 else None
    value = context.args[2] if len(context.args) > 2  else None 

    # display which params to update
    if key is None and value is None:   

        strategy = StrategyFactory().get(s)
        default_args = list(inspect.signature(type(strategy).__init__).parameters.keys())        

        button_list = []        
        button_list.append([InlineKeyboardButton(text='Back', callback_data=f'/action {s}')])
        button_list.append([InlineKeyboardButton(text='Config', callback_data=f'/config {s}')])
        
        for p in default_update_options:
            if p in default_args:
                button_list.append([InlineKeyboardButton(text=p, callback_data=f'/update {s} {p}')])    

        markup = InlineKeyboardMarkup(button_list)    
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Strategy {s} params:', reply_markup=markup)

    # display a default options
    elif key is not None and value is None:
        button_list = []   
        button_list.append([InlineKeyboardButton(text='Back', callback_data=f'/update {s}')])
        button_list.append([InlineKeyboardButton(text='Config', callback_data=f'/config {s}')])

        current_val = ExecuteSetup(s).read()[key]                
        default_vals = default_update_options[key]
        default_vals.append(current_val)
        default_vals = sorted(list(set(default_vals)))

        for v in default_vals:
            text = f'{v} (current)' if v == current_val else v                    
            button_list.append([InlineKeyboardButton(text=text, callback_data=f'/update {s} {key} {v}')])    

        markup = InlineKeyboardMarkup(button_list)    
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Strategy {s} params {key}:', reply_markup=markup)

    # update a key and value
    elif key is not None and value is not None:
        strategy = StrategyFactory().get(s)        
        setup = ExecuteSetup(s)        
        setup.update(key, value, type(strategy))
        config = setup.read()
        config = str(json.dumps(config, indent=4))        
        msg = 'succeeded\n'
        msg += config
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


@handler_expcetion
@handler_print
async def config(update: Update, context: ContextTypes.DEFAULT_TYPE):    
    '''
        Print out all strategy configuration        
    '''
    config = ExecuteSetup.read_all()
    if len(context.args) > 0:
        strategy_id = context.args[0]
        config = config[strategy_id]
    
    msg = str(json.dumps(config, indent=4))    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

@handler_expcetion
@handler_print
async def pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):        
    '''
        Table summary of strategy pnl performance 
    '''        
    strategy_id = context.args[0]
    time = " ".join(context.args[1:])    
    time = "" if time == 'LTD' else time

    strategy = StrategyFactory().get(strategy_id)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)
    strategy.load_data(time)

    table = strategy.summary_table()
    table = tabulate(table, headers='keys', tablefmt='psql', showindex=False)    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)

@handler_expcetion
@handler_print
async def plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
        Plot summary of strategy performance
    '''    
    strategy_id = context.args[0]
    time = " ".join(context.args[1:])    
    time = "" if time == 'LTD' else time

    strategy = StrategyFactory().get(strategy_id)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)
    strategy.load_data(time)

    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as file:        
        strategy.summary(plot_orders=True, lastn=20, save_jpg_path=file.name, show_pnl_metrics=False)        
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file.name)    

@handler_expcetion
@handler_print
async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
        display pnl metrics summary table and plots
    '''    
    strategy_id = context.args[0]
    time = " ".join(context.args[1:])
    time = "" if time == 'LTD' else time
    
    strategy = StrategyFactory().get(strategy_id)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)
    strategy.load_data(time)

    with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as file:        
        strategy.summary(plot_orders=True, save_jpg_path=file.name, show_pnl_metrics=False)
        table = strategy.df_pnl_metrics
        table = tabulate(table, headers='keys', tablefmt='psql', showindex=False)    
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file.name)        

if __name__ == '__main__':
    # tunnel localhost:5001 to public static domain
    subprocess.Popen(f"nohup ngrok http --domain={NGROK_DOMAIN} {NGROK_PORT_TUNNEL} >/dev/null 2>&1 &", shell=True)

    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()        
    application.add_handler(CommandHandler('start', start)) 
    application.add_handler(CommandHandler('config', config))       
    application.add_handler(CommandHandler('update', update))   
    application.add_handler(CommandHandler('pnl', pnl))  
    application.add_handler(CommandHandler('plot', plot))  
    application.add_handler(CommandHandler('summary', summary))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.run_webhook(listen='127.0.0.1', port=5001, webhook_url=NGROK_DOMAIN_URL)