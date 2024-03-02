from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
from telegram.constants import ParseMode
from utils.credentials import TELEGRAM_BOT_API_KEY, NGROK_DOMAIN, NGROK_PORT_TUNNEL, NGROK_DOMAIN_URL
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.Strategy import GridArithmeticStrategy
from tabulate import tabulate
from utils.logging import get_logger
import json
import warnings
import tempfile
import subprocess

warnings.filterwarnings('ignore')
logger = get_logger('Telegram Bot')

def handler_print(func):
    '''
        Decorator: Print the handle command here
    '''
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message:
            logger.info(f"[{update.message.from_user.username}] {update.message.text}")
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
            logger.error(e)
            msg = f'failed - {e}'
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    return inner

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
        This is the callback whenever user clicks the button
    '''
    data = update.callback_query.data    
    command = data.split(" ")[0].replace('/', '')
    context.args = data.split(" ")[1:]    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=data)    
    await eval(command)(update, context)

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
    markup = InlineKeyboardMarkup(button_list)    
    await context.bot.send_message(chat_id=update.effective_chat.id, text='List of strategy:', reply_markup=markup)

@handler_expcetion
@handler_print
async def action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = context.args[0]
    button_list = []    
    button_list.append([InlineKeyboardButton(text='Back', callback_data=f'/start')]) 
    button_list.append([InlineKeyboardButton(text='Config', callback_data=f'/config {s}')])
    times = ['2 Hours Ago', '4 Hours Ago', '12 Hours Ago', '1 Days Ago', '5 Days Ago']
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
async def update(update: Update, context: ContextTypes.DEFAULT_TYPE):   
    '''
        Update strategy configuration
    '''

    strategy_id, key, value = context.args
    setup = ExecuteSetup(strategy_id)
    setup.update(key, value)
    config = setup.read()
    config = str(json.dumps(config, indent=4))        
    msg = 'succeeded\n'
    msg += config

    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

@handler_expcetion
@handler_print
async def pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):        
    '''
        Table summary of strategy pnl performance 
    '''        
    strategy_id = context.args[0]
    time = " ".join(context.args[1:])
    params = ExecuteSetup(strategy_id).read()

    strategy = GridArithmeticStrategy(**params)
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
    params = ExecuteSetup(strategy_id).read()

    strategy = GridArithmeticStrategy(**params)
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
    params = ExecuteSetup(strategy_id).read()

    strategy = GridArithmeticStrategy(**params)
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)
    strategy.load_data(time)

    with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as file:        
        strategy.summary(plot_orders=True, lastn=20, save_jpg_path=file.name, show_pnl_metrics=False)
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