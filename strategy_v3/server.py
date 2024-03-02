from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from utils.credentials import TELEGRAM_BOT_API_KEY
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.Strategy import GridArithmeticStrategy
from tabulate import tabulate
import json
import warnings
warnings.filterwarnings('ignore')

def handler_expcetion(func):
    '''
        Handle Exception for each handler to response back the error message to users
    '''
    async def inner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await func(update=update, context=context)
        except Exception as e:
            msg = f'failed - {e}'
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    return inner

@handler_expcetion
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):            
    '''
        Print out all active strategy
    '''
    config = ExecuteSetup.read_all()
    strategy = list(config.keys())
    msg = "List of strategy:\n"
    msg += "\n".join([f"/{s}" for s in strategy])                
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

@handler_expcetion
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
async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):        
    '''
        Table summary of strategy performance
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
    msg = tabulate(table, headers='keys', tablefmt='psql')    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()        
    application.add_handler(CommandHandler('start', start)) 
    application.add_handler(CommandHandler('config', config))   
    application.add_handler(CommandHandler('update', update))   
    application.add_handler(CommandHandler('performance', performance))  

    application.run_polling()    