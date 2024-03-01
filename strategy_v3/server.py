from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from utils.credentials import TELEGRAM_BOT_API_KEY

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global x
    x = x + 1
    print(x)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=str(x))

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)    
    application.run_polling()    