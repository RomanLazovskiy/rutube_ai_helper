import os
import logging
import time
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from handlers.handlers import start, handle_message, info, help_command

def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не установлена")
        exit(1)

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    application = ApplicationBuilder().token(TOKEN).build()
    application.bot.set_my_commands([
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("info", "Информация о боте и его возможностях"),
        BotCommand("help", "Показать список команд")
    ])

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('info', info))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    while True:
        try:
            logger.info("Запуск Polling...")
            application.run_polling(allowed_updates=Update.ALL_TYPES, timeout=10)
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")
            logger.info("Перезапуск Polling через 5 секунд...")
            time.sleep(5)

if __name__ == '__main__':
    main()
