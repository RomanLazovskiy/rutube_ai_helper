import aiohttp
import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Start", "Info", "Help"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    greeting = "Добрый день! Это интеллектуальный помощник RUTUBE от команды Уральские мандарины. Какой у вас вопрос?"
    await update.message.reply_text(greeting, reply_markup=reply_markup)

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info_message = (
        "Я интеллектуальный помощник оператора службы поддержки видеохостинга RUTUBE. "
        "Моя задача — помогать вам, отвечая на ваши вопросы о платформе RUTUBE. "
        "Если у вас возникнут вопросы по работе с видеохостингом, я постараюсь предоставить вам всю необходимую информацию."
    )
    await update.message.reply_text(info_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "Список доступных команд:\n"
        "/start — Начать работу с ботом\n"
        "/info — Информация о боте и его возможностях\n"
        "/help — Показать список команд"
    )
    await update.message.reply_text(help_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_message = update.message.text

    logger.info(f"Вопрос пользователя {user.username} ({user.id}): {user_message}")

    if user_message.lower() == "start":
        await start(update, context)
        return
    elif user_message.lower() == "info":
        await info(update, context)
        return
    elif user_message.lower() == "help":
        await help_command(update, context)
        return

    url = "http://176.123.167.65:8001/predict"
    payload = {"question": user_message}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    assistant_reply = response_data['answer']
                    logger.info(f"Ответ бота пользователю {user.username} ({user.id}): {assistant_reply}")
                    await update.message.reply_text(assistant_reply)
                else:
                    error_message = f"Ошибка при запросе: {await resp.text()}"
                    await update.message.reply_text(error_message)
    except aiohttp.ClientConnectionError:
        error_message = "Ошибка соединения. Сервер недоступен по IP или порту."
        await update.message.reply_text(error_message)
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")
