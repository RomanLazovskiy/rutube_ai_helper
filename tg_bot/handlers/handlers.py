import aiohttp
from telegram import Update
from telegram.ext import ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greeting = "Добрый день! Это интеллектуальный помощник RUTUBE от команды Уральские мандарины. Какой у вас вопрос?"
    await update.message.reply_text(greeting)


# Функция для обработки сообщений и отправки запроса на ваш API /predict
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    url = "http://176.123.167.65:8001/predict"

    payload = {
        "question": user_message
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    assistant_reply = response_data['answer']
                    await update.message.reply_text(assistant_reply)
                else:
                    error_message = f"Ошибка при запросе: {await resp.text()}"
                    await update.message.reply_text(error_message)
    except aiohttp.ClientConnectionError:
        error_message = "Ошибка соединения. Сервер недоступен по IP или порту."
        await update.message.reply_text(error_message)
    except Exception as e:
        error_message = f"Произошла ошибка: {e}"
        await update.message.reply_text(error_message)
