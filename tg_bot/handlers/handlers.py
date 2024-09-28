import aiohttp
from telegram import Update
from telegram.ext import ContextTypes


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greeting = "Добрый день! Это интеллектуальный помощник RUTUBE от команды Уральские мандарины. Какой у вас вопрос?"
    await update.message.reply_text(greeting)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    # Адрес вашего vLLM-сервера
    url = "http://176.123.167.65:8000/v1/chat/completions"

    # Запрос на инференс модели
    payload = {
        "model": "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
        "messages": [{"role": "user", "content": user_message}]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    assistant_reply = response_data['choices'][0]['message']['content']
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
