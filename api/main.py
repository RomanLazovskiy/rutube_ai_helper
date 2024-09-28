import asyncio
from fastapi import FastAPI
from scripts.api_models import Request, Response
from scripts.inference import get_answer
import uvicorn

app = FastAPI()


@app.get("/")
def index():
    return {"text": "Интеллектуальный помощник оператора службы поддержки."}


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    # Получаем ответ на вопрос
    answer = get_answer(query=request.question)

    # Возвращаем ответ
    return Response(
        answer=answer,
        class_1="Классификатор 1 уровня",
        class_2="Классификатор 2 уровня"
    )


# Запуск приложения
if __name__ == "__main__":
    host = "0.0.0.0"
    config = uvicorn.Config(app, host=host, port=8001)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
