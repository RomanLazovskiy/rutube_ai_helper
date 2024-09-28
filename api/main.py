import asyncio
from fastapi import FastAPI
from scripts.api_models import Request, Response
from scripts.inference import AiHelper
import uvicorn

ai_helper = AiHelper()

app = FastAPI()


@app.get("/")
def index():
    return {"text": "Интеллектуальный помощник оператора службы поддержки."}


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    response = ai_helper(request.question)
    return response


if __name__ == "__main__":
    host = "0.0.0.0"
    config = uvicorn.Config(app, host=host, port=8001)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
