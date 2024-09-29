import logging
import asyncio
from fastapi import FastAPI
from scripts.api_models import Request, Response
from scripts.inference import AiHelper
from scripts.model_config import llm, ensemble_retriever, cross_encoder
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ai_helper = AiHelper(llm=llm, retriever=ensemble_retriever, cross_encoder=cross_encoder)

app = FastAPI()

@app.get("/")
def index():
    return {"text": "Интеллектуальный помощник оператора службы поддержки."}

@app.post("/predict", response_model=Response)
async def predict(request: Request):
    logger.info(f"Получен вопрос пользователя: {request.question}")
    result = ai_helper(request.question)
    classification_result = ai_helper.inference_classification(request.question)

    logger.info(f"Ответ бота: {result}, Классификация: class_1={classification_result['class_1']}, class_2={classification_result['class_2']}")

    return Response(
        answer=result,
        class_1=classification_result['class_1'],
        class_2=classification_result['class_2']
    )

if __name__ == "__main__":
    host = "0.0.0.0"
    config = uvicorn.Config(app, host=host, port=8001)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
