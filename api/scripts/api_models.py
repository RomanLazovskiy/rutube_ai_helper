from pydantic import BaseModel

class Request(BaseModel):
    question: str

class Response(BaseModel):
    answer: str
    class_1: str = "N/A"
    class_2: str = "N/A"
    hallucination: bool = False
