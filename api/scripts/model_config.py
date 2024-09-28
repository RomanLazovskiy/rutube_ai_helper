from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_community.llms import VLLMOpenAI
from transformers import pipeline
import pandas as pd

df_knowledge_base = pd.read_excel('data/knowledge_base.xlsx')[["Вопрос из БЗ", "Ответ из БЗ"]].drop_duplicates()
df_cases = pd.read_excel('data/real_cases.xlsx')[["Вопрос пользователя", "Ответ из БЗ"]].drop_duplicates()

docs_knowledge = [Document(page_content=x, metadata=dict(answer=ans)) for x, ans in zip(df_knowledge_base['Вопрос из БЗ'], df_knowledge_base['Ответ из БЗ'])]
docs_cases = [Document(page_content=x, metadata=dict(answer=ans)) for x, ans in zip(df_cases['Вопрос пользователя'], df_cases['Ответ из БЗ'])]
all_docs = docs_knowledge + docs_cases

embeddings = HuggingFaceBgeEmbeddings(
    model_name='jinaai/jina-embeddings-v3',
    model_kwargs=dict(trust_remote_code=True, device='cpu'),
    encode_kwargs=dict(task="text-matching"),
    query_instruction=''
)
vector_store = FAISS.from_documents(docs_knowledge, embeddings)

retriever_bm25 = BM25Retriever.from_documents(all_docs)
retriever_bm25.k = 5

llm = VLLMOpenAI(
    model_name="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    openai_api_key="EMPTY",
    temperature=0,
    openai_api_base="http://176.123.167.65:8000/v1/",
    model_kwargs={"stop": ["."]}
)

re_query_prompt = PromptTemplate.from_template("""
Вы - ассистент с искусственным интеллектом, которому поручено переформулировать запросы пользователей для улучшения поиска в системе RAG.
Учитывая исходный запрос, перепишите его, чтобы он был более конкретным, подробным и позволял с большей вероятностью извлекать релевантную информацию.
Исходный запрос: {question}

Переписанный запрос:
""")

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs=dict(k=5)),
    llm=llm,
    prompt=re_query_prompt
)

ensemble_retriever = EnsembleRetriever(retrievers=[retriever_bm25, retriever_from_llm], weights=[0.5, 0.5])

cross_encoder = CrossEncoder("./models/cross_encoder", device='cpu')

classif_lvl1 = pipeline("text-classification", model="./models/flan-t5-v2-lvl1")
classif_lvl2 = pipeline("text-classification", model="./models/flan-t5-class2-v3")
