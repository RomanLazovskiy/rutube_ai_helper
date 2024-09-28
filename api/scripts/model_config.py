import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_community.llms import VLLMOpenAI

# Загрузка данных
df = pd.read_excel('data/knowledge_base.xlsx')
q = df['Вопрос из БЗ'].tolist()
a = df['Ответ из БЗ'].tolist()
docs = [Document(page_content=x, metadata=dict(answer=ans)) for x, ans in zip(q, a)]

embeddings = HuggingFaceBgeEmbeddings(
    model_name='jinaai/jina-embeddings-v3',
    model_kwargs=dict(trust_remote_code=True, device='cpu'),
    encode_kwargs=dict(task="text-matching"),
    query_instruction=''
)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs=dict(k=5))
retriever_bm25 = BM25Retriever.from_documents(docs)
retriever_bm25.k = 5

llm = VLLMOpenAI(
    model_name="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    openai_api_key="EMPTY",
    temperature=0,
    openai_api_base="http://176.123.167.65:8000/v1/",
    model_kwargs={"stop": ["."]}
)

prompt = PromptTemplate.from_template("""
Вы - ассистент с искусственным интеллектом, которому поручено переформулировать запросы пользователей для улучшения поиска в системе RAG.
Учитывая исходный запрос, перепишите его, чтобы он был более конкретным, подробным и позволял с большей вероятностью извлекать релевантную информацию. 
Исходный запрос: {question} 

Переписанный запрос:""")

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs=dict(k=3)),
    llm=llm,
    prompt=prompt
)

ensemble_retriever = EnsembleRetriever(retrievers=[retriever_bm25, retriever_from_llm], weights=[0.5, 0.5])

ce = CrossEncoder("api/models/cross_encoder", device='cpu')

# Настройка шаблона для llm
llm_template = '''Ты интеллектуальный помощник оператора службы поддержки видеохостинга RUTUBE. Твоя задача максимально точно и вежливо отвечать на запросы пользователя только о платформе. 
{examples}
Вопрос пользователя: {q}
Ответ:
'''

llm_template = PromptTemplate.from_template(llm_template)
chain = llm_template | llm
