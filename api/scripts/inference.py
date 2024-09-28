from .model_config import ensemble_retriever, llm, cross_encoder, classif_lvl1, classif_lvl2
from .api_models import Request, Response
from langchain.prompts import ChatPromptTemplate, PromptTemplate

class HallucinationDetector:
    """
    Класс для определения галлюцинаций в ответах.
    True - галлюцинация
    False - корректный ответ
    """
    def __init__(self, llm):
        system = """Пожалуйста, проанализируйте сгенерированный ответ на основе входного запроса и релевантных документов. 
        Ваша задача — определить, соответствует ли ответ содержанию документов и запросу, или содержит ли он галлюцинации. 
        Если ответ является галлюцинацией, отвечай "да", в противном случае отвечай "нет"."""
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation> \n\n"),
        ])
        self.hallucination_grader = hallucination_prompt | llm

    @staticmethod
    def format_docs(docs):
        return "\n".join(f"<doc{i+1}>:\nQuestion:{doc.page_content}\nAnswer:{doc.metadata['answer']}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

    def detect(self, model_response, docs_to_use):
        response_hall = self.hallucination_grader.invoke({"documents": self.format_docs(docs_to_use), "generation": model_response})
        return True if 'да' in response_hall.lower() else False

class AiHelper:
    def __init__(self):
        self.retriever = ensemble_retriever
        self.cross_encoder = cross_encoder
        self.hallucination_detector = HallucinationDetector(llm)
        llm_template = """
        Ты интеллектуальный помощник оператора службы поддержки видеохостинга RUTUBE. Твоя задача максимально точно и вежливо отвечать на запросы пользователя только о платформе.
        {examples}
        Вопрос пользователя: {query}
        Ответ:
        """
        llm_template = PromptTemplate.from_template(llm_template)
        self.answer_generation_llm = llm_template | llm

    def get_relevant_docs(self, query: str):
        return self.retriever.invoke(query)

    def rank_documents(self, query: str, relevant_docs: list):
        ranking_docs = self.cross_encoder.rank(query, [x.page_content for x in relevant_docs])
        top_relevant_docs = [relevant_docs[x['corpus_id']] for x in ranking_docs[:5]]
        return top_relevant_docs

    def create_examples(self, relevant_docs):
        examples = ""
        for x in relevant_docs:
            q = x.page_content
            a = x.metadata["answer"]
            examples += f"Вопрос: {q}\nОтвет: {a}\n\n"
        return examples

    def get_answer(self, query: str):
        relevant_docs = self.get_relevant_docs(query)
        top_ranking_docs = self.rank_documents(query, relevant_docs)
        examples = self.create_examples(top_ranking_docs)
        model_response = self.answer_generation_llm.invoke({"query": query, "examples": examples})
        hallucination_detected = self.hallucination_detector.detect(model_response, top_ranking_docs)

        # Получение классов
        class_1 = classif_lvl1(query)[0]['label']
        class_2 = classif_lvl2(query)[0]['label']

        return Response(
            answer=model_response,
            class_1=class_1,
            class_2=class_2,
            hallucination=hallucination_detected
        )

    def __call__(self, query: str):
        return self.get_answer(query)
