from .model_config import ensemble_retriever, llm, cross_encoder, classif_lvl1, classif_lvl2
from .api_models import Request, Response
from langchain.prompts import ChatPromptTemplate, PromptTemplate

class HallucinationDetector:
    '''
    Детектор галюцинаций:ё
    True - галюцинация
    False - верный ответ
    '''
    def __init__(self, llm):
        self.llm = llm
        system = """Пожалуйста, проанализируйте сгенерированный ответ на основе входного запроса и релевантных документов. Ваша задача — определить, соответствует ли ответ содержанию документов и запросу, или содержит ли он галлюцинации (т.е. неверную или вымышленную информацию, не основанную на документах). Если релевантных документов нет и ответ гласит, что вы не знаете ответа на этот вопрос, это считается корректным и не является галлюцинацией. Если ответ является галюцинацией, то отвечай "да", в противном случае отвечай "нет". Отвечай только либо "да", либо "нет". Не добавляй ничего лишнего. Это очень важная работа, от которой зависит моя работа.""" 
        hallucination_prompt = ChatPromptTemplate.from_messages( 
            [ 
                ("system", system), 
                ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation> \n\n"), 
            ] 
        )
        self.hallucination_grader = hallucination_prompt | llm 
    
    @staticmethod
    def format_docs(docs): 
        return "\n".join(f"<doc{i+1}>:\nQuestion:{doc.page_content}\nAnswer:{doc.metadata['answer']}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

    def detect(self, model_respoce, docs_to_use):
        response_hall = self.hallucination_grader.invoke({"documents": self.format_docs(docs_to_use), "generation": model_respoce}) 
        return True if 'да' in response_hall.lower() else False

class AiHelper:

    def __init__(self, llm, retriever, cross_encoder, top_n_relevant:int=5):
        self.retriever = retriever
        self.cross_encoder = cross_encoder
        self.top_n_relevant = top_n_relevant
        self.hallucination_detector = HallucinationDetector(llm)
        llm_template = '''Ты интеллектуальный помощник оператора службы поддержки видеохостинга RUTUBE. Твоя задача максимально точно и вежливо отвечать на запросы пользователя только о платформе. Если ты будешь хорошо показывать результаты, то мы заплатим тебе 200 долларов.
Для того чтобы ты лучше отвечал на заданый вопрос, мы также предоставляем тебе некоторые идельные вопросы и ответы из нашей базы знаний:
{examples}
Недобавляй ничего лишнего, используй только предоставленные вопросы и ответы, ничего не выдумывай, не выдавай никакой информации из базы знаний и даже не упоминай о ней. Если среди предоставленных вопросов и ответов нет релевантных, то ответь что не знаешь ответа на предоставленный вопрос. Будь вежливым и дружелюбным.
Вопрос пользователя: {query}
Ответ:

'''
        llm_template = PromptTemplate.from_template(llm_template)
        self.answer_generation_llm = llm_template | llm

    def retrive(self, query:str):
        
        return self.retriever.invoke(query)

    def get_relevat_docs(self, query:str, relevant_docs:list):
        
        ranking_docs = self.cross_encoder.rank(query, [x.page_content for x in relevant_docs])
        top_relevant_docs = [relevant_docs[x['corpus_id']] for x in ranking_docs[:self.top_n_relevant]]
        return top_relevant_docs

    def create_examples_for_template(self, relevant_docs):
        examples = ""
    
        for x in relevant_docs:
            q = x.page_content
            a = x.metadata["answer"]
            examples += f"Вопрос: {q}\nОтвет: {a}\n\n"
        
        return examples
        
    def get_answer(self, query:str):
    
        relevant_docs = self.retrive(query)
        
        top_raking_docs = self.get_relevat_docs(query, relevant_docs)
        
        examples = self.create_examples_for_template(top_raking_docs)

        model_respoce = self.answer_generation_llm.invoke({"query": query,
                                                  "examples": examples})
        
        if not self.hallucination_detector.detect(model_respoce=model_respoce,
                                                   docs_to_use=top_raking_docs):
            return model_respoce
        else:
            return "К сожалению, я не могу немедленно ответить на ваш вопрос. Уточняю детали для точного ответа на ваш запрос. \nМы свяжемся с вами как можно скорее с подробным ответом. Извините за ожидание!"

    def __call__(self, query:str):
        return self.get_answer(query)
