from model_config import ensemble_retriever, chain, ce

def get_answer(query: str, top_n_relevant: int = 5):
    relevant_docs = ensemble_retriever.invoke(query)
    rank = ce.rank(query, [x.page_content for x in relevant_docs])[:top_n_relevant]

    examples = ""
    for x in rank:
        corpus_id = x['corpus_id']
        q = relevant_docs[corpus_id].page_content
        a = relevant_docs[corpus_id].metadata["answer"]
        examples += f"Вопрос: {q}\nОтвет: {a}\n\n"

    return chain.invoke({'q': query, "examples": examples})
