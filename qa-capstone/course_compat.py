"""Optional original-course RetrievalQA adapter; the app uses modern runnables."""
from qabot import CLOUD, get_llm


def course_retriever_qa(index, query):
    from langchain_classic.chains import RetrievalQA
    if index.mode != CLOUD:
        raise ValueError('The original course QA chain requires IBM mode.')
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(), chain_type='stuff',
        retriever=index.store.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
    )
    return chain.invoke({'query': query})
