"""Course-shaped PDF QA interface, using genuine IBM services."""
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

from qabot import get_llm, watsonx_embedding, vector_database


# Task 1: PDF document loader
def document_loader(file):
    path = Path(file)
    if path.suffix.lower() != '.pdf' or not path.is_file():
        raise ValueError('Upload a PDF document.')
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Use a PDF smaller than 20 MB.')
    loaded_document = PyPDFLoader(str(path)).load()
    loaded_document = [d for d in loaded_document if d.page_content.strip()]
    if not loaded_document:
        raise ValueError('The PDF has no readable text. Scanned pages require OCR.')
    return loaded_document


# Task 2: text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len,
    )
    return splitter.split_documents(data)


# Tasks 3 and 4 are shared with qabot.py:
# watsonx_embedding() returns WatsonxEmbeddings.
# vector_database(chunks) calls Chroma.from_documents(chunks, embedding_model).


# Task 5: complete file-to-retriever pipeline
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    if len(chunks) > 2000:
        raise ValueError('Use a smaller PDF (maximum 2,000 chunks).')
    vectordb = vector_database(chunks)
    return vectordb.as_retriever(search_type='similarity', search_kwargs={'k': 4})


# Task 6: original course chain and function signature
def retriever_qa(file, query):
    if not file or not query or not query.strip():
        raise ValueError('Upload a PDF and enter a question.')
    if len(query) > 2000:
        raise ValueError('Use a question shorter than 2,000 characters.')
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever=retriever_obj,
        return_source_documents=True,
    )
    response = qa.invoke({'query': query.strip()})
    return response['result']


def build_ui():
    import gradio as gr
    def answer(file, query):
        try:
            return retriever_qa(file, query)
        except Exception:
            raise gr.Error('Check your PDF, question, IBM credentials and model availability. No answer was generated.') from None
    return gr.Interface(
        fn=answer,
        inputs=[gr.File(label='Upload PDF File', file_count='single',
                        file_types=['.pdf'], type='filepath'),
                gr.Textbox(label='Input Query', lines=2,
                           value='What this paper is talking about?')],
        outputs=gr.Textbox(label='Answer', lines=12),
        title='Document Compass · Course QA Bot',
        description='Upload a research PDF and ask a question using IBM watsonx and LangChain RetrievalQA. Each submission embeds the PDF again and may consume service allowance.',
    )


if __name__ == '__main__':
    build_ui().queue().launch(server_name=os.getenv('HOST', '127.0.0.1'),
                              server_port=int(os.getenv('PORT', '7860')), share=False)

