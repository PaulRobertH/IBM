"""Local semantic RAG with MiniLM embeddings and FLAN-T5; no IBM account."""
import os
import logging
from functools import lru_cache
from threading import Lock

from qabot import document_loader, source_label
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')


@lru_cache(maxsize=1)
def language_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


def build_library(files):
    if not files or len(files) > 10:
        raise ValueError('Upload between one and ten documents.')
    documents = [d for file in files for d in document_loader(file)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(documents)
    if len(chunks) > 2000:
        raise ValueError('Upload smaller documents (maximum 2,000 chunks).')
    with MODEL_LOCK:
        vectors = embedding_model().encode([d.page_content for d in chunks], normalize_embeddings=True)
    return {'chunks': chunks, 'vectors': vectors}


def answer_question(library, question):
    import numpy as np
    import torch
    if library is None:
        raise ValueError('Build the document library first.')
    question = (question or '').strip()
    if not question or len(question) > 500:
        raise ValueError('Enter a question of 1–500 characters.')
    with MODEL_LOCK:
        query_vector = embedding_model().encode(question, normalize_embeddings=True)
        scores = library['vectors'] @ query_vector
        positions = np.argsort(-scores)[:3]
        docs = [library['chunks'][int(i)] for i in positions]
        tokenizer, model = language_model()
        # Reserve space for the question and instructions; avoid silently truncating them.
        prefix = f'Answer the question using the context. If it is not supported, say you do not know. Question: {question}\nContext: '
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        budget = 510 - len(prefix_ids)
        if budget < 50:
            raise ValueError('Shorten the question to leave room for document evidence.')
        context_ids = []
        used = []
        for doc in docs:
            ids = tokenizer.encode(doc.page_content + '\n', add_special_tokens=False)
            available = budget - len(context_ids)
            if available <= 0:
                break
            selected = ids[:available]
            context_ids.extend(selected)
            used.append((doc, tokenizer.decode(selected, skip_special_tokens=True)))
        inputs = torch.tensor([tokenizer.build_inputs_with_special_tokens(prefix_ids + context_ids)])
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    max_new_tokens=180, do_sample=False)
        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    rows = [[source_label(doc), text] for doc, text in used]
    return answer or 'The model did not produce an answer.', rows


def build_ui():
    import gradio as gr
    with gr.Blocks(title='Document Compass · Local AI') as app:
        gr.Markdown('# 🧭 Document Compass · Local AI\nPDF questions answered without an API key.')
        gr.Markdown('MiniLM finds relevant passages; FLAN-T5 generates a short answer. First use downloads the models. Small-model answers can be inaccurate—check the evidence. This mode does not use watsonx.')
        state = gr.State(None)
        files = gr.File(label='Upload PDF / TXT / Markdown', file_count='multiple',
                        file_types=['.pdf', '.txt', '.md'], type='filepath')
        build = gr.Button('Build document library', variant='primary')
        status = gr.Textbox(label='Library status', interactive=False)
        question = gr.Textbox(label='Your question', value='What this paper is talking about?')
        ask = gr.Button('Find an answer', variant='primary')
        answer = gr.Textbox(label='Generated answer', lines=5)
        sources = gr.Dataframe(headers=['Source / page', 'Evidence supplied to model'], interactive=False)
        def prepare(paths):
            try:
                library = build_library(paths)
                return library, f'Ready: {len(library["chunks"])} chunks', '', []
            except ValueError as error:
                raise gr.Error(str(error)) from None
            except Exception:
                logging.exception('Local library build failed')
                raise gr.Error('Could not load files or download the embedding model. Check the terminal and internet connection.') from None
        def respond(library, query):
            try:
                return answer_question(library, query)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            except Exception:
                logging.exception('Local answer generation failed')
                raise gr.Error('Model execution failed. Check the terminal, free memory and model download connection.') from None
        build.click(prepare, files, [state, status, answer, sources])
        ask.click(respond, [state, question], [answer, sources])
        question.submit(respond, [state, question], [answer, sources])
    return app


if __name__ == '__main__':
    build_ui().queue().launch(server_name=os.getenv('HOST', '127.0.0.1'),
                              server_port=int(os.getenv('PORT', '7860')), share=False)

