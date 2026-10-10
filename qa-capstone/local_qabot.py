"""Local semantic RAG with MiniLM embeddings and FLAN-T5; no IBM account."""
import os
import logging
import re
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
    name = os.getenv('LOCAL_QA_MODEL', 'google/flan-t5-base')
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


def is_overview(question):
    return bool(re.search(r'\b(summary|summarize|summarise|overview)\b|paper.*(talking about|about)|document.*about', question.lower()))


def select_evidence(library, question):
    """Blend semantic relevance with overview intent; exclude bibliography-heavy chunks."""
    import numpy as np
    overview = is_overview(question)
    search = 'research paper abstract main contribution proposed system methodology conclusion' if overview else question
    vector = embedding_model().encode(search, normalize_embeddings=True)
    scores = library['vectors'] @ vector
    ranked = []
    for i, doc in enumerate(library['chunks']):
        text = doc.page_content
        # Bibliographies often contain multiple numbered references or URLs.
        if len(re.findall(r'\[\d+\]', text)) >= 4 or len(re.findall(r'https?://', text)) >= 2:
            continue
        score = float(scores[i])
        if overview:
            if re.search(r'\b(abstract|conclusion)\b', text, re.I): score += 0.3
            if re.search(r'this paper|we propose|proposed system', text, re.I): score += 0.15
        ranked.append((score, i, doc))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected = []
    for _, _, doc in ranked:
        words = set(re.findall(r'\w+', doc.page_content.lower()))
        if any(len(words & old)/max(1, len(words | old)) > 0.75 for _, old in selected):
            continue
        selected.append((doc, words))
        if len(selected) == 5: break
    return [doc for doc, _ in selected]


def pack_context(docs, tokenizer, budget):
    """Share the token budget across evidence rather than letting one chunk consume it."""
    used = []
    encoded = []
    separator = tokenizer.encode('\n', add_special_tokens=False)
    per_doc = max(1, budget // max(1, len(docs)) - len(separator))
    for doc in docs:
        text = re.sub(r'\s+', ' ', doc.page_content).strip()
        ids = tokenizer.encode(text, add_special_tokens=False)[:per_doc]
        encoded.extend(ids)
        used.append((doc, tokenizer.decode(ids, skip_special_tokens=True)))
        if len(encoded) + len(separator) <= budget:
            encoded.extend(separator)
    return encoded[:budget], used


def useful_answer(answer):
    cleaned = re.sub(r'\[\d+\]', '', answer).strip()
    return len(re.findall(r'\w+', cleaned)) >= 5


def answer_question(library, question):
    import numpy as np
    import torch
    if library is None:
        raise ValueError('Build the document library first.')
    question = (question or '').strip()
    if not question or len(question) > 500:
        raise ValueError('Enter a question of 1–500 characters.')
    with MODEL_LOCK:
        docs = select_evidence(library, question)
        if not docs:
            return 'No usable evidence was found outside the references. Try a more specific question.', []
        tokenizer, model = language_model()
        # Reserve space for the question and instructions; avoid silently truncating them.
        task = ('Summarize the main topic, proposed system and technologies of the paper in three sentences.'
                if is_overview(question) else 'Answer every part of the question in complete sentences.')
        prefix = f'{task} Use only the context. If information is missing, say so. Question: {question}\nContext: '
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        budget = 510 - len(prefix_ids)
        if budget < 50:
            raise ValueError('Shorten the question to leave room for document evidence.')
        context_ids, used = pack_context(docs, tokenizer, budget)
        inputs = torch.tensor([tokenizer.build_inputs_with_special_tokens(prefix_ids + context_ids)])
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    max_new_tokens=180, do_sample=False, num_beams=4,
                                    repetition_penalty=1.1)
        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    rows = [[source_label(doc), text] for doc, text in used]
    if not useful_answer(answer):
        answer = ('The model returned an incomplete answer. Retrieved evidence is shown below; '
                  'try a more specific question. Raw model output: ' + repr(answer))
    return answer, rows


def build_ui():
    import gradio as gr
    with gr.Blocks(title='Document Compass · Local AI') as app:
        gr.Markdown('# 🧭 Document Compass · Local AI\nPDF questions answered without an API key.')
        gr.Markdown('MiniLM finds relevant passages; FLAN-T5-base generates an answer. First use downloads the models. Answers can be inaccurate—check the evidence. This mode does not use watsonx.')
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

