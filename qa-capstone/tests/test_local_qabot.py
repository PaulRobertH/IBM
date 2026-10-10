import unittest
from unittest.mock import patch
import numpy as np
from langchain_core.documents import Document
import local_qabot as local


class LocalTests(unittest.TestCase):
    def test_overview_prefers_abstract_and_excludes_references(self):
        docs = [Document(page_content='[1] One [2] Two [3] Three [4] Four'), Document(page_content='Abstract: This paper proposes a chatbot.'), Document(page_content='Background details.')]
        class Embedder:
            def encode(self, text, **kwargs): return np.array([1.])
        with patch.object(local, 'embedding_model', return_value=Embedder()):
            selected = local.select_evidence({'chunks':docs, 'vectors':np.array([[1.],[0.8],[0.9]])}, 'What this paper is talking about?')
        self.assertEqual(selected[0], docs[1])
        self.assertNotIn(docs[0], selected)
        self.assertFalse(local.useful_answer('[1].'))
        self.assertFalse(local.useful_answer('Streamlit'))

    def test_validation(self):
        with self.assertRaises(ValueError): local.build_library([])
        with self.assertRaises(ValueError): local.answer_question(None, 'hello')

    def test_semantic_retrieval_and_generation(self):
        class Embedder:
            def encode(self, text, **kwargs):
                return np.array([1., 0.]) if isinstance(text, str) else np.array([[1., 0.], [0., 1.]])
        class Tokenizer:
            def encode(self, text, **kwargs): return list(text.encode())
            def decode(self, ids, **kwargs): return bytes(ids).decode()
            def build_inputs_with_special_tokens(self, ids): return ids + [0]
        class Model:
            def generate(self, **kwargs):
                self.prompt = bytes(kwargs['input_ids'][0].tolist()[:-1]).decode()
                return [list(b'The paper describes a chatbot.')]
        model = Model()
        docs = [Document(page_content='A chatbot paper.', metadata={'source':'paper.pdf','page':0}),
                Document(page_content='Other material.', metadata={'source':'other.txt'})]
        with patch.object(local, 'embedding_model', return_value=Embedder()), patch.object(local, 'language_model', return_value=(Tokenizer(), model)):
            with patch.object(local, 'document_loader', return_value=docs):
                library = local.build_library(['paper.pdf'])
            answer, rows = local.answer_question(library, 'What is this paper about?')
        self.assertEqual(answer, 'The paper describes a chatbot.')
        self.assertEqual(rows[0][0], 'paper.pdf, page 1')
        self.assertIn('A chatbot paper.', model.prompt)
        self.assertIn('What is this paper about?', model.prompt)


if __name__ == '__main__': unittest.main()

