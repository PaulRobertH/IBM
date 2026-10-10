import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import qabot


class QAIntegrationTests(unittest.TestCase):
    def test_load_split_retrieve_and_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policies.txt'
            path.write_text('Email policy: encrypt confidential attachments.\n\n' +
                            'Smoking is prohibited inside offices. ' * 80, encoding='utf-8')
            index = qabot.build_index([str(path)])
            self.assertGreater(len(index.chunks), 1)
            self.assertTrue(all(len(d.page_content) <= 1000 for d in index.chunks))
            answer, sources = qabot.retriever_qa(index, 'email attachments', 1)
            self.assertIn('encrypt', answer)
            self.assertEqual(sources[0][1], 'policies.txt')
            answer, sources = qabot.retriever_qa(index, 'xylophone unicorn')
            self.assertEqual(sources, [])
            self.assertIn('could not find', answer)

    def test_library_isolation(self):
        left = qabot.Index([Document(page_content='alpha policy')], qabot.DEMO)
        right = qabot.Index([Document(page_content='beta policy')], qabot.DEMO)
        self.assertEqual(qabot.retriever(right, 'alpha'), [])
        self.assertEqual(len(qabot.retriever(left, 'alpha')), 1)

    def test_pdf_page_metadata(self):
        from pypdf import PdfWriter
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper.pdf'
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(path)
            with self.assertRaisesRegex(ValueError, 'no readable text'):
                qabot.document_loader(path)
        self.assertEqual(qabot.source_label(Document(page_content='a', metadata={'source':'paper.pdf','page':0})), 'paper.pdf, page 1')

    def test_validation(self):
        with self.assertRaises(ValueError): qabot.build_index([])
        with self.assertRaises(ValueError): qabot.retriever_qa(None, 'hello')
        with self.assertRaises(ValueError): qabot.retriever_qa(qabot.Index([], qabot.DEMO), '')

    def test_cloud_chain_with_stub_services(self):
        docs = [Document(page_content='The experiment used 40 samples.', metadata={'source':'paper.pdf','page':2})]
        class Store:
            def as_retriever(self, **kwargs):
                return RunnableLambda(lambda question: docs)
        seen = []
        def generate(prompt):
            seen.append(prompt.to_string())
            return 'The experiment used 40 samples [1].'
        with patch.object(qabot, 'get_llm', return_value=RunnableLambda(generate)):
            answer, rows = qabot.retriever_qa(qabot.Index(docs, qabot.CLOUD, Store()), 'How many samples?')
        self.assertIn('40 samples', answer)
        self.assertIn('40 samples', seen[0])
        self.assertIn('How many samples?', seen[0])
        self.assertEqual(rows[0][1], 'paper.pdf, page 3')


if __name__ == '__main__':
    unittest.main()
