# Run Document Compass without IBM credentials

In the Codespaces terminal, stop the old app with Ctrl+C. From the repository:

```bash
cd /workspaces/IBM
git pull origin main
cd qa-capstone
python3.12 -m venv .venv-local
./.venv-local/bin/python -m pip install -r requirements-local.txt
HOST=0.0.0.0 ./.venv-local/bin/python local_qabot.py
```

If no virtual environment exists, create it with `python3 -m venv .venv` first. Python 3.11 or 3.12 is recommended for the local model dependencies. Keep the forwarded port 7860 private.

Open port 7860, upload your PDF, build the library and click Find an answer. The required question is prefilled. Initial model downloads and CPU inference may take several minutes. Leave the terminal running. Models are cached by Hugging Face, while document indexes stay in server memory.

This implementation uses [MiniLM semantic embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [FLAN-T5-base](https://huggingface.co/google/flan-t5-base), a small instruction model. It needs no IBM account, API key or GPU. Dependencies and model weights require internet access; document inference happens in your Codespace. Generated answers are brief and can be wrong. The evidence table shows the actual passages supplied to the model; passages may be shortened to fit its 512-token input budget. No confidence or correctness guarantee is implied.

This is an alternative working RAG app, not the course's watsonx embedding implementation. A screenshot should identify this local mode honestly. The assignment may require IBM-specific evidence for full marks.

The updated version prefers abstracts/conclusions for summary questions, excludes bibliography-heavy chunks and shares the input budget across retrieved passages. FLAN-T5-base uses more memory and downloads a larger model than the earlier small version. Set LOCAL_QA_MODEL=google/flan-t5-small to opt back into the smaller model. Tests check retrieval selection and generation wiring with stubbed models; improved real model answer quality still needs testing in your Codespace.

