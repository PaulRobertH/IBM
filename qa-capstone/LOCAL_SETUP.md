# Run Document Compass without IBM credentials

In the Codespaces terminal, stop the old app with Ctrl+C. From the repository:

```bash
cd /workspaces/IBM
git pull origin main
cd qa-capstone
source .venv/bin/activate
python -m pip install -r requirements-local.txt
HOST=0.0.0.0 python local_qabot.py
```

If no virtual environment exists, create it with `python3 -m venv .venv` first. Python 3.11 or 3.12 is recommended for the local model dependencies. Keep the forwarded port 7860 private.

Open port 7860, upload your PDF, build the library and click Find an answer. The required question is prefilled. Initial model downloads and CPU inference may take several minutes. Leave the terminal running. Models are cached by Hugging Face, while document indexes stay in server memory.

This implementation uses [MiniLM semantic embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [FLAN-T5-small](https://huggingface.co/google/flan-t5-small), a small instruction model. It needs no IBM account, API key or GPU. Dependencies and model weights require internet access; document inference happens in your Codespace. Generated answers are brief and can be wrong. The evidence table shows the actual passages supplied to the model; passages may be shortened to fit its 512-token input budget. No confidence or correctness guarantee is implied.

This is an alternative working RAG app, not the course's watsonx embedding implementation. A screenshot should identify this local mode honestly. The assignment may require IBM-specific evidence for full marks.
