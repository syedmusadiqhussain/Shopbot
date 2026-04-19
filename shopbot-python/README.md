# ShopBot (BeautyBox) — AI Customer Support Chatbot

ShopBot is a lightweight Flask + Ollama chatbot UI you can customize for any ecommerce client. It answers common questions (shipping, returns, hours, contact) instantly from configured store policies, and uses a local LLM (via Ollama) for richer conversations.

## Screenshot

![ShopBot UI](Screenshot%202026-04-20%20005944.png)

## Why this is client-ready

- Fast and reliable: policy questions respond immediately even if the AI model is busy.
- On-brand: store name, tagline, and policies come from a simple `STORE` dictionary.
- Simple stack: Flask + a single HTML file (no frontend frameworks).
- Local AI: runs on Ollama so you can demo without paid APIs.

## Quickstart (Windows)

1) Install Ollama: https://ollama.com  
2) Pull a free model:

```bash
ollama pull mistral
```

3) Install Python packages:

```bash
pip install -r requirements.txt
```

4) Run the app:

```bash
python app.py
```

Open: http://localhost:5000

## Customize for a new client

Edit [app.py](app.py) and update:
- `STORE` (name, shipping, returns, contact info)
- `SYSTEM_PROMPT` (brand voice + product category)

## Switch AI models

Download the model:

```bash
ollama pull llama3
```

Change one line in [app.py](app.py):

```python
MODEL = "llama3"
```

## API endpoints

- `GET /` → serves the UI
- `POST /chat` → chat endpoint (sends conversation history to backend)
- `GET /store-info` → store config used by the UI
- `GET /health` → Ollama status + available models
