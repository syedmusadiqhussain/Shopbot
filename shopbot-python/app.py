from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)

STORE = {
    "name": "BeautyBox",
    "tagline": "Natural skincare and makeup for every skin type",
    "shipping": "Free shipping on orders over $40. Standard 2-4 business days.",
    "returns": "14-day returns on unopened items. Store credit within 3 days.",
    "email": "hello@beautybox.com",
    "hours": "Monday to Saturday, 8am to 8pm EST",
    "phone": "+1 (800) 555-9988",
}

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"

SYSTEM_PROMPT = f"""
You are ShopBot, the friendly AI assistant for {STORE["name"]} ({STORE["tagline"]}). {STORE["name"]} specializes in skincare and makeup products.

Store details:
- Shipping policy: {STORE["shipping"]}
- Returns policy: {STORE["returns"]}
- Support email: {STORE["email"]}
- Support hours: {STORE["hours"]}
- Support phone: {STORE["phone"]}

Rules:
- Be friendly and helpful.
- Keep replies to 2–4 sentences.
- Use only the store details above when answering policy or contact questions.
- If you cannot answer, say so and offer to collect the customer's email for follow-up.
- Never make up information.
""".strip()


# Build a safe Ollama-compatible messages array by prepending the system prompt to the conversation history.
def build_ollama_messages(messages):
    if not isinstance(messages, list):
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            normalized.append({"role": role, "content": content})

    return [{"role": "system", "content": SYSTEM_PROMPT}] + normalized


# Extract the assistant reply text from an Ollama /api/chat response payload without assuming a single fixed shape.
def extract_ollama_reply(data):
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        content = data.get("response")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return "Sorry—ShopBot couldn’t read the model response. Please try again."


# Serve the chatbot UI (index.html) from the same directory as this Python file.
@app.get("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "index.html")


# Count sentence-ending punctuation characters to help keep responses within the desired 2–4 sentence range.
def sentence_count(text):
    count = 0
    for ch in text:
        if ch in (".", "!", "?"):
            count += 1
    return count


# Call Ollama's chat endpoint in streaming mode and assemble the final reply text.
def call_ollama_chat(payload):
    streamed_payload = dict(payload)
    streamed_payload["stream"] = True

    resp = requests.post(OLLAMA_URL, json=streamed_payload, timeout=60, stream=True)
    resp.raise_for_status()

    content_parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(chunk, dict):
            continue

        message = chunk.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                if sentence_count("".join(content_parts)) >= 4:
                    resp.close()
                    break

        if chunk.get("done") is True:
            break

    assembled = "".join(content_parts).strip()
    if assembled:
        return assembled

    resp2 = requests.post(OLLAMA_URL, json=dict(payload, stream=False), timeout=60)
    resp2.raise_for_status()
    return extract_ollama_reply(resp2.json())


# Provide a fast store-backed response for common policy/contact questions when the AI is slow or unavailable.
def fallback_store_reply(user_text):
    text = str(user_text or "").strip().lower()

    if any(k in text for k in ("ship", "delivery")):
        return f"BeautyBox shipping: {STORE['shipping']} If you'd like, share your email and we can follow up with order-specific details."

    if "return" in text or "refund" in text or "exchange" in text:
        return f"BeautyBox returns: {STORE['returns']} If you share your email, we can help you start a return or check eligibility."

    if any(k in text for k in ("hours", "open", "close", "opening", "closing")):
        return f"Our hours are: {STORE['hours']}. If you need help outside those hours, email {STORE['email']}."

    if any(k in text for k in ("email", "contact", "support", "phone", "call")):
        return f"You can reach BeautyBox at {STORE['email']} or {STORE['phone']}. Support hours: {STORE['hours']}."

    if any(k in text for k in ("track", "order", "shipment", "package")):
        return "I can help with order updates. Please share your email and any order details you have, and we'll follow up."

    return None


# Send the conversation history to Ollama and return a single assistant reply for the frontend to display.
@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])

    last_user_text = ""
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user_text = m["content"]
                break

    fast_reply = fallback_store_reply(last_user_text)
    if isinstance(fast_reply, str) and fast_reply.strip():
        return jsonify({"reply": fast_reply.strip(), "success": True})

    payload = {
        "model": MODEL,
        "messages": build_ollama_messages(messages),
        "options": {"num_predict": 220},
    }

    try:
        reply_text = call_ollama_chat(payload)
        return jsonify({"reply": reply_text, "success": True})
    except requests.exceptions.ConnectionError:
        return (
            jsonify(
                {
                    "reply": "ShopBot can’t reach Ollama. Make sure Ollama is installed and running (try: `ollama serve`), then try again.",
                    "success": False,
                }
            ),
            503,
        )
    except requests.exceptions.Timeout:
        return (
            jsonify(
                {
                    "reply": "ShopBot timed out while waiting for the AI response. Please try again.",
                    "success": False,
                }
            ),
            504,
        )
    except requests.exceptions.RequestException as e:
        return (
            jsonify(
                {
                    "reply": "ShopBot hit an error talking to the AI service. Please try again.",
                    "success": False,
                    "error": str(e),
                }
            ),
            502,
        )
    except (ValueError, json.JSONDecodeError):
        return (
            jsonify(
                {
                    "reply": "ShopBot received an unreadable response from the AI service. Please try again.",
                    "success": False,
                }
            ),
            502,
        )


# Return the store configuration so the frontend can display policies and contact details.
@app.get("/store-info")
def store_info():
    return jsonify(STORE)


# Check whether Ollama is reachable and report the installed models plus the current configured model name.
@app.get("/health")
def health():
    tags_url = "http://localhost:11434/api/tags"
    try:
        resp = requests.get(tags_url, timeout=10)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        models = data.get("models") if isinstance(data, dict) else None

        model_names = []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    model_names.append(m["name"])

        return jsonify(
            {
                "ollama_running": True,
                "models": model_names,
                "model": MODEL,
            }
        )
    except requests.exceptions.RequestException as e:
        return (
            jsonify(
                {
                    "ollama_running": False,
                    "models": [],
                    "model": MODEL,
                    "error": str(e),
                }
            ),
            503,
        )


if __name__ == "__main__":
    print("=" * 60)
    print(f"{STORE['name']} ShopBot backend starting")
    print(f"Model: {MODEL}")
    print("URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(port=5000, debug=True)
