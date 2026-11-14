# Gemma 4 31B — Chainlit UI

A chat interface for **Google Gemma 4 31B** via **NVIDIA NIM**, with streaming and extended thinking support built on [Chainlit](https://chainlit.io).

---

## Features

| Feature | Details |
|---|---|
| 🧠 Extended Thinking | Shows Gemma's reasoning in a collapsible step |
| 🌊 Streaming | Token-by-token streaming response |
| 🖼️ Image Upload | Attach images for multimodal queries |
| 💬 Multi-turn | Full conversation history per session |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your API key via environment variable

The key is already embedded in `app.py`, but you can override it:

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

> If you use env var, update `app.py` line:
> ```python
> API_KEY = os.environ.get("NVIDIA_API_KEY", "your-fallback-key")
> ```

### 3. Run the app

```bash
chainlit run app.py -w
```

The `-w` flag enables **auto-reload** on file changes.

### 4. Open in browser

```
http://localhost:8000
```

---

## Project Structure

```
.
├── app.py              # Main Chainlit application
├── requirements.txt    # Python dependencies
├── .chainlit/
│   └── config.toml    # UI configuration & branding
└── README.md
```

---

## How It Works

1. User sends a message (optionally with images).
2. The app builds a multimodal message payload and calls the NVIDIA NIM API with `stream=True`.
3. Streamed tokens are parsed — `<think>…</think>` blocks are shown in a collapsible **Thinking** step; the rest renders as the main reply.
4. The full conversation history is maintained per session for multi-turn dialogue.
