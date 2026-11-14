import chainlit as cl
import requests
import base64
import json
import re

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "[REDACTED]"
MODEL = "google/gemma-4-31b-it"


def read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def encode_uploaded_file(file_path: str, mime_type: str) -> dict:
    """Encode a file to base64 for multimodal input."""
    b64 = read_b64(file_path)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64}"
        }
    }


def build_content(text: str, files: list = None) -> list | str:
    """Build message content, optionally with image attachments."""
    if not files:
        return text

    content = []
    for f in files:
        if f.mime and f.mime.startswith("image/"):
            content.append(encode_uploaded_file(f.path, f.mime))

    content.append({"type": "text", "text": text})
    return content


def stream_nvidia_response(messages: list):
    """Generator that streams token chunks from NVIDIA API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 1.00,
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    with requests.post(INVOKE_URL, headers=headers, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    continue


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "👋 **Welcome! I'm powered by Google Gemma 4 31B via NVIDIA NIM.**\n\n"
            "💡 I have **extended thinking** enabled — I reason deeply before responding.\n"
            "🖼️ You can **attach images** to your messages for multimodal queries.\n\n"
            "Ask me anything!"
        ),
        author="System",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    history: list = cl.user_session.get("history", [])

    # Build user content (text + optional images)
    user_content = build_content(message.content, message.elements)
    history.append({"role": "user", "content": user_content})

    # Thinking step — shown while model reasons
    thinking_buffer = []
    response_buffer = []
    in_thinking = False

    async with cl.Step(name="🧠 Gemma is thinking…", show_input=False) as think_step:
        think_text = ""

        msg = cl.Message(content="", author="Gemma 4 31B")
        await msg.send()

        full_stream = ""
        try:
            for token in stream_nvidia_response(history):
                full_stream += token

                # Detect <think>...</think> blocks inline
                if "<think>" in full_stream and not in_thinking:
                    in_thinking = True

                if in_thinking:
                    if "</think>" in full_stream:
                        # Extract thinking content
                        match = re.search(r"<think>(.*?)</think>", full_stream, re.DOTALL)
                        if match:
                            think_text = match.group(1).strip()
                            await think_step.stream_token(think_text)
                        in_thinking = False
                        # Remainder after </think> is real response
                        remainder = full_stream.split("</think>", 1)[-1].lstrip("\n")
                        if remainder:
                            await msg.stream_token(remainder)
                    # Still inside <think> block — buffer silently
                else:
                    # Check if token itself contains leftover think tag noise
                    clean = re.sub(r"</?think>", "", token)
                    if clean:
                        await msg.stream_token(clean)

        except requests.HTTPError as e:
            await msg.stream_token(f"\n\n❌ **API error:** {e}")

        await msg.update()

        # Close thinking step with summary
        if think_text:
            think_step.output = think_text[:500] + ("…" if len(think_text) > 500 else "")
        else:
            think_step.output = "No explicit thinking block returned."

    # Save clean assistant reply to history
    clean_reply = re.sub(r"<think>.*?</think>", "", full_stream, flags=re.DOTALL).strip()
    history.append({"role": "assistant", "content": clean_reply})
    cl.user_session.set("history", history)


@cl.on_chat_end
async def on_chat_end():
    cl.user_session.set("history", [])
