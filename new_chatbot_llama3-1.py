
pip install ipywidget.chatbot

pip install ipywidgets requests

import json, os, requests, ipywidgets as widgets
from IPython.display import display

# --- Configure Ollama ---
BASE_URL = "http://localhost:11434"       # Ollama default
MODEL    = "llama3.1:latest"              # or your exact tag, e.g. "llama3.1:8b-instruct"

# --- Context controls ---
MAX_TURNS = 30        # keep the last N user+assistant exchanges (plus system)
SESSION_FILE = "ollama_session.json"  # persist across notebook restarts

# --- Load / init history ---
def load_history():
    if os.path.exists(SESSION_FILE):
        try:
            return json.load(open(SESSION_FILE, "r"))
        except Exception:
            pass
    # fresh start with a system message (tweak tone/persona as you like)
    return [{"role":"system","content":"You are a concise, helpful assistant."}]

def save_history(history):
    try:
        json.dump(history, open(SESSION_FILE, "w"))
    except Exception:
        pass

def trim_history(history, max_turns=MAX_TURNS):
    """
    Keep system + last N (user,assistant) turns.
    """
    sys = [m for m in history if m["role"]=="system"][:1]
    rest = [m for m in history if m["role"]!="system"]
    # each turn is 2 messages (user, assistant); keep last 2*MAX_TURNS
    keep = rest[-2*max_turns:]
    return sys + keep

history = load_history()

# --- UI ---
chat_log   = widgets.Output(layout={'border':'1px solid gray','height':'300px','overflow_y':'auto'})
user_input = widgets.Textarea(placeholder="Type your message here...", layout={'width':'100%','height':'70px'})
send_btn   = widgets.Button(description="Send", button_style='primary')
reset_btn  = widgets.Button(description="Reset memory", button_style='warning')

def send_message(_):
    global history
    user_text = user_input.value.strip()
    if not user_text:
        return
    user_input.value = ""

    # append user msg
    history.append({"role":"user","content":user_text})

    # trim before sending (protect context window)
    hist_to_send = trim_history(history)

    with chat_log:
        print(f"You: {user_text}")

    try:
        # Ollama chat API: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
        r = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": hist_to_send,
                "stream": False
            },
            timeout=300
        )
        r.raise_for_status()
        j = r.json()
        reply = j.get("message",{}).get("content") or j.get("response") or "(no reply)"
    except Exception as e:
        reply = f"Error: {e}"

    # append assistant msg & save/persist
    history.append({"role":"assistant","content":reply})
    history = trim_history(history)
    save_history(history)

    with chat_log:
        print(f"FouFlou:{reply}\n")

def reset_memory(_):
    global history
    history = [{"role":"system","content":"You are a concise, helpful assistant."}]
    save_history(history)
    with chat_log:
        print("— memory cleared —\n")

send_btn.on_click(send_message)
reset_btn.on_click(reset_memory)
display(chat_log, user_input, widgets.HBox([send_btn, reset_btn]))

