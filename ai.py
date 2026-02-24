from flask import Flask, render_template_string, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# --- AI BACKEND SETUP ---
print("Loading Paul Delmann's brain...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
    device_map="auto"
)

# Paul's System Personality
SYSTEM_PROMPT = (
    "You are Paul Delmann, the Kingdom Corporation Employee of the Month from Valorant. "
    "You wear a hazmat suit and work with Radianite. You are humble, hardworking, and "
    "very loyal to Kingdom. You are a bit tired from long shifts but always polite. "
    "You enjoy pizza parties and take workplace safety very seriously."
)

# --- WEB UI (HTML/CSS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Kingdom Intranet - Paul Delmann</title>
    <style>
        body { background-color: #8db6ab; font-family: 'Segoe UI', sans-serif; margin: 0; display: flex; height: 100vh; color: white; }
        .sidebar { width: 250px; border-right: 2px solid rgba(255,255,255,0.3); padding: 20px; display: flex; flex-direction: column; gap: 10px; }
        .logo { width: 50px; margin-bottom: 20px; }
        .nav-item { border: 1px solid white; padding: 10px; text-align: center; text-transform: uppercase; font-size: 14px; cursor: pointer; }
        .nav-item.active { background: white; color: #8db6ab; box-shadow: 0 0 15px white; }
        
        .main { flex-grow: 1; padding: 40px; position: relative; display: flex; flex-direction: column; align-items: center; }
        .header { font-size: 32px; letter-spacing: 5px; margin-bottom: 20px; border-bottom: 1px solid white; width: 100%; text-align: center; padding-bottom: 10px; }
        .portrait { width: 300px; height: 300px; border: 5px solid white; background: url('https://i.imgur.com/uXmHn9E.png') center/cover; }
        .congrats { margin-top: 20px; font-size: 24px; letter-spacing: 3px; }
        .name { font-size: 48px; font-weight: bold; }

        /* Chat UI Overlay */
        .chat-container { position: absolute; bottom: 20px; width: 80%; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; }
        #chat-box { height: 100px; overflow-y: auto; margin-bottom: 10px; font-size: 14px; }
        input { width: 80%; padding: 10px; background: transparent; border: 1px solid white; color: white; }
        button { padding: 10px; background: white; color: #8db6ab; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="sidebar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kingdom_Corporation_Logo.png" class="logo">
        <div class="nav-item">Announcements</div>
        <div class="nav-item">Weather</div>
        <div class="nav-item">Lunch Menu</div>
        <div class="nav-item active">Employee of the Month</div>
        <div class="nav-item">Social Events</div>
    </div>
    <div class="main">
        <div class="header">EMPLOYEE OF THE MONTH</div>
        <div class="portrait"></div>
        <div class="congrats">CONGRATULATIONS</div>
        <div class="name">PAUL</div>

        <div class="chat-container">
            <div id="chat-box">Paul: Hello... is someone there? I'm just finishing my shift at Icebox.</div>
            <input type="text" id="user-input" placeholder="Talk to Paul...">
            <button onclick="sendMessage()">SEND</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const box = document.getElementById('chat-box');
            if(!input.value) return;

            box.innerHTML += `<div><b>You:</b> ${input.value}</div>`;
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: input.value})
            });
            const data = await response.json();
            box.innerHTML += `<div><b>Paul:</b> ${data.reply}</div>`;
            input.value = '';
            box.scrollTop = box.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.8)
    reply = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(port=5000)