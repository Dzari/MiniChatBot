import gradio as gr
from .backends import create_backend
from .chat import ChatSession

backend = create_backend(backend="hf", model_name="distilgpt2", device="auto")
session = ChatSession(backend)

def respond(message, chat_history):
    reply = session.ask(message)
    chat_history = chat_history + [(message, reply)]
    return chat_history, ""

with gr.Blocks(title="Mini ChatGPT") as demo:
    gr.Markdown("## Mini ChatGPT (HF backend)")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type a message...", show_label=False)
    btn = gr.Button("Send")
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    btn.click(respond, [msg, chatbot], [chatbot, msg])
    def reset():
        global session
        session = ChatSession(backend)
        return [], ""
    clear.click(reset, None, [chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch()
