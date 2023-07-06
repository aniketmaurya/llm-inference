import random
import time

import gradio as gr
from langchain.prompts import PromptTemplate

from chatbot import LitGPTConversationChain, LitGPTLLM
from chatbot.templates import longchat_template

path = "checkpoints/lmsys/longchat-7b-16k"
llm = LitGPTLLM(checkpoint_dir=path)
bot = LitGPTConversationChain.from_llm(llm=llm)
prompt = PromptTemplate(
    input_variables=["input", "history"], template=longchat_template
)
bot.prompt = prompt


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = bot.send(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
