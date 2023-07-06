import gradio as gr
from chatbot import LitGPTConversationChain, LitGPTLLM
from llm_inference import prepare_weights
from rich import print

from langchain.prompts import PromptTemplate
from chatbot.templates import longchat_template


path = "checkpoints/lmsys/longchat-7b-16k"
llm = LitGPTLLM(checkpoint_dir=path)
bot = LitGPTConversationChain.from_llm(llm=llm)
prompt = PromptTemplate(input_variables =["input", "history"], template=longchat_template)
bot.prompt = prompt
 
def chat(prompt):
    return bot.send(prompt)
    
with gr.Blocks() as demo: 
    name = gr.Textbox(label="Type your msg") 
    output = gr.Textbox(label="Reply") 
    greet_btn = gr.Button("Send") 
    greet_btn.click(fn=chat, inputs=name, outputs=output, api_name="chat") 
    
demo.launch()
