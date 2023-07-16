import gradio as gr

from llm_chain import LitGPTConversationChain, LitGPTLLM
from llm_chain.templates import longchat_prompt_template
from llm_inference import prepare_weights

# path = prepare_weights("lmsys/longchat-13b-16k")
path = "checkpoints/lmsys/longchat-13b-16k"
llm = LitGPTLLM(checkpoint_dir=path)
bot = LitGPTConversationChain.from_llm(llm=llm, prompt=longchat_prompt_template)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = bot.send(message)
        chat_history.append((f"👤 {message}", f"🤖 {bot_message}"))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
