import gradio as gr

from llm_chain import LitGPTConversationChain, LitGPTLLM
from llm_chain.templates import llama2_prompt_template
from llm_inference import prepare_weights


path = str(prepare_weights("meta-llama/Llama-2-7b-chat-hf"))
llm = LitGPTLLM(checkpoint_dir=path, quantize="bnb.nf4")
llm("warmup")
bot = LitGPTConversationChain.from_llm(llm=llm, prompt=llama2_prompt_template)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    clear.click(fn=bot.clear)

    def respond(message, chat_history):
        bot_message = bot.send(message)
        chat_history.append((f"👤 {message}", f"{bot_message}"))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
