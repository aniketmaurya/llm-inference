# Large Language Model (LLM) Inference API and Chatbot 🦙

![project banner](https://github.com/aniketmaurya/llm-inference/raw/main/assets/llm-inference-min.png)

Inference API for LLMs like LLaMA and Falcon powered by Lit-GPT from [Lightning AI](https://lightning.ai)

```
pip install llm-inference
```

### Install from main branch
```bash
pip install git+https://github.com/aniketmaurya/llm-inference.git@main
```

> **Note**: You need to manually install [Lit-GPT](https://github.com/Lightning-AI/lit-gpt) and setup the model weights to use this project.

```
pip install lit_gpt@git+https://github.com/aniketmaurya/install-lit-gpt.git@install
```

## For Inference

```python
from llm_inference import LLMInference, prepare_weights
from rich import print

path = prepare_weights("EleutherAI/pythia-70m")
model = LLMInference(checkpoint_dir=path)

print(model("New York is located in"))
```


## How to use the Chatbot


<div style="overflow:hidden;margin-left:auto;margin-right:auto;border-radius:10px;width:100%;max-width:682px;position:relative"><div style="width:100%;padding-bottom:64.22287390029325%"></div><iframe width="682" height="438" title="" src="https://snappify.com/embed/4d90490d-da09-4648-a8e6-fd9fb4e37aaa?responsive=1" allow="clipboard-write" allowfullscreen="" style="background:#89CFFDFF;position:absolute;left:0;top:0;width:100%" frameborder="0"></iframe></div>



```python
from llm_chain import LitGPTConversationChain, LitGPTLLM
from llm_inference import prepare_weights
from rich import print


path = str(prepare_weights("meta-llama/Llama-2-7b-chat-hf"))
llm = LitGPTLLM(checkpoint_dir=path, quantize="bnb.nf4")  # 7GB GPU memory
bot = LitGPTConversationChain.from_llm(llm=llm, prompt=llama2_prompt_template)

print(bot.send("hi, what is the capital of France?"))
```

## Launch Chatbot App

<video width="320" height="240" controls>
  <source src="/assets/chatbot-demo.mov" type="video/mp4">
</video>

**1. Download weights**
```py
from llm_inference import prepare_weights
path = prepare_weights("meta-llama/Llama-2-7b-chat-hf")
```

**2. Launch Gradio App**

```
python examples/chatbot/gradio_demo.py
```
