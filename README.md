# Large Language Model (LLM) Inference API and Chatbot 🦙

![project banner](https://github.com/aniketmaurya/llm-inference/raw/main/assets/llama-inference-api-min.png)

Inference API for LLMs like LLaMA and Falcon powered by Lit-GPT from [Lightning AI](https://lightning.ai)

```
pip install llm-inference
```

### Install from main branch
```bash
pip install git+https://github.com/aniketmaurya/llm-inference.git@main
```

## For Inference

```python
from llm_inference import LLMInference, prepare_weights
from rich import print

path = prepare_weights("EleutherAI/pythia-70m")
model = LLMInference(checkpoint_dir=path)

print(model("New York is located in"))
```


## For deploying as a REST API

Create a Python file `app.py` and initialize the `ServeLLaMA` App.

```python
# app.py
from llm_inference.serve import ServeLLaMA, Response, PromptRequest

import lightning as L

component = ServeLLaMA(input_type=PromptRequest, output_type=Response)
app = L.LightningApp(component)
```

```bash
lightning run app app.py
```

## How to use the Chatbot

```python
from chatbot import LitGPTChatBot

checkpoint_dir = "weights"

bot = LitGPTChatBot(
    checkpoint_dir=checkpoint_dir)

print(bot.send("hi, what is the capital of France?"))
```
