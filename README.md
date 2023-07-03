# Large Language Model (LLM) Inference API and Chatbot 🦙

![project banner](https://github.com/aniketmaurya/LLaMA-Inference-API/raw/main/assets/llama-inference-api-min.png)

Inference API for LLaMA

```
pip install llama-inference

# to use chatbot
pip install llama-inference[chatbot]
```

### Install from main branch
```bash
pip install git+https://github.com/aniketmaurya/llama-inference-api.git@main
```

> **Note**: You need to manually install [Lit-GPT](https://github.com/Lightning-AI/lit-gpt) and setup the model weights to use this project.

```
pip install lit-gpt@git+https://github.com/Lightning-AI/lit-gpt.git@main
```


## For Inference

```python
from llama_inference import LLMInference
import os

WEIGHTS_PATH = os.environ["WEIGHTS"]

checkpoint_dir = f"checkpoints/tiiuae/falcon-7b"

model = LLMInference(checkpoint_dir=checkpoint_dir, precision="bf16-true")

print(model("New York is located in"))
```


## For deploying as a REST API

Create a Python file `app.py` and initialize the `ServeLLaMA` App.

```python
# app.py
from llama_inference.serve import ServeLLaMA, Response, PromptRequest

import lightning as L

component = ServeLLaMA(input_type=PromptRequest, output_type=Response)
app = L.LightningApp(component)
```

```bash
lightning run app app.py
```

## How to use the Chatbot

```python
from chatbot import LLaMAChatBot

checkpoint_path = f"../../weights/state_dict.pth"
tokenizer_path = f"../../weights/tokenizer.model"

bot = LLaMAChatBot(
    checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path
)

print(bot.send("hi, what is the capital of France?"))
```
