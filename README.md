# LLaMA Inference API and Chatbot 🦙

![project banner](https://github.com/aniketmaurya/LLaMA-Inference-API/raw/main/assets/llama-inference-api-min.png)

Inference API for LLaMA

```
pip install llama-inference

or

pip install git+https://github.com/aniketmaurya/llama-inference-api.git@main
```

> **Note**: You need to manually install [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) and setup the model weights to use this project.

```
pip install lit-llama@git+https://github.com/Lightning-AI/lit-llama.git@main
```


## For Inference

```python
from llama_inference import LLaMAInference
import os

WEIGHTS_PATH = os.environ["WEIGHTS"]

checkpoint_path = f"{WEIGHTS_PATH}/lit-llama/7B/state_dict.pth"
tokenizer_path = f"{WEIGHTS_PATH}/lit-llama/tokenizer.model"

model = LLaMAInference(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, dtype="bfloat16")

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
