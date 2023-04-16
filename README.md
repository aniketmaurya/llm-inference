# LLaMA Inference API 🦙

![project banner](./assets/llama-inference-api-min.png)

Inference API for LLaMA

```
pip install llama-inference

or

pip install git+https://github.com/aniketmaurya/llama-inference-api.git@main
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
from llama_inference.serve import ServeLLaMA, Response

import lightning as L

component = ServeLLaMA(input_type=PromptRequest, output_type=Response)
app = L.LightningApp(component)
```

```bash
lightning run app app.py
```
