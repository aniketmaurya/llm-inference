# LLaMA Inference API 🦙

Inference API for LLaMA


**To use this library you must install `lit-llama`**

```
pip install llama-inference
```

or

```
pip install git+https://github.com/Lightning-AI/lit-llama.git@main
```

```python
from llama_inference import LLaMAInference
import os

WEIGHTS_PATH = os.environ["WEIGHTS"]

checkpoint_path = f"{WEIGHTS_PATH}/lit-llama/7B/state_dict.pth"
tokenizer_path = f"{WEIGHTS_PATH}/lit-llama/tokenizer.model"

model = LLaMAInference(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, dtype="bfloat16")

print(model("New York is located in"))
```