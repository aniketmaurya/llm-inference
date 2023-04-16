import logging
from typing import Any, Optional

import lightning as L
from lightning.app.components import PythonServer, Text
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "google/flan-T5-base"


def load_lit_llama(checkpoint_path, tokenizer_path):
    from llama_inference.model import LLaMAInference

    llama = LLaMAInference(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, dtype="bfloat16")
    return llama

def load_hf_llm(model_id: str):
    from langchain.llms import HuggingFacePipeline
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return HuggingFacePipeline(pipeline=pipe)


class PromptSchema(BaseModel):
    # prompt: str = Field(title="Your msg to chatbot", max_length=300, min_length=1)
    prompt: str


class LLMServe(PythonServer):
    def __init__(self, model_id: Optional[str] = None, **kwargs):
        super().__init__(input_type=PromptSchema, output_type=Text, **kwargs)
        self.model_id = model_id or _DEFAULT_MODEL_ID

    def setup(self, *args, **kwargs) -> None:
        self._model = load_hf_llm(self.model_id)

    def predict(self, request: PromptSchema) -> Any:
        return {"text": self._model(request.prompt)}


if __name__ == "main":
    app = L.LightningApp(LLMServe())
