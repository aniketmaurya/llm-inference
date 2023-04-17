from typing import Any

import lightning as L
from lightning.app.components import PythonServer
from pydantic import BaseModel

from llama_inference.model import LLaMAInference


class PromptRequest(BaseModel):
    prompt: str


class Response(BaseModel):
    result: str


class ServeLLaMA(PythonServer):
    def __init__(
        self, input_type, output_type, checkpoint_path: str=None, tokenizer_path: str=None
    ):
        super().__init__(input_type, output_type)
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self._model = LLaMAInference(
            checkpoint_path=self.checkpoint_path, tokenizer_path=self.tokenizer_path, dtype="bfloat16"
        )

    def predict(self, request: PromptRequest) -> Any:
        result = self._model(request.prompt)
        return Response(result=result)


if __name__ == "__main__":
    component = ServeLLaMA(input_type=PromptRequest, output_type=Response,)
    app = L.LightningApp(component)
