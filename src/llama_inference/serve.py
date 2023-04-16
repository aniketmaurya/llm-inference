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
    def setup(self, *args: Any, **kwargs: Any) -> None:
        self._model = LLaMAInference(*args, **kwargs)

    def predict(self, request: PromptRequest) -> Any:
        result = self._model(request.prompt)
        return Response(result=result)


if __name__ == "__main__":
    component = ServeLLaMA(input_type=PromptRequest, output_type=Response)
    app = L.LightningApp(component)
