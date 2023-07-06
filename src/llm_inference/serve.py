from typing import Any

import lightning as L
from lightning.app.components import PythonServer
from pydantic import BaseModel

from llm_inference.model import LLMInference


class PromptRequest(BaseModel):
    prompt: str


class Response(BaseModel):
    result: str


class ServeLitGPT(PythonServer):
    def __init__(
        self,
        input_type,
        output_type,
        checkpoint_dir: str = None,
    ):
        super().__init__(input_type, output_type)
        self.checkpoint_dir = checkpoint_dir

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self._model = LLMInference(
            checkpoint_dir=self.checkpoint_dir,
        )

    def predict(self, request: PromptRequest) -> Any:
        result = self._model.chat(request.prompt)
        return Response(result=result)


if __name__ == "__main__":
    component = ServeLitGPT(
        input_type=PromptRequest,
        output_type=Response,
        checkpoint_dir="examples/chatbot/checkpoints/lmsys/longchat-7b-16k/",
    )
    app = L.LightningApp(component)
