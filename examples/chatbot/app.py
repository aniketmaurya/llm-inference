import lightning as L
import lightning.app.frontend as frontend

from chatserver.components import LLMServe
from llama_inference.serve import ServeLLaMA, Response, PromptRequest
from chatserver.ui import ui_render_fn


class ChatBotApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # self.llm_serve = LLMServe(
        #     model_id="google/flan-ul2", cloud_compute=L.CloudCompute("gpu")
        # )
        self.llm_serve = ServeLLaMA(input_type=PromptRequest, output_type=Response)
        self.llm_url = ""

    def run(self):
        self.llm_serve.run()
        if self.llm_serve.url:
            self.llm_url = self.llm_serve.url

    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=ui_render_fn)


app = L.LightningApp(ChatBotApp())
