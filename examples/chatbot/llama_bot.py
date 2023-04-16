import lightning as L
import lightning.app.frontend as frontend

from chatserver.ui import ui_render_fn
from llama_inference.serve import PromptRequest, Response, ServeLLaMA


checkpoint_path = "weights/state_dict.pth"
tokenizer_path = "weights/tokenizer.model"


class ChatBotApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.llm_serve = ServeLLaMA(
            input_type=PromptRequest,
            output_type=Response,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )  
        self.llm_url = ""

    def run(self):
        
        self.llm_serve.run()
        if self.llm_serve.url:
            self.llm_url = self.llm_serve.url

    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=ui_render_fn)


app = L.LightningApp(ChatBotApp())
