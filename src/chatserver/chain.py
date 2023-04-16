"""Wrapper around Lightning App."""
import logging
from typing import List, Optional

import requests
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.llms.base import LLM
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def lit_chain(url: str):
    return load_chain(LightningChain(url=url))


def load_chain(llm):
    """Logic for loading the chain you want to use should go here."""

    input_key = "input"
    output_key = "response"
    memory = ConversationSummaryBufferMemory(
        llm=llm, output_key=output_key, input_key=input_key
    )
    chain = ConversationChain(
        llm=llm, verbose=True, memory=memory, output_key=output_key, input_key=input_key
    )
    return chain


class LightningChain(LLM, BaseModel):
    url: str = ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        if self.url == "":
            raise Exception("Server URL not set!")

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        assert isinstance(prompt, str)
        json_data = {"prompt": prompt}
        response = requests.post(
            url=self.url + "/predict", headers=headers, json=json_data
        )
        logger.error(response.raise_for_status())
        return response.json()["text"]

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Lightning"
