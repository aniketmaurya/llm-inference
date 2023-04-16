"""Wrapper around Lightning App."""
import logging
from typing import List, Optional

import requests
from langchain.llms.base import LLM
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
