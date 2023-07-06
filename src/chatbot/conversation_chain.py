"""Wrapper around Lightning App."""
import logging
from collections import deque
from typing import Optional, Union

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from .llm import DummyLLM, LitGPTLLM, ServerLLM

logger = logging.getLogger(__name__)


class LitGPTConversationChain(ConversationChain):
    @staticmethod
    def from_llm(
        llm: Union[str, LitGPTLLM],
        memory: Optional[None] = None,
        input_key="input",
        output_key="response",
        verbose=False,
        url: Optional[str] = None,
    ):
        if llm == "dummy":
            llm = DummyLLM()

        if llm == "server":
            llm = ServerLLM(url=url)

        if not memory:
            memory = ConversationSummaryBufferMemory(
                llm=llm, output_key=output_key, input_key=input_key
            )
        chain = LitGPTConversationChain(
            llm=llm,
            verbose=verbose,
            memory=memory,
            output_key=output_key,
            input_key=input_key,
        )
        return chain

    @staticmethod
    def from_lit_gpt(
        checkpoint_dir: str,
        precision: str = "bf16-mixed",
        quantize: Optional[str] = None,
        accelerator: str = "auto",
        input_key="input",
        output_key="response",
        verbose=False,
    ):
        llm = LitGPTLLM(
            checkpoint_dir=checkpoint_dir,
            precision=precision,
            quantize=quantize,
            accelerator=accelerator,
        )
        return LitGPTConversationChain.from_llm(
            llm=llm,
            input_key=input_key,
            output_key=output_key,
            verbose=verbose,
        )

    def send(self, prompt: str, **kwargs):
        return self(prompt)["response"]

    @property
    def history(self):
        return self.memory.buffer

    def clear(self):
        self.memory.clear()


def build_server_chain(
    url: str, input_key: str = "input", output_key: str = "response"
) -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""

    logger.info(f"Initializing ServerLLM using url: {url}")

    llm = ServerLLM(url=url)

    memory = ConversationSummaryBufferMemory(
        llm=llm, output_key=output_key, input_key=input_key
    )
    chain = ConversationChain(
        llm=llm, verbose=True, memory=memory, output_key=output_key, input_key=input_key
    )
    logger.info("Created Conversational Chain")
    return chain
