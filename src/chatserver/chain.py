"""Wrapper around Lightning App."""
import logging
from collections import deque

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from .base import DummyLLM, ServerLLM

logger = logging.getLogger(__name__)


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


class BaseChatBot:
    def __init__(self, llm, input_key="input", output_key="response") -> None:
        memory = ConversationSummaryBufferMemory(
            llm=llm, output_key=output_key, input_key=input_key
        )
        self.chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            output_key=output_key,
            input_key=input_key,
        )

    def send(self, msg: str) -> str:
        return self.chain.predict(input=msg)

    @property
    def history(self):
        return self.chain.memory.buffer

    def clear(self):
        self.chain.memory.clear()


class DummyChatBot(BaseChatBot):
    def __init__(self) -> None:
        super().__init__(llm=DummyLLM())
