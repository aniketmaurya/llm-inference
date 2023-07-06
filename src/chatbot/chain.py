"""Wrapper around Lightning App."""
import logging
from collections import deque

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from .base import DummyLLM, LitGPTLLM, ServerLLM

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
    def __init__(
        self, llm, input_key="input", output_key="response", verbose=False
    ) -> None:
        memory = ConversationSummaryBufferMemory(
            llm=llm, output_key=output_key, input_key=input_key
        )
        self.chain = ConversationChain(
            llm=llm,
            verbose=verbose,
            memory=memory,
            output_key=output_key,
            input_key=input_key,
        )

    def send(self, msg: str) -> str:
        return self.chain.predict(input=msg)

    def predict(self, input: str) -> str:
        return self.chain.predict(input=input)

    @property
    def history(self):
        return self.chain.memory.buffer

    def clear(self):
        self.chain.memory.clear()


class DummyChatBot(BaseChatBot):
    def __init__(self, verbose=True) -> None:
        super().__init__(llm=DummyLLM(), verbose=verbose)


class ServerChatBot(BaseChatBot):
    def __init__(
        self, url: str, input_key="input", output_key="response", verbose=False
    ) -> None:
        llm = ServerLLM(url=url)
        super().__init__(llm, input_key, output_key, verbose)


class LitGPTChatBot(BaseChatBot):
    def __init__(self, checkpoint_dir: str, verbose=False) -> None:
        llm = LitGPTLLM(checkpoint_dir=checkpoint_dir)
        super().__init__(llm=llm, verbose=verbose)
