"""Wrapper around Lightning App."""
import logging

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from .base import ServerLLM

logger = logging.getLogger(__name__)


def build_server_chain(
    url: str, input_key: str = "input", output_key: str = "response"
):
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
