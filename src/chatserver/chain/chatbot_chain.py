from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from chatserver.lightning_client import LightningChain


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
