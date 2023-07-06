from chatbot.conversation_chain import LitGPTConversationChain


def test_dummybot():
    bot = LitGPTConversationChain.from_llm("dummy")
    prompt = "Hello, I am testing you!"
    response = bot.send(prompt)
    assert isinstance(response, str)
    assert response == "Hi, I am a helpful chatbot!"
