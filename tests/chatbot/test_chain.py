from chatbot.conversation_chain import DummyChatBot


def test_dummybot():
    bot = DummyChatBot()
    prompt = "testing dummy bot"
    response = bot.send(prompt)
    assert isinstance(response, str)
    assert prompt in response
