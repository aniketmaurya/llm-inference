from chatbot.chain import DummyChatBot


def test_dummybot():
    bot = DummyChatBot()
    prompt = "testing dummy bot"
    response = bot.send(prompt)
    assert isinstance(response, str)
    assert f"Bot: {prompt}" in response
