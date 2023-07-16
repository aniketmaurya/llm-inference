# pip install discord.py
# Learn more here - https://github.com/aniketmaurya/docs-QnA-discord-bot/tree/main
import os

import discord
from dotenv import load_dotenv

from llm_chain import LitGPTConversationChain, LitGPTLLM
from llm_chain.templates import longchat_prompt_template
from llm_inference import prepare_weights

load_dotenv()

# path = prepare_weights("lmsys/longchat-7b-16k")
path = "checkpoints/lmsys/longchat-13b-16k"
llm = LitGPTLLM(checkpoint_dir=path, quantize="bnb.nf4")
llm("warm up!")
TOKEN = os.environ.get("DISCORD_BOT_TOKEN")


class MyClient(discord.Client):
    BOT_INSTANCE = {}

    def chat(self, user_id, query):
        if user_id in self.BOT_INSTANCE:
            return self.BOT_INSTANCE[user_id].send(query)

        self.BOT_INSTANCE[user_id] = LitGPTConversationChain.from_llm(
            llm=llm, prompt=longchat_prompt_template
        )
        return self.BOT_INSTANCE[user_id].send(query)

    bot = LitGPTConversationChain.from_llm(llm=llm, prompt=longchat_prompt_template)

    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        print(f"Message from {message.author}: {message.content}")

        if message.content.startswith("!help"):
            query = message.content.replace("!help", "")
            result = self.bot.send(query)
            await message.reply(result, mention_author=True)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(TOKEN)
