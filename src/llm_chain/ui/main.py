"""Python file to serve as the frontend"""

import logging

import rich
import streamlit as st
from streamlit_chat import message

from llm_chain import ServerChatBot

logger = logging.getLogger(__name__)


def run(lightning_app_state):
    if not lightning_app_state.llm_url:
        st.info("Waiting for server to get ready...")
        return

    print("lightning_app_state", lightning_app_state)

    if "model" not in st.session_state:
        # build unique conversational chain per session state
        bot = ServerChatBot(lightning_app_state.llm_url)
        st.session_state["model"] = bot
        logger.info("loaded model into state session")

    else:
        bot = st.session_state["model"]

    # From here down is all the StreamLit UI.
    st.set_page_config(page_title="LLaMA Demo", page_icon=":robot:")
    st.header("LLM Demo")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    def get_text():
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        rich.print("user input:", user_input)
        output = bot.predict(input=user_input)
        rich.print("buffer:", bot.memory.buffer)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
