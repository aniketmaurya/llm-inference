"""Python file to serve as the frontend"""
import rich
import streamlit as st
from streamlit_chat import message

from chatserver.chain import lit_chain


def run(lightning_app_state):
    if not lightning_app_state.llm_url:
        st.info("Waiting for server to get ready... :clock:")
        return

    print("lightning_app_state", lightning_app_state)

    if "model" not in st.session_state:
        chain = lit_chain(lightning_app_state.llm_url)
        st.session_state["model"] = chain

    else:
        chain = st.session_state["model"]

    # From here down is all the StreamLit UI.
    st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
    st.header("ChatBot Demo")

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
        output = chain.predict(input=user_input)
        rich.print("buffer:", chain.memory.buffer)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
