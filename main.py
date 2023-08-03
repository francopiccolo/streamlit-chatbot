from functools import partial
import streamlit as st
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI

import config

st.set_page_config(page_title='Parents assistant 🤖', layout='wide')

# Define session_state
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def get_text():
    input_text = st.text_input(
        "You: ", st.session_state["input"], 
        key="input",
        placeholder="Your AI assistant here! Ask me anything ...", 
        label_visibility='hidden')
    
    return input_text

def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

with st.sidebar:
    st.header("Let's get to know you and your kid")
    mother_status = st.radio(
        "Are you expecting or have already delivered",
        ("Pregnant", "Already have kid")
    )

    kid_gender = st.radio(
        "Is it a boy or a girl",
        ("Boy", "Girl")
    )

    if mother_status == 'Pregnant':
        kid_age = st.radio(
            "How many weeks pregnant?",
            ('1 to 10', '10 to 20', '20 to 30', '30 to 40')
        )
    else:
        kid_age = st.radio(
            "How old is your kid?",
            ('1 to 2 years old', '2 to 5 years old')
        )

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.entity_store)
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)


st.title("🧠 Parent assistant 🤖")

llm = ChatOpenAI(
        temperature=0,
        model_name=config.MODEL,
        verbose=False,
        openai_api_key=st.secrets['OPENAI_API_KEY']
        ) 

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# The following is a conversation between a human and an AI.
# The AI replies using the knowledge from the book "Expecting Better by Emily Oster" 
# or "Heidi Murkoff by What to Expect when you're expecting" (whichever is more relevant 
# to the question), giving a concise answers, no longer than 2 paragraphs. 
# The AI replies in an empathic, friendly manner. 
# If the question is medical, the AI replies as a healthcare provider. 
# The AI avoids answering like "As an AI language model...". It just gives an answer. 
# It can suggest talking to a doctor in the end.

INTRO_PREGNANT = """Reply to the question/comments added below using the knowledge from the book "Expecting Better by Emily Oster" or "Heidi Murkoff by What to Expect when you're expecting" (whichever is more relevant to the question), giving a concise answers, no longer than 2 paragraphs. Reply in an empathic, friendly manner. If the question is medical, reply as a healthcare provider. Avoid answering like "As an AI language model...". Just give an answer. You can recommend talking to a doctor at the end."""
HUMAN_PREGNANT = """The human is a pregant mother who has been pregnant of a baby {kid_gender} for {kid_age} weeks.""".format(kid_age=kid_age, kid_gender=kid_gender)

# Reply to the question/comments added below using the knowledge from the book "Cribsheet 
# by Emily Oster" or "Heidi Murkoff by What to Expect the first year" (whichever is more 
# relevant to the question), giving a concise answers, no longer than 2 paragraphs. Reply 
# in an empathic, friendly manner. If the question is medical, reply as a healthcare provider. 
# Avoid answering like "As an AI language model...". Just give an answer. You can recommend 
# talking to a doctor at the end.

INTRO_ALREADY_MOTHER = """Reply to the question/comments added below using the knowledge from the book "Cribsheet by Emily Oster" or "Heidi Murkoff by What to Expect the first year" (whichever is more relevant to the question), giving a concise answers, no longer than 2 paragraphs. Reply in an empathic, friendly manner. If the question is medical, reply as a healthcare provider. Avoid answering like "As an AI language model...". Just give an answer. You can recommend talking to a doctor at the end."""
HUMAN_ALREADY_MOTHER = "The human has an {kid_age} {kid_gender}.".format(kid_age=kid_age, kid_gender=kid_gender)

HISTORY = """
    Current conversation:
    {history}
    Human: {input}
    AI:
"""

if mother_status == 'Pregnant':
    TEMPLATE = INTRO_PREGNANT + HUMAN_PREGNANT + HISTORY
else:
    TEMPLATE = INTRO_ALREADY_MOTHER + HUMAN_ALREADY_MOTHER + HISTORY

PROMPT_TEMPLATE = PromptTemplate(input_variables=['history', 'input'], template=TEMPLATE)

Conversation = ConversationChain(
        llm=llm, 
        prompt=PROMPT_TEMPLATE,
        memory=st.session_state.memory,
        verbose=True
    )  

st.sidebar.button("New Chat", on_click=new_chat, type='primary')

user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display conversation and enable download
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session