import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

import config

st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

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

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.entity_store)
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)


st.title("🧠 Parent assistant 🤖")

llm = OpenAI(
        temperature=0,
        model_name=config.MODEL,
        verbose=False,
        openai_api_key=st.secrets['OPENAI_API_KEY']
        ) 

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
    
Conversation = ConversationChain(
        llm=llm, 
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )  

st.sidebar.button("New Chat", on_click=new_chat, type='primary')

user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session