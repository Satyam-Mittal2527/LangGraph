import streamlit as st
from langgraph_backend_with_database import workflow, retreive_all_threads
from langchain_core.messages import HumanMessage
import uuid

#------------------Utility Function-------------#
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    # add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    response = workflow.get_state(config={
        'configurable': {
            'thread_id': thread_id
        }
    }).values

    if response:
        return response['messages']

    return []
#--------------------------------------------------#

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()

add_thread(st.session_state['thread_id'])

#---------------SideBarUI--------------------#

st.sidebar.title('LangGraph VhatBot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            # Extract content
            if isinstance(message.content, str):
                content = message.content

            elif isinstance(message.content, list):
                content = ""

                for block in message.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")

            else:
                content = str(message.content)

            temp_messages.append({
                "role": role,
                "content": content
            })

        st.session_state["message_history"] = temp_messages

#---------------------------------------------#

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

CONFIG = {
        'configurable': {
            'thread_id': st.session_state['thread_id']
        }
    }

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append({'role':'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)

    def generate_response():
        for message_chunk, metadata in workflow.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG,
            stream_mode="messages"
        ):
            content = message_chunk.content

            if isinstance(content, str):
                yield content

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block.get("text", "")

                        
    with st.chat_message("assistant"):
        ai_message = st.write_stream(generate_response())

        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })
