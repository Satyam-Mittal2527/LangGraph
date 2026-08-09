import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage

CONFIG = {
        'configurable': {
            'thread_id': 'thread_1'
        }
    }
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role': 'user', 'content': 'Hi'}
# {'role': 'assistant', 'content': 'How can I help you'}

user_input = st.chat_input('Type here')
if user_input:

    st.session_state['message_history'].append({'role':'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)

    # response = workflow.invoke({
    #     'messages': HumanMessage(content = user_input)
    # }, config = CONFIG)

    # ai_message = response['messages'][-1].content[0]['text']

    

   

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
