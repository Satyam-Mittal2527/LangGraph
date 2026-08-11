
import streamlit as st
from langgraph_backend_with_database import workflow, retreive_all_threads
from langchain_core.messages import HumanMessage
import uuid


# ============================================================
#                    UTILITY FUNCTIONS
# ============================================================

# Generate a thread ID using the first 5 characters
# of the user's first message.
def generate_thread_id(user_text):
    thread_id = user_text[:30]
    return thread_id


# Reset the current chat.
# This removes the active thread ID and clears the
# messages currently displayed in the chat.
def reset_chat():
    # thread_id = generate_thread_id()
    # st.session_state['thread_id'] = thread_id
    # add_thread(st.session_state['thread_id'])

    st.session_state['thread_id'] = None
    st.session_state['message_history'] = []


# Add a thread ID to the list of conversations
# if it does not already exist.
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


# Load the saved conversation for a particular thread ID
# from the LangGraph checkpointer.
def load_conversation(thread_id):
    response = workflow.get_state(
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    ).values

    if response:
        return response['messages']

    return []


# ============================================================
#                  SESSION STATE INITIALIZATION
# ============================================================

# Stores the messages displayed in the current conversation.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# Stores the ID of the currently active conversation.
# None means that no conversation has been started yet.
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None


# Stores all previously created conversation/thread IDs.
# These are retrieved from the LangGraph SQLite database.
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()


# add_thread(st.session_state['thread_id'])


# ============================================================
#                       SIDEBAR UI
# ============================================================

st.sidebar.title('LangGraph ChatBot')


# Create a new chat.
# This resets the active thread and message history.
if st.sidebar.button('New Chat'):
    reset_chat()


st.sidebar.header('My Conversation')


# Display all previously created conversations.
# The [::-1] reverses the list so the latest thread
# appears at the top.
for thread_id in st.session_state['chat_threads'][::-1]:

    # When a conversation is selected from the sidebar,
    # make it the active thread.
    if st.sidebar.button(str(thread_id)):

        st.session_state['thread_id'] = thread_id

        # Load the saved messages for the selected thread.
        messages = load_conversation(thread_id)

        temp_messages = []

        # Convert LangChain messages into the format
        # expected by Streamlit's chat UI.
        for message in messages:

            # HumanMessage → user
            # Other messages → assistant
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'


            # ------------------------------------------------
            # Extract the message content
            # ------------------------------------------------

            # Case 1: Content is already a string.
            if isinstance(message.content, str):
                content = message.content


            # Case 2: Content is a list of content blocks.
            elif isinstance(message.content, list):
                content = ""

                for block in message.content:

                    # Extract only text blocks.
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        content += block.get("text", "")


            # Case 3: Convert any other content type to string.
            else:
                content = str(message.content)


            # Store the message in Streamlit's message history.
            temp_messages.append({
                "role": role,
                "content": content
            })


        # Replace the current message history with
        # the selected conversation's messages.
        st.session_state["message_history"] = temp_messages


# ============================================================
#                DISPLAY CURRENT CHAT HISTORY
# ============================================================

# Display all messages belonging to the currently
# selected conversation.
for message in st.session_state['message_history']:

    with st.chat_message(message['role']):
        st.text(message['content'])


# ============================================================
#                     USER INPUT
# ============================================================

# CONFIG = {
#     'configurable': {
#         'thread_id': st.session_state['thread_id']
#     }
# }


# Chat input displayed at the bottom of the page.
user_input = st.chat_input('Type here')


# ============================================================
#                  PROCESS USER MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Create a new thread only when there is no active thread.
    #
    # This means:
    #   First message  → creates a new thread
    #   Later messages → reuse the same thread
    # --------------------------------------------------------
    if st.session_state['thread_id'] is None:

        thread = generate_thread_id(user_input)

        st.session_state['thread_id'] = thread

        # Add the newly created thread to the sidebar.
        add_thread(thread)


    # Add the user's message to the current chat history.
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })


    # Display the user's message immediately.
    with st.chat_message('user'):
        st.text(user_input)


    # ========================================================
    #                 GENERATE AI RESPONSE
    # ========================================================

    def generate_response():

        # Stream the response from LangGraph.
        for message_chunk, metadata in workflow.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config={
                'configurable': {
                    'thread_id': st.session_state['thread_id']
                }
            },
            stream_mode="messages"
        ):

            content = message_chunk.content


            # ------------------------------------------------
            # Case 1: AI response is a string
            # ------------------------------------------------
            if isinstance(content, str):
                yield content


            # ------------------------------------------------
            # Case 2: AI response contains content blocks
            # ------------------------------------------------
            elif isinstance(content, list):

                for block in content:

                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        yield block.get("text", "")


    # ========================================================
    #                DISPLAY AI RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        # Stream the AI response to the UI.
        ai_message = st.write_stream(generate_response())


        # Save the complete AI response to the message history.
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })

