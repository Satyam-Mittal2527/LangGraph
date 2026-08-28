from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio

@tool
def calculator(first_num: float, second_num: float, opeator: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers,
    Supported operations: add, sub, mul, div
    """
    try:
        if opeator=="add":
            result = first_num+second_num
        elif opeator == "sub":
            result = first_num - second_num
        elif opeator == "mul":
            result = first_num * second_num
        elif opeator == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            resullt = first_num / second_num
        else:
            return {"error": "Unsupported operator {operator}"}
    except Exception as e:
        return {"Exception:", e}

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_GENAI_KEY"),
)


tools = [calculator]

llm_with_tools = llm.bind_tools(tools)
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def build_graph():
    async def chat_node(state: ChatState):
        "LLM node that may answer the chat or call the tools "
        message = state['messages']
        
        response = await llm_with_tools.ainvoke(message)

        return {
            'messages': [response]
        }

    tool_node = ToolNode(tools)



    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)
    graph.add_edge(START, 'chat_node')
    # graph.add_edge('chat_node', END)
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot


async def main():

    chatbot = build_graph()

    response = await chatbot.ainvoke({'messages': [HumanMessage(content='Find the modulus of 12345 and 23')]})
    
    print(response['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())