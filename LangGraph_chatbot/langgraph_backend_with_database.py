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
import requests

search_tool = DuckDuckGoSearchRun(region="us-en")

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

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol.
    Example: AAPL, TSLA, MSFT
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={os.getenv("ALPHAVANTAGE_API_KEY")}"
    )

    r = requests.get(url)
    data = r.json()

    quote = data.get("Global Quote", {})

    if not quote:
        return {
            "error": f"Could not fetch stock price for {symbol}"
        }

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent")
    }

conn = sqlite3.connect(
    database="chatbot.db", 
    check_same_thread=False
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_GENAI_KEY"),
)

tools = [get_stock_price, search_tool, calculator]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    "LLM node that may answer the chat or call the tools "
    message = state['messages']
    
    response = llm_with_tools.invoke(message)

    return {
        'messages': [response]
    }

tool_node = ToolNode(tools)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
# graph.add_edge('chat_node', END)
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge("tools", "chat_node")

workflow = graph.compile(checkpointer=checkpointer)

def retreive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)