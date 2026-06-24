import os
from multiprocessing import Condition
from operator import add
from typing import Annotated, List, TypedDict

import arxiv
import requests
from arxiv import ArxivError, Client, Search
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.tools import tool
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, human
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from rich.console import Console

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


@tool
def ducksearch_tool(query: str) -> str:
    """use this tool to get latest infromation or latest updates about query"""
    duck_search = DuckDuckGoSearchRun()
    result = duck_search.invoke(query)
    print(result)
    return result


@tool
def arxiv_tool(query: str) -> str:
    """this tools uses arixv database to search for papers"""
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=2,
    )
    papers = list(client.results(search))
    if papers:
        for paper in papers:
            print(paper.title)
            return paper.title
        return "papers found"

    else:
        return "no results found"


@tool  # creating custom tool
def wiki_tool(Query: str) -> str:
    """use this tool to search wikipedia for the give query"""

    query = Query

    url = "https://en.wikipedia.org/w/api.php"

    params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
    response = requests.get(url, params=params)

    data = response.json()
    print(data["query"]["search"][0]["title"])
    return data["query"]["search"][0]["title"]


tools = [arxiv_tool, wiki_tool, ducksearch_tool]

llm_tools = llm.bind_tools(tools)

response = llm_tools.invoke("Tell me what is the lastest news on attention residuals")


class graph_schema(TypedDict):
    messages: List


Initial_promt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpfull/Funny AI agent"),
        MessagesPlaceholder("messages"),
    ]
)


def llm_node(state: graph_schema) -> graph_schema:

    chain = Initial_promt | llm_tools

    response = chain.invoke({"messages": state["messages"]})

    state["messages"].append(response)  # not content because we want the ai message obj

    return state


def tool_node(state: graph_schema) -> graph_schema:

    message = state["messages"]

    tools_by_name = {tool.name: tool for tool in tools}
    result = []

    for tool_call in message[-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        observation = tool.invoke(tool_call["args"])

        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    state["messages"].extend(result)
    return state


graph = StateGraph(graph_schema)

graph.add_node("LLM", llm_node)
graph.add_node("Tools", tool_node)


def tool_call_conditon(state: graph_schema) -> str:

    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage) and last_message.tool_calls
    ):  # is the final message asks for a tool call
        return "Tools"  # go to tool node
    else:
        return "end"


graph.add_edge(START, "LLM")
graph.add_conditional_edges("LLM", tool_call_conditon, {"Tools": "Tools", "end": END})
graph.add_edge("Tools", "LLM")  # get info back from tool


chat = graph.compile()
# png_data = chat.get_graph().draw_mermaid_png()


# with open("graph.png", "wb") as f:
#     f.write(png_data)


# chat.invoke(
#     {"messages": [HumanMessage(content="latest news about attention residuals")]}
# )

console = Console()
for chunk in chat.stream(
    {"messages": [HumanMessage(content="latest news about attention residuals")]},
    stream_mode="updates",
):
    for node, value in chunk.items():
        msg = value["messages"][-1]

        console.rule(f"[bold cyan]{node}")

        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                console.print("[yellow]Tool Calls:[/yellow]")

                for tc in msg.tool_calls:
                    console.print(f"[bold]{tc['name']}[/bold] {tc['args']}")

            else:
                console.print(msg.content, style="green")

        elif isinstance(msg, ToolMessage):
            console.print("[magenta]Tool Output:[/magenta]")
            console.print(msg.content)
