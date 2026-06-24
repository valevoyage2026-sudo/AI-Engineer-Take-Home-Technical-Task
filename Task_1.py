import json
import os
from enum import Enum
from multiprocessing import Condition
from operator import add
from typing import Annotated, List, Literal, Optional, TypedDict, cast

import arxiv
import pandas as pd
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
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from requests.utils import InvalidURL
from rich.console import Console

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


class AccountTier(str, Enum):
    STANDARD = "standard"
    VIP = "vip"


class Tcategory(str, Enum):
    SHIPPING = "Shipping & Delivery"
    RETURNS = "Returns & Refunds"
    BILLING = "Billing & Payments"
    TECHNICAL = "Technical Issues"
    GENERAL = "General Enquiry"


class Tpriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


class ConfidenceFlag(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class Graph_Schema(BaseModel):
    messages: List[str]
    ticket_ID: int
    customer_name: str
    account_tier: AccountTier
    subject: str
    priority: Optional[Tpriority] = None
    category: Optional[Tcategory] = None
    confidence_Flag: Optional[ConfidenceFlag] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    critic: Optional[str] = None
    csv_export_path: Optional[str] = None


class ClassificationOutput(BaseModel):
    category: Tcategory
    priority: Tpriority
    reasoning: str


class ConfidenceOutput(BaseModel):
    confidence_flag: ConfidenceFlag
    confidence: float


class CriticOutput(BaseModel):
    critic: Literal["Valid", "Invalid"]


prompt_classification = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the ticket and assign priority."),
        (
            "human",
            """
Ticket ID: {ticket_ID}
Customer: {customer_name}
account_tier: {account_tier}
subject: {subject}
Messages: {messages}
""",
        ),
    ]
)

prompt_confidence = ChatPromptTemplate.from_messages(
    [
        ("system", "Assess the classification confidence."),
        (
            "human",
            """
Ticket ID: {ticket_ID}
Customer: {customer_name}
account_tier: {account_tier}
subject: {subject}
Messages: {messages}
""",
        ),
    ]
)

prompt_crtic = ChatPromptTemplate.from_messages(
    [
        ("system", "Assess the classification confidence."),
        (
            "human",
            """
Ticket ID: {ticket_ID}
Customer: {customer_name}
account_tier: {account_tier}
subject: {subject}
Messages: {messages}
priority: {priority}
category: {category}
reasoning : {reasoning}
confidence_flag:{confidence_flag}
confidence:{confidence}
""",
        ),
    ]
)


def LLm_Classify_priority(state: Graph_Schema) -> dict:
    structured_llm = llm.with_structured_output(ClassificationOutput)
    chain = prompt_classification | structured_llm

    response = cast(
        ClassificationOutput,
        chain.invoke(
            {
                "ticket_ID": state.ticket_ID,
                "customer_name": state.customer_name,
                "account_tier": state.account_tier,
                "subject": state.subject,
                "messages": state.messages,
            }
        ),
    )

    # Return only the updates as a plain dictionary
    return {
        "priority": response.priority,
        "category": response.category,
        "reasoning": response.reasoning,
    }


def Response_Node(state: Graph_Schema) -> dict:
    structured_llm = llm.with_structured_output(ConfidenceOutput)
    chain = prompt_confidence | structured_llm

    response = cast(
        ConfidenceOutput,
        chain.invoke(
            {
                "ticket_ID": state.ticket_ID,
                "customer_name": state.customer_name,
                "account_tier": state.account_tier,
                "subject": state.subject,
                "messages": state.messages,
            }
        ),
    )

    # Return only the updates as a plain dictionary
    return {
        "confidence": response.confidence,
        "confidence_Flag": response.confidence_flag,
    }


def Critic_Node(state: Graph_Schema) -> dict:
    structured_llm = llm.with_structured_output(CriticOutput)
    chain = prompt_crtic | structured_llm

    response = cast(
        CriticOutput,
        chain.invoke(
            {
                "ticket_ID": state.ticket_ID,
                "customer_name": state.customer_name,
                "account_tier": state.account_tier,
                "subject": state.subject,
                "messages": state.messages,
                "priority": state.priority,
                "category": state.category,
                "reasoning": state.reasoning,
                "confidence_flag": state.confidence_Flag,
                "confidence": state.confidence,
            }
        ),
    )

    # Return only the updates as a plain dictionary
    return {"critic": response.critic}


def Generate_CSV_Node(state: Graph_Schema) -> dict:
    ticket_data = {
        "ticket_ID": state.ticket_ID,
        "customer_name": state.customer_name,
        "account_tier": getattr(state.account_tier, "value", str(state.account_tier)),
        "subject": state.subject,
        "category": getattr(state.category, "value", str(state.category)),
        "priority": getattr(state.priority, "value", str(state.priority)),
        "reasoning": state.reasoning,
        "Confidence_Flag": getattr(
            state.confidence_Flag, "value", str(state.confidence_Flag)
        ),
        "Confidence_Score": state.confidence,
        "critic": state.critic,
    }

    df = pd.DataFrame([ticket_data])
    os.makedirs("exports", exist_ok=True)
    filename = f"exports/ticket_{state.ticket_ID}_classification.csv"
    df.to_csv(filename, index=False, encoding="utf-8")

    print(f"Successfully generated CSV at: {filename}")

    # Return only the updates as a plain dictionary
    return {"csv_export_path": filename}


workflow = StateGraph(Graph_Schema)


def call_conditon(state: Graph_Schema) -> str:

    Critic_out = state.critic

    if Critic_out == "Valid":  # is the final message asks for a tool call
        return "Generate_CSV_Node"  # go to tool node
    else:
        return "end"


workflow.add_node("LLm_Classify_priority", LLm_Classify_priority)
workflow.add_node("Response_Node", Response_Node)
workflow.add_node("Critic_Node", Critic_Node)
workflow.add_node("Generate_CSV_Node", Generate_CSV_Node)

# Wiring Flow
workflow.add_edge(START, "Response_Node")
workflow.add_edge(START, "LLm_Classify_priority")
# 2. Fan-In: Both parallel paths must arrive here before the Critic runs
workflow.add_edge("LLm_Classify_priority", "Critic_Node")
workflow.add_edge("Response_Node", "Critic_Node")
# --- Conditional Router ---
workflow.add_conditional_edges(
    "Critic_Node", call_conditon, {"Generate_CSV_Node": "Generate_CSV_Node", "end": END}
)
workflow.add_edge("Generate_CSV_Node", END)

chat = workflow.compile()
png_data = chat.get_graph().draw_mermaid_png()


with open("Task1.png", "wb") as f:
    f.write(png_data)

# # 1. Instantiate the Graph_Schema object directly
# dummy_state = Graph_Schema(
#     ticket_ID=4092,
#     customer_name="Sarah Jenkins",
#     account_tier=AccountTier.VIP,
#     subject="Urgent double billing charge on subscription",
#     messages=[
#         "Hi support, I checked my bank statement today and noticed I was charged $49.99 twice..."
#     ],
# )

# # 2. Invoke the graph with the proper object
# final_state = Graph_Schema.model_validate(chat.invoke(dummy_state))

# print(final_state.category)
# print(final_state.priority)
# print(final_state.critic)
# print(final_state.csv_export_path)
