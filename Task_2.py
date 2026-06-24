import json
import os
from enum import Enum
from logging import critical
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
from langchain_core.utils.pydantic import model_validate
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
# # Change this block in your code
# llm = ChatOllama(
#     model="gemma3:4b",  # Or "qwen2.5", "mistral", whatever model you have pulled locally
#     temperature=0.1,  # Dropping down from 1.0 slightly so structural responses are more stable
# )


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


# class ConfidenceFlag(str, Enum):
#     OK = "OK"
#     REVIEW_REQUIRED = "REVIEW_REQUIRED"


class Graph_Schema(BaseModel):
    messages: List[str]
    ticket_ID: int
    customer_name: str
    account_tier: AccountTier
    subject: str
    priority: Optional[Tpriority] = None
    category: Optional[Tcategory] = None
    # confidence_Flag: Optional[ConfidenceFlag] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    critic: Optional[str] = None
    response: Optional[str] = None
    csv_export_path: Optional[str] = None


class ClassificationOutput(BaseModel):
    category: Tcategory
    priority: Tpriority


class ConfidenceOutput(BaseModel):
    # confidence_flag: ConfidenceFlag
    confidence: float
    reasoning: str


class CriticOutput(BaseModel):
    critic: Literal["Valid", "Invalid"]
    response: str


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
        (
            "system",
            """Assess the parameter for confidence
- If the message by the user is not relevant, then confidence is 0.
- If the predicted category and priority match the subject and the message, assign high confidence.
- If the message contains unreasonable demands, give an appropriate reasoning.
- If the confidence score is between 0.5 and 0.6, set the confidence flag to "REVIEW_REQUIRED".
 - if the message is from vip account holder and urgent set the confidence to 0.55""",
        ),
        (
            "human",
            """
Ticket ID: {ticket_ID}
Customer: {customer_name}
Account Tier: {account_tier}
Subject: {subject}
Messages: {messages}

[Predicted Classifications to Evaluate]
Predicted Category: {category}
Predicted Priority: {priority}
""",
        ),
    ]
)
prompt_crtic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict Quality Assurance critic. Your job is to verify if the ticket classification matches the customer's request.

                    CRITERIA FOR 'Valid':
                    - The assigned category directly relates to the message content.
                    - The priority matches the urgency (e.g., billing issues or VIP tiers generally get higher priority).
                    - The reasoning makes logical sense.
                    -confidence>.5.

                    If the classification satisfies these rules, output 'Valid'. If there is a clear mismatch, output 'Invalid'.

                    Criteria for response
                    -depending upon the confidence rate ,classifiaction,priority genrate a appropriate response.
                    -the messages with urgent,high priority should have carefull and quick action promising response.
                    -the messages from vip account tier should treated with carefull consederations
                    -the messages with vip tier and high,urgent prioprity should have a proper response assuring immediate solution.

                    """,
        ),
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

    if state.account_tier == AccountTier.VIP:
        response.priority = Tpriority.URGENT
    # Return only the updates as a plain dictionary
    return {
        "priority": response.priority,
        "category": response.category,
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
                "priority": state.priority,
                "category": state.category,
            }
        ),
    )

    # Return only the updates as a plain dictionary
    return {
        "confidence": response.confidence,
        # "confidence_Flag": response.confidence_flag,
        "reasoning": response.reasoning,
    }


def Routing(state: Graph_Schema):
    return {}


def Condition_Divider(state: Graph_Schema):

    if state.confidence is None:
        return
    elif (
        0.45 < state.confidence <= 0.6
    ):  # and state.confidence_Flag == "REVIEW_REQUIRED":
        return "Human_Review_Node"
        # elif state.confidence_Flag == "REVIEW_REQUIRED":
        # return "Human_Review_Node"
    else:
        return "Critic_Node"


def Human_Review_Node(state: Graph_Schema):
    print("\nHUMAN REVIEW REQUIRED ")
    print(f"Ticket ID: {state.ticket_ID}")
    print(f"Subject: {state.subject}")
    print(f"Message: {state.messages[0]}")
    print(f"Predicted Category: {state.category}")
    print(f"Predicted Priority: {state.priority}")
    print(f"Confidence: {state.confidence}")
    print(f"Account: {state.account_tier}")

    print(
        """    1.Category (Shipping & Delivery,
               2.Returns & Refunds,
               3. Billing & Payments,
               4.Technical Issues,
               5. General Enquiry): """
    )

    mapping_cat = {
        1: Tcategory.SHIPPING,
        2: Tcategory.RETURNS,
        3: Tcategory.BILLING,
        4: Tcategory.TECHNICAL,
        5: Tcategory.GENERAL,
    }

    while True:
        try:
            choice = int(input("\nEnter Category choice:"))
            if choice in mapping_cat:
                category = mapping_cat[choice]
                break
            else:
                print("\n Invalid input,please try again(1,2,3,4)")
        except ValueError:
            print("Invalid input,please try again(1,2,3,4)")

    print("Priority :1.Low, 2.Medium, 3.High, 4.Urgent")
    mapping_pri = {
        1: Tpriority.LOW,
        2: Tpriority.MEDIUM,
        3: Tpriority.HIGH,
        4: Tpriority.URGENT,
    }
    pri_choice = int(input("\nEnter priority choice:"))

    priority = mapping_pri[pri_choice]

    response = input("enter the resposne:")

    critic = input("Critic[Valid or Invalid]:")

    return {
        "category": category,
        "priority": priority,
        "critic": critic,
        "response": response,
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
                # "confidence_flag": state.confidence_Flag,
                "confidence": state.confidence,
            }
        ),
    )

    # Return only the updates as a plain dictionary
    return {"critic": response.critic, "response": response.response}


# def Generate_CSV_Node(state: Graph_Schema) -> dict:
#     ticket_data = {
#         "ticket_ID": state.ticket_ID,
#         "customer_name": state.customer_name,
#         "account_tier": getattr(state.account_tier, "value", str(state.account_tier)),
#         "subject": state.subject,
#         "category": getattr(state.category, "value", str(state.category)),
#         "priority": getattr(state.priority, "value", str(state.priority)),
#         "reasoning": state.reasoning,
#         "Confidence_Flag": getattr(
#             state.confidence_Flag, "value", str(state.confidence_Flag)
#         ),
#         "Confidence_Score": state.confidence,
#         "critic": state.critic,
#     }

#     df = pd.DataFrame([ticket_data])
#     os.makedirs("exports", exist_ok=True)
#     filename = f"exports/ticket_{state.ticket_ID}_classification.csv"
#     df.to_csv(filename, index=False, encoding="utf-8")

#     print(f"Successfully generated CSV at: {filename}")

#     # Return only the updates as a plain dictionary
#     return {"csv_export_path": filename}


workflow = StateGraph(Graph_Schema)


# def call_conditon(state: Graph_Schema) -> str:

#     Critic_out = state.critic

#     if Critic_out == "Valid":  # is the final message asks for a tool call
#         return "Generate_CSV_Node"  # go to tool node
#     else:
#         return "end"


workflow.add_node("LLm_Classify_priority", LLm_Classify_priority)
workflow.add_node("Response_Node", Response_Node)
workflow.add_node("Critic_Node", Critic_Node)
workflow.add_node("Human_Review_Node", Human_Review_Node)
workflow.add_node("Routing", Routing)

# Wiring Flow
workflow.add_edge(START, "LLm_Classify_priority")
workflow.add_edge("LLm_Classify_priority", "Response_Node")

# 2. Fan-In: Both parallel paths must arrive here before the Critic runs

workflow.add_edge("Response_Node", "Routing")
workflow.add_conditional_edges(
    "Routing",
    Condition_Divider,
    {"Human_Review_Node": "Human_Review_Node", "Critic_Node": "Critic_Node"},
)


# Conditional Router
workflow.add_edge("Critic_Node", END)
workflow.add_edge("Human_Review_Node", END)

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


class batch_schema(BaseModel):
    csv_path: str
    results: list[Graph_Schema] = []
    tickets: list[Graph_Schema] = []

    output_csv_path: Optional[str] = None


def load_tickets(state: batch_schema):

    df = pd.read_csv(state.csv_path)

    tickets_list = df.apply(
        lambda row: Graph_Schema(
            ticket_ID=row["ticket_id"],
            customer_name=row["customer_name"],
            # Lowercase and strip whitespace to prevent mismatch errors
            account_tier=AccountTier(str(row["account_tier"]).strip().lower()),
            subject=row["subject"],
            messages=[row["message"]],
        ),
        axis=1,
    ).tolist()  # a tickets takes lists

    return {"tickets": tickets_list}


def process_ticket_csv(state: batch_schema):

    results = []

    for ticket in state.tickets:
        result = Graph_Schema.model_validate(chat.invoke(ticket))

        results.append(result)

    return {"results": results}


def save_to_csv(state: batch_schema):

    rows = [result.model_dump() for result in state.results]
    df = pd.DataFrame(rows)
    Filename = "Processed_csv.csv"
    df.to_csv(Filename, index=False)

    return {"output_csv_path": Filename}


Batch_load = StateGraph(batch_schema)

Batch_load.add_node("load_tickets", load_tickets)
Batch_load.add_node("process_ticket_csv", process_ticket_csv)
Batch_load.add_node("save_to_csv", save_to_csv)

Batch_load.add_edge(START, "load_tickets")
Batch_load.add_edge("load_tickets", "process_ticket_csv")
Batch_load.add_edge("process_ticket_csv", "save_to_csv")
Batch_load.add_edge("save_to_csv", END)

batch = Batch_load.compile()
# png_data = batch.get_graph().draw_mermaid_png()


# with open("Task2.png", "wb") as f:
#     f.write(png_data)


Input = batch_schema(csv_path="t.csv", output_csv_path="")

batch_schema.model_validate(batch.invoke(Input))
