from typing import cast

from models.models import ConfidenceOutput
from models.schema import Graph_Schema
from prompts.prompts import prompt_confidence
from services.LLM_service import llm


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
