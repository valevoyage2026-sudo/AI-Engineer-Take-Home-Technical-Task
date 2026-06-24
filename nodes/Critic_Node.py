from typing import cast

from models.models import CriticOutput
from models.schema import Graph_Schema
from prompts.prompts import prompt_crtic
from services.LLM_service import llm


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
