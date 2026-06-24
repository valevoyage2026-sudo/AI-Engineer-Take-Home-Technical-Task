from typing import cast

from models.models import AccountTier, ClassificationOutput, Tpriority
from models.schema import Graph_Schema
from prompts.prompts import prompt_classification
from services.LLM_service import llm


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
