from langchain_core.prompts import ChatPromptTemplate

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
