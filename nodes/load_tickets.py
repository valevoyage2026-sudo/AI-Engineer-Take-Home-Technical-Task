import pandas as pd

from graph_workflow.ticket_processor import chat
from models.models import AccountTier
from models.schema import Graph_Schema, batch_schema


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
