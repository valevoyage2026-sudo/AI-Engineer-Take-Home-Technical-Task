from langgraph.graph import END, START, StateGraph

from models.schema import batch_schema
from nodes.load_tickets import load_tickets, process_ticket_csv, save_to_csv

Batch_load = StateGraph(batch_schema)

Batch_load.add_node("load_tickets", load_tickets)
Batch_load.add_node("process_ticket_csv", process_ticket_csv)
Batch_load.add_node("save_to_csv", save_to_csv)

Batch_load.add_edge(START, "load_tickets")
Batch_load.add_edge("load_tickets", "process_ticket_csv")
Batch_load.add_edge("process_ticket_csv", "save_to_csv")
Batch_load.add_edge("save_to_csv", END)

batch = Batch_load.compile()
