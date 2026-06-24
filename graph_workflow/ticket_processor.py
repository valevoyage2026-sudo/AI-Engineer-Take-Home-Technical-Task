from langgraph.graph import END, START, StateGraph

from models.schema import Graph_Schema
from nodes.Condition_Divider import Condition_Divider
from nodes.Critic_Node import Critic_Node
from nodes.Human_Review_Node import Human_Review_Node
from nodes.LLM_Classifier import LLm_Classify_priority
from nodes.Response_Node import Response_Node
from nodes.Routing_Node import Routing

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
