from models.schema import Graph_Schema


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
