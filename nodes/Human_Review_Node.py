from models.models import Tcategory, Tpriority
from models.schema import Graph_Schema


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
