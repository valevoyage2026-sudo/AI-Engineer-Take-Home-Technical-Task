from typing import List, Optional

from pydantic import BaseModel

from models.models import AccountTier, Tcategory, Tpriority


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


class batch_schema(BaseModel):
    csv_path: str
    results: list[Graph_Schema] = []
    tickets: list[Graph_Schema] = []

    output_csv_path: Optional[str] = None
