from enum import Enum
from typing import Literal

from pydantic import BaseModel


class AccountTier(str, Enum):
    STANDARD = "standard"
    VIP = "vip"


class Tcategory(str, Enum):
    SHIPPING = "Shipping & Delivery"
    RETURNS = "Returns & Refunds"
    BILLING = "Billing & Payments"
    TECHNICAL = "Technical Issues"
    GENERAL = "General Enquiry"


class Tpriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


# class ConfidenceFlag(str, Enum):
#     OK = "OK"
#     REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ClassificationOutput(BaseModel):
    category: Tcategory
    priority: Tpriority


class ConfidenceOutput(BaseModel):
    # confidence_flag: ConfidenceFlag
    confidence: float
    reasoning: str


class CriticOutput(BaseModel):
    critic: Literal["Valid", "Invalid"]
    response: str
