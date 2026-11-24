"""Domain models (issue #7). Importing this package registers all tables."""

from app.models.base import Base, DataOrigin, JsonScalar, JsonValue
from app.models.churn_prediction import ChurnPrediction
from app.models.customer import Customer
from app.models.customer_service import CustomerService

__all__ = [
    "Base",
    "ChurnPrediction",
    "Customer",
    "CustomerService",
    "DataOrigin",
    "JsonScalar",
    "JsonValue",
]
