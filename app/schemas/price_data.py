from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal


class PriceDataBase(BaseModel):
    symbol: str
    timeframe: str
    datetime: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class PriceDataCreate(PriceDataBase):
    pass


class PriceDataResponse(PriceDataBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}