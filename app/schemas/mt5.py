from pydantic import BaseModel
from typing import Optional
from decimal import Decimal


class MT5ConnectionStatus(BaseModel):
    connected: bool
    message: str


class SymbolInfo(BaseModel):
    symbol: str
    bid: Decimal
    ask: Decimal
    spread: int
    digits: int
    point: Decimal


class AccountInfo(BaseModel):
    login: int
    balance: Decimal
    equity: Decimal
    profit: Decimal
    margin: Decimal
    currency: str


class DataFetchRequest(BaseModel):
    symbol: str
    timeframe: str = "M1"
    count: int = 100