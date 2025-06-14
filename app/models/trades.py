from sqlalchemy import Column, Integer, String, DateTime, Numeric
from sqlalchemy.sql import func
from app.core.database import Base


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    entry_price = Column(Numeric(10, 5), nullable=False)
    exit_price = Column(Numeric(10, 5))
    position_size = Column(Numeric(10, 2), nullable=False)
    profit_loss = Column(Numeric(10, 2))
    strategy_name = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())