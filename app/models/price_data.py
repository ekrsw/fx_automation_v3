from sqlalchemy import Column, Integer, String, DateTime, Numeric, Index
from sqlalchemy.sql import func
from app.core.database import Base


class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    open = Column(Numeric(10, 5), nullable=False)
    high = Column(Numeric(10, 5), nullable=False)
    low = Column(Numeric(10, 5), nullable=False)
    close = Column(Numeric(10, 5), nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_symbol_timeframe_datetime', 'symbol', 'timeframe', 'datetime'),
    )