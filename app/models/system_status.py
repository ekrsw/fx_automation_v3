from sqlalchemy import Column, Integer, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class SystemStatus(Base):
    __tablename__ = "system_status"

    id = Column(Integer, primary_key=True, index=True)
    is_trading_enabled = Column(Boolean, default=False, nullable=False)
    last_update = Column(DateTime, default=func.now(), onupdate=func.now())
    active_strategies = Column(Text)