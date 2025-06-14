from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class TechnicalAnalysis(Base):
    __tablename__ = "technical_analysis"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    dow_trend = Column(String(20))
    elliott_wave_count = Column(String(10))
    swing_points = Column(Text)
    signals = Column(Text)
    created_at = Column(DateTime, default=func.now())