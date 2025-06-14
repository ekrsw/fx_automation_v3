import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from decimal import Decimal
from app.core.database import Base
from app.models.price_data import PriceData
from app.models.trades import Trade
from app.models.technical_analysis import TechnicalAnalysis
from app.models.system_status import SystemStatus

# テスト用データベース
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_price_data_model(db):
    price_data = PriceData(
        symbol="USDJPY",
        timeframe="M1",
        datetime=datetime.now(),
        open=Decimal("110.00"),
        high=Decimal("110.50"),
        low=Decimal("109.50"),
        close=Decimal("110.25"),
        volume=1000
    )
    
    db.add(price_data)
    db.commit()
    
    saved_data = db.query(PriceData).filter(PriceData.symbol == "USDJPY").first()
    assert saved_data is not None
    assert saved_data.symbol == "USDJPY"
    assert saved_data.timeframe == "M1"
    assert saved_data.open == Decimal("110.00")


def test_trade_model(db):
    trade = Trade(
        symbol="USDJPY",
        entry_time=datetime.now(),
        entry_price=Decimal("110.00"),
        position_size=Decimal("1.0"),
        strategy_name="Test Strategy"
    )
    
    db.add(trade)
    db.commit()
    
    saved_trade = db.query(Trade).filter(Trade.symbol == "USDJPY").first()
    assert saved_trade is not None
    assert saved_trade.symbol == "USDJPY"
    assert saved_trade.strategy_name == "Test Strategy"


def test_technical_analysis_model(db):
    analysis = TechnicalAnalysis(
        symbol="USDJPY",
        datetime=datetime.now(),
        dow_trend="UPTREND",
        elliott_wave_count="Wave3",
        swing_points="test_points",
        signals="test_signals"
    )
    
    db.add(analysis)
    db.commit()
    
    saved_analysis = db.query(TechnicalAnalysis).filter(
        TechnicalAnalysis.symbol == "USDJPY"
    ).first()
    assert saved_analysis is not None
    assert saved_analysis.dow_trend == "UPTREND"


def test_system_status_model(db):
    status = SystemStatus(
        is_trading_enabled=True,
        active_strategies="Strategy1,Strategy2"
    )
    
    db.add(status)
    db.commit()
    
    saved_status = db.query(SystemStatus).first()
    assert saved_status is not None
    assert saved_status.is_trading_enabled == True