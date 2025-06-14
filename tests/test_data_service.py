import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.price_data import PriceData
from app.services.data_service import DataService

# テスト用データベース
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_data.db"
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


@pytest.fixture
def data_service(db):
    return DataService(db)


def test_save_price_data(data_service):
    test_data = pd.DataFrame({
        'datetime': [datetime.now()],
        'open': [110.0],
        'high': [110.5],
        'low': [109.5],
        'close': [110.2],
        'volume': [1000]
    })
    
    saved_count = data_service.save_price_data("USDJPY", "M1", test_data)
    
    assert saved_count == 1
    
    # 重複保存テスト
    saved_count = data_service.save_price_data("USDJPY", "M1", test_data)
    assert saved_count == 0  # 既存データなので保存されない


def test_fetch_and_save_data_success(data_service):
    with patch.object(data_service, 'mt5_service') as mock_mt5:
        mock_mt5.connect.return_value = True
        
        test_data = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [110.0],
            'high': [110.5],
            'low': [109.5],
            'close': [110.2],
            'volume': [1000]
        })
        mock_mt5.get_price_data.return_value = test_data
        
        result = data_service.fetch_and_save_data("USDJPY", "M1", 100)
        
        assert result == 1
        mock_mt5.connect.assert_called_once()
        mock_mt5.disconnect.assert_called_once()


def test_fetch_and_save_data_connection_failed(data_service):
    with patch.object(data_service, 'mt5_service') as mock_mt5:
        mock_mt5.connect.return_value = False
        
        result = data_service.fetch_and_save_data("USDJPY", "M1", 100)
        
        assert result == 0


def test_get_price_data(data_service, db):
    # テストデータ作成
    price_data = PriceData(
        symbol="USDJPY",
        timeframe="M1",
        datetime=datetime.now(),
        open=Decimal("110.0"),
        high=Decimal("110.5"),
        low=Decimal("109.5"),
        close=Decimal("110.2"),
        volume=1000
    )
    db.add(price_data)
    db.commit()
    
    result = data_service.get_price_data("USDJPY", "M1", 10)
    
    assert len(result) == 1
    assert result[0].symbol == "USDJPY"


def test_get_latest_price(data_service, db):
    # テストデータ作成
    price_data = PriceData(
        symbol="USDJPY",
        timeframe="M1",
        datetime=datetime.now(),
        open=Decimal("110.0"),
        high=Decimal("110.5"),
        low=Decimal("109.5"),
        close=Decimal("110.2"),
        volume=1000
    )
    db.add(price_data)
    db.commit()
    
    result = data_service.get_latest_price("USDJPY", "M1")
    
    assert result is not None
    assert result.symbol == "USDJPY"
    assert result.close == Decimal("110.2")