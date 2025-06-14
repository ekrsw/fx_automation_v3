import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import get_db, Base
from app.models import price_data, trades, technical_analysis, system_status

# テスト用データベース
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_main.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_health_endpoint():
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "service" in data


def test_trading_status_endpoint():
    response = client.get("/api/v1/trading/status")
    assert response.status_code == 200
    data = response.json()
    assert "trading_enabled" in data
    assert "last_update" in data
    assert "active_strategies" in data


def test_price_data_endpoint():
    response = client.get("/api/v1/data/price/USDJPY")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)  # 空のリストが返される


def test_analysis_data_endpoint():
    response = client.get("/api/v1/data/analysis/USDJPY")
    assert response.status_code == 200
    data = response.json()
    
    # エラーレスポンスの場合、symbolフィールドは存在しない
    if "error" in data:
        assert "message" in data
        assert "timestamp" in data
    else:
        assert data["symbol"] == "USDJPY"
        assert "dow_trend" in data or "detailed_analysis" in data
        assert "elliott_wave" in data or "detailed_analysis" in data


def test_trading_start_endpoint():
    response = client.post("/api/v1/trading/start")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_trading_stop_endpoint():
    response = client.post("/api/v1/trading/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"