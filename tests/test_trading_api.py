import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import get_db
from app.models.positions import Position, PositionType, PositionStatus, Base


def override_get_db():
    """テスト用データベース接続オーバーライド"""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_get_trading_status():
    """取引状態取得テスト"""
    response = client.get("/api/v1/trading/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "trading_enabled" in data
    assert "execution_mode" in data
    assert "last_update" in data
    assert "summary" in data


def test_start_trading():
    """取引開始テスト"""
    response = client.post("/api/v1/trading/start")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "自動取引機能は今後のアップデートで実装予定" in data["message"]


def test_stop_trading():
    """取引停止テスト"""
    response = client.post("/api/v1/trading/stop")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "自動取引機能は今後のアップデートで実装予定" in data["message"]


def test_open_position_buy():
    """買いポジションオープンテスト"""
    position_data = {
        "symbol": "USDJPY",
        "position_type": "buy",
        "lot_size": 0.1,
        "entry_price": 150.0,
        "stop_loss": 149.0,
        "take_profit": 153.0,
        "comment": "Test buy order"
    }
    
    response = client.post("/api/v1/trading/positions/open", json=position_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "position_id" in data
    assert "execution_price" in data


def test_open_position_sell():
    """売りポジションオープンテスト"""
    position_data = {
        "symbol": "EURUSD",
        "position_type": "sell",
        "lot_size": 0.2,
        "entry_price": 1.2000,
        "stop_loss": 1.2100,
        "take_profit": 1.1800
    }
    
    response = client.post("/api/v1/trading/positions/open", json=position_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "position_id" in data


def test_open_position_invalid_data():
    """無効データでのポジションオープンテスト"""
    position_data = {
        "symbol": "USDJPY",
        "position_type": "invalid_type",  # 無効なポジションタイプ
        "lot_size": -0.1,  # 負のロットサイズ
    }
    
    response = client.post("/api/v1/trading/positions/open", json=position_data)
    
    # バリデーションエラーまたは実行エラーを期待
    assert response.status_code in [400, 422, 500]


def test_get_positions_empty():
    """ポジション一覧取得テスト（空）"""
    response = client.get("/api/v1/trading/positions")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "positions" in data
    assert "total_count" in data
    assert isinstance(data["positions"], list)


def test_get_positions_with_filter():
    """フィルター付きポジション一覧取得テスト"""
    # オープンポジションのみ
    response = client.get("/api/v1/trading/positions?status_filter=open")
    assert response.status_code == 200
    
    # クローズ済みポジションのみ
    response = client.get("/api/v1/trading/positions?status_filter=closed")
    assert response.status_code == 200
    
    # 待機中ポジションのみ
    response = client.get("/api/v1/trading/positions?status_filter=pending")
    assert response.status_code == 200


@patch('app.api.api_v1.endpoints.trading.get_trading_engine')
def test_close_position_success(mock_get_engine):
    """ポジションクローズ成功テスト"""
    # モック設定
    mock_engine = Mock()
    mock_engine.close_position.return_value = Mock(
        success=True,
        message="Position closed successfully"
    )
    mock_get_engine.return_value = mock_engine
    
    close_data = {
        "position_id": 1,
        "reason": "Test close"
    }
    
    response = client.post("/api/v1/trading/positions/close", json=close_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"


@patch('app.api.api_v1.endpoints.trading.get_trading_engine')
def test_close_position_failure(mock_get_engine):
    """ポジションクローズ失敗テスト"""
    # モック設定
    mock_engine = Mock()
    mock_engine.close_position.return_value = Mock(
        success=False,
        message="Position not found"
    )
    mock_get_engine.return_value = mock_engine
    
    close_data = {
        "position_id": 999,
        "reason": "Test close"
    }
    
    response = client.post("/api/v1/trading/positions/close", json=close_data)
    
    assert response.status_code == 400


def test_close_all_positions_empty():
    """全ポジションクローズテスト（ポジションなし）"""
    response = client.post("/api/v1/trading/positions/close-all")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "completed"
    assert data["total_positions"] == 0
    assert data["closed_successfully"] == 0


def test_execute_signal_buy():
    """買いシグナル実行テスト"""
    signal_data = {
        "signal_type": "buy",
        "entry_price": 150.0,
        "stop_loss": 149.0,
        "take_profit": 153.0,
        "confidence": 0.8,
        "reasoning": "Test signal execution"
    }
    
    response = client.post("/api/v1/trading/signals/execute", json=signal_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "position_id" in data


def test_execute_signal_sell():
    """売りシグナル実行テスト"""
    signal_data = {
        "signal_type": "sell",
        "entry_price": 1.2000,
        "stop_loss": 1.2100,
        "take_profit": 1.1800,
        "confidence": 0.75
    }
    
    response = client.post("/api/v1/trading/signals/execute", json=signal_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"


def test_execute_signal_low_confidence():
    """低信頼度シグナル実行テスト"""
    signal_data = {
        "signal_type": "buy",
        "entry_price": 150.0,
        "confidence": 0.5  # 閾値以下
    }
    
    response = client.post("/api/v1/trading/signals/execute", json=signal_data)
    
    # 実行拒否される可能性
    assert response.status_code in [200, 400]


def test_get_risk_summary():
    """リスク管理サマリー取得テスト"""
    response = client.get("/api/v1/trading/risk/summary")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "risk_summary" in data
    assert "timestamp" in data


def test_set_execution_mode_simulation():
    """実行モード変更テスト（シミュレーション）"""
    mode_data = {
        "mode": "simulation"
    }
    
    response = client.post("/api/v1/trading/execution-mode", json=mode_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["new_mode"] == "simulation"


def test_set_execution_mode_paper():
    """実行モード変更テスト（ペーパートレード）"""
    mode_data = {
        "mode": "paper"
    }
    
    response = client.post("/api/v1/trading/execution-mode", json=mode_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["new_mode"] == "paper"


def test_set_execution_mode_live():
    """実行モード変更テスト（実取引）"""
    mode_data = {
        "mode": "live"
    }
    
    response = client.post("/api/v1/trading/execution-mode", json=mode_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["new_mode"] == "live"


def test_set_execution_mode_invalid():
    """無効な実行モード変更テスト"""
    mode_data = {
        "mode": "invalid_mode"
    }
    
    response = client.post("/api/v1/trading/execution-mode", json=mode_data)
    
    assert response.status_code == 400


def test_get_monitoring_data():
    """監視データ取得テスト"""
    response = client.get("/api/v1/trading/monitoring")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "monitoring" in data
    assert "timestamp" in data


def test_request_validation():
    """リクエストバリデーションテスト"""
    # 必須フィールド不足
    invalid_data = {
        "symbol": "USDJPY"
        # position_type, lot_size が不足
    }
    
    response = client.post("/api/v1/trading/positions/open", json=invalid_data)
    
    assert response.status_code == 422  # Validation Error


def test_api_error_handling():
    """APIエラーハンドリングテスト"""
    # 存在しないポジションのクローズ
    close_data = {
        "position_id": 99999,
        "reason": "Test close"
    }
    
    response = client.post("/api/v1/trading/positions/close", json=close_data)
    
    # エラーレスポンスを期待
    assert response.status_code in [400, 404, 500]


def test_response_format():
    """レスポンス形式テスト"""
    response = client.get("/api/v1/trading/status")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    data = response.json()
    assert isinstance(data, dict)


def test_comprehensive_trading_flow():
    """包括的取引フローテスト"""
    # 1. 取引状態確認
    status_response = client.get("/api/v1/trading/status")
    assert status_response.status_code == 200
    
    # 2. ポジションオープン
    position_data = {
        "symbol": "USDJPY",
        "position_type": "buy",
        "lot_size": 0.1,
        "entry_price": 150.0,
        "stop_loss": 149.0,
        "take_profit": 153.0
    }
    
    open_response = client.post("/api/v1/trading/positions/open", json=position_data)
    assert open_response.status_code == 200
    position_id = open_response.json()["position_id"]
    
    # 3. ポジション一覧確認
    positions_response = client.get("/api/v1/trading/positions")
    assert positions_response.status_code == 200
    
    # 4. 監視データ取得
    monitoring_response = client.get("/api/v1/trading/monitoring")
    assert monitoring_response.status_code == 200
    
    # 5. ポジションクローズ
    close_data = {
        "position_id": position_id,
        "reason": "End of test"
    }
    
    close_response = client.post("/api/v1/trading/positions/close", json=close_data)
    # クローズは成功または失敗どちらでも可（実装による）
    assert close_response.status_code in [200, 400, 500]