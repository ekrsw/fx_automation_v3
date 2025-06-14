import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.positions import Position, PositionType, PositionStatus, PositionHistory, Base


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_position_creation(db_session):
    """ポジション作成テスト"""
    position = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.PENDING,
        lot_size=0.1,
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=152.0
    )
    
    db_session.add(position)
    db_session.commit()
    
    assert position.id is not None
    assert position.symbol == "USDJPY"
    assert position.position_type == PositionType.BUY
    assert position.is_open is False  # PENDING状態


def test_position_properties(db_session):
    """ポジションプロパティテスト"""
    position = Position(
        symbol="EURUSD",
        position_type=PositionType.SELL,
        status=PositionStatus.OPEN,
        lot_size=0.2,
        entry_price=1.2000,
        current_price=1.1950,
        stop_loss=1.2100,
        take_profit=1.1800,
        profit_loss=100.0,
        net_profit=95.0  # 手数料差引後の利益
    )
    
    # ステータス確認
    assert position.is_open is True
    assert position.is_closed is False
    assert position.is_profitable is True
    
    # 含み損益計算
    unrealized_pnl = position.unrealized_pnl
    assert unrealized_pnl > 0  # SELL position with current_price < entry_price
    
    # リスクリワード比
    risk_reward = position.risk_reward_ratio
    assert risk_reward > 0
    expected_rr = abs(1.1800 - 1.2000) / abs(1.2000 - 1.2100)
    assert abs(risk_reward - expected_rr) < 0.01


def test_position_unrealized_pnl_buy():
    """買いポジションの含み損益計算"""
    position = Position(
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        entry_price=150.0,
        current_price=151.0
    )
    
    # 買いポジションで価格上昇 → 利益
    assert position.unrealized_pnl > 0


def test_position_unrealized_pnl_sell():
    """売りポジションの含み損益計算"""
    position = Position(
        position_type=PositionType.SELL,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        entry_price=150.0,
        current_price=149.0
    )
    
    # 売りポジションで価格下落 → 利益
    assert position.unrealized_pnl > 0


def test_position_closed_state():
    """クローズ済みポジションのテスト"""
    position = Position(
        symbol="GBPJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.CLOSED,
        lot_size=0.05,
        entry_price=180.0,
        exit_price=182.0,
        profit_loss=100.0,
        net_profit=95.0  # 手数料考慮
    )
    
    assert position.is_closed is True
    assert position.is_open is False
    assert position.is_profitable is True


def test_position_risk_reward_ratio_edge_cases():
    """リスクリワード比の境界値テスト"""
    # 全て設定済み
    position1 = Position(
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=106.0
    )
    assert position1.risk_reward_ratio == 3.0  # 6/2
    
    # 値が未設定
    position2 = Position()
    assert position2.risk_reward_ratio == 0.0
    
    # ストップロスがエントリー価格と同じ（リスクゼロ）
    position3 = Position(
        entry_price=100.0,
        stop_loss=100.0,
        take_profit=105.0
    )
    assert position3.risk_reward_ratio == 0.0


def test_position_history_creation(db_session):
    """ポジション履歴作成テスト"""
    position = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1
    )
    
    db_session.add(position)
    db_session.commit()
    
    # 履歴レコード作成
    history = PositionHistory(
        position_id=position.id,
        action="open",
        price=150.0,
        volume=0.1,
        reason="Initial open",
        comments="Test position opened"
    )
    
    db_session.add(history)
    db_session.commit()
    
    assert history.id is not None
    assert history.position_id == position.id
    assert history.action == "open"


def test_position_lifecycle(db_session):
    """ポジションライフサイクルテスト"""
    # 1. 作成
    position = Position(
        symbol="EURJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.PENDING,
        lot_size=0.1,
        entry_price=160.0,
        stop_loss=158.0,
        take_profit=164.0
    )
    
    db_session.add(position)
    db_session.commit()
    
    assert position.status == PositionStatus.PENDING
    
    # 2. オープン
    position.status = PositionStatus.OPEN
    position.opened_at = datetime.now()
    position.current_price = 160.5
    
    db_session.commit()
    
    assert position.is_open is True
    assert position.opened_at is not None
    
    # 3. クローズ
    position.status = PositionStatus.CLOSED
    position.exit_price = 163.0
    position.closed_at = datetime.now()
    position.profit_loss = 300.0  # 3 pips * 0.1 lot
    position.net_profit = 295.0   # 手数料差引
    
    db_session.commit()
    
    assert position.is_closed is True
    assert position.is_profitable is True
    assert position.closed_at is not None


def test_position_repr():
    """ポジション文字列表現テスト"""
    position = Position(
        id=1,
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1
    )
    
    repr_str = repr(position)
    assert "Position(id=1" in repr_str
    assert "symbol=USDJPY" in repr_str
    assert "type=PositionType.BUY" in repr_str
    assert "status=PositionStatus.OPEN" in repr_str
    assert "lot=0.1" in repr_str


def test_position_history_repr():
    """ポジション履歴文字列表現テスト"""
    history = PositionHistory(
        id=1,
        position_id=123,
        action="close"
    )
    
    repr_str = repr(history)
    assert "PositionHistory(id=1" in repr_str
    assert "position_id=123" in repr_str
    assert "action=close" in repr_str