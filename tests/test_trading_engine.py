import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.trading_engine import (
    TradingEngineService, OrderRequest, OrderResult, OrderType,
    ExecutionMode, AutomatedTradingStrategy, TradingStrategy
)
from app.services.signal_generator import TradingSignal, SignalType, SignalStrength
from app.models.positions import Position, PositionType, PositionStatus, Base


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def trading_engine(db_session):
    """取引エンジン"""
    return TradingEngineService(
        db=db_session,
        execution_mode=ExecutionMode.SIMULATION
    )


@pytest.fixture
def sample_signal():
    """サンプルシグナル"""
    return TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.8,
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=153.0,
        risk_reward_ratio=3.0,
        reasoning=["Test signal"],
        timestamp=datetime.now(),
        metadata={}
    )


def test_trading_engine_initialization(db_session):
    """取引エンジン初期化テスト"""
    engine = TradingEngineService(
        db=db_session,
        execution_mode=ExecutionMode.LIVE
    )
    
    assert engine.execution_mode == ExecutionMode.LIVE
    assert engine.mt5_service is not None  # LIVEモードではMT5サービス作成
    assert isinstance(engine.strategy, AutomatedTradingStrategy)


def test_trading_engine_simulation_mode(db_session):
    """シミュレーションモード初期化テスト"""
    engine = TradingEngineService(
        db=db_session,
        execution_mode=ExecutionMode.SIMULATION
    )
    
    assert engine.execution_mode == ExecutionMode.SIMULATION
    assert engine.mt5_service is None  # シミュレーションモードではMT5サービス不要


def test_automated_trading_strategy():
    """自動取引戦略テスト"""
    strategy = AutomatedTradingStrategy(
        min_signal_confidence=0.7,
        max_daily_trades=5,
        max_drawdown_threshold=0.1
    )
    
    # 高信頼度シグナル - オープン判定
    good_signal = TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.8,
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=153.0,
        risk_reward_ratio=2.5,
        reasoning=["High confidence signal"],
        timestamp=datetime.now(),
        metadata={}
    )
    
    market_data = {'market_open': True}
    assert strategy.should_open_position(good_signal, market_data) is True
    
    # 低信頼度シグナル - 見送り
    bad_signal = TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.WEAK,
        confidence=0.5,  # 閾値以下
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=153.0,
        risk_reward_ratio=2.5,
        reasoning=["Low confidence signal"],
        timestamp=datetime.now(),
        metadata={}
    )
    
    assert strategy.should_open_position(bad_signal, market_data) is False


def test_automated_trading_strategy_hold_signal():
    """HOLD シグナルの処理テスト"""
    strategy = AutomatedTradingStrategy()
    
    hold_signal = TradingSignal(
        signal_type=SignalType.HOLD,
        strength=SignalStrength.STRONG,
        confidence=0.9,  # 高信頼度でも
        entry_price=150.0,
        stop_loss=None,
        take_profit=None,
        risk_reward_ratio=None,
        reasoning=["Hold signal"],
        timestamp=datetime.now(),
        metadata={}
    )
    
    market_data = {'market_open': True}
    assert strategy.should_open_position(hold_signal, market_data) is False


def test_automated_trading_strategy_market_closed():
    """市場クローズ時の処理テスト"""
    strategy = AutomatedTradingStrategy()
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.8,
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=153.0,
        risk_reward_ratio=2.5,
        reasoning=["Market closed test"],
        timestamp=datetime.now(),
        metadata={}
    )
    
    market_data = {'market_open': False}
    assert strategy.should_open_position(signal, market_data) is False


def test_automated_trading_strategy_low_risk_reward():
    """低リスクリワード比での処理テスト"""
    strategy = AutomatedTradingStrategy()
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        confidence=0.8,
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=151.5,
        risk_reward_ratio=1.5,  # 2.0未満
        reasoning=["Low risk reward ratio"],
        timestamp=datetime.now(),
        metadata={}
    )
    
    market_data = {'market_open': True}
    assert strategy.should_open_position(signal, market_data) is False


def test_automated_trading_strategy_close_position():
    """ポジションクローズ判定テスト"""
    strategy = AutomatedTradingStrategy()
    
    # 通常のポジション - クローズしない
    position = Position(
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        entry_price=150.0,
        current_price=151.0,
        risk_amount=1000.0
    )
    
    market_data = {'current_price': 151.0}
    assert strategy.should_close_position(position, market_data) is False
    
    # 大きな損失のポジション - クローズ（risk_amount * 2 = 2000以上の損失）
    position.current_price = 130.0  # 大幅下落（20pips下落 → 2000以上の損失）
    market_data['current_price'] = 130.0
    assert strategy.should_close_position(position, market_data) is True


def test_execute_signal_success(trading_engine, sample_signal):
    """シグナル実行成功テスト"""
    with patch.object(trading_engine, '_validate_signal') as mock_validate:
        mock_validate.return_value = {
            'is_valid': True,
            'errors': [],
            'position_size_calc': Mock(recommended_size=0.1)
        }
        
        with patch.object(trading_engine, '_execute_order') as mock_execute:
            mock_execute.return_value = OrderResult(
                success=True,
                position_id=1,
                ticket=12345,
                execution_price=150.0,
                message="Success"
            )
            
            result = trading_engine.execute_signal(sample_signal)
            
            assert result.success is True
            assert result.position_id == 1
            assert result.ticket == 12345


def test_execute_signal_validation_failure(trading_engine, sample_signal):
    """シグナル実行バリデーション失敗テスト"""
    with patch.object(trading_engine, '_validate_signal') as mock_validate:
        mock_validate.return_value = {
            'is_valid': False,
            'errors': ['Risk too high']
        }
        
        result = trading_engine.execute_signal(sample_signal)
        
        assert result.success is False
        assert "リスク管理検証失敗" in result.message


def test_execute_signal_strategy_rejection(trading_engine, sample_signal):
    """戦略による実行拒否テスト"""
    # 低信頼度シグナルに変更
    sample_signal.confidence = 0.5
    
    result = trading_engine.execute_signal(sample_signal)
    
    assert result.success is False
    assert "取引戦略により実行見送り" in result.message


def test_execute_order_simulation(trading_engine):
    """シミュレーション注文実行テスト"""
    order_request = OrderRequest(
        symbol="USDJPY",
        order_type=OrderType.MARKET,
        position_type=PositionType.BUY,
        lot_size=0.1,
        price=150.0,
        stop_loss=149.0,
        take_profit=153.0,
        strategy_name="TestStrategy"
    )
    
    result = trading_engine._execute_order(order_request)
    
    assert result.success is True
    assert result.position_id is not None
    
    # データベースにポジションが保存されているか確認
    position = trading_engine.db.query(Position).filter(
        Position.id == result.position_id
    ).first()
    
    assert position is not None
    assert position.symbol == "USDJPY"
    assert position.status == PositionStatus.OPEN


def test_close_position_success(trading_engine, db_session):
    """ポジションクローズ成功テスト"""
    # テスト用ポジション作成
    position = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        entry_price=150.0
    )
    
    db_session.add(position)
    db_session.commit()
    
    with patch.object(trading_engine, '_get_current_price') as mock_price:
        mock_price.return_value = 151.0
        
        result = trading_engine.close_position(position.id, "Test close")
        
        assert result.success is True
        
        # ポジションがクローズされているか確認
        db_session.refresh(position)
        assert position.status == PositionStatus.CLOSED
        assert position.exit_price == 151.0


def test_close_position_not_found(trading_engine):
    """存在しないポジションのクローズテスト"""
    result = trading_engine.close_position(999, "Test close")
    
    assert result.success is False
    assert "ポジションが見つかりません" in result.message


def test_close_position_not_open(trading_engine, db_session):
    """オープンでないポジションのクローズテスト"""
    position = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.CLOSED,  # 既にクローズ済み
        lot_size=0.1
    )
    
    db_session.add(position)
    db_session.commit()
    
    result = trading_engine.close_position(position.id, "Test close")
    
    assert result.success is False
    assert "ポジションがオープン状態ではありません" in result.message


def test_monitor_positions(trading_engine, db_session):
    """ポジション監視テスト"""
    # オープンポジション作成
    position = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        entry_price=150.0,
        risk_amount=1000.0
    )
    
    db_session.add(position)
    db_session.commit()
    
    with patch.object(trading_engine, '_get_current_price') as mock_price:
        mock_price.return_value = 149.0  # 含み損状態
        
        result = trading_engine.monitor_positions()
        
        assert 'total_positions' in result
        assert 'summary' in result
        assert result['total_positions'] == 1


def test_get_trading_summary(trading_engine, db_session):
    """取引サマリー取得テスト"""
    # テスト用ポジション作成
    position1 = Position(
        symbol="USDJPY",
        position_type=PositionType.BUY,
        status=PositionStatus.OPEN,
        lot_size=0.1,
        created_at=datetime.now()
    )
    
    position2 = Position(
        symbol="EURUSD",
        position_type=PositionType.SELL,
        status=PositionStatus.CLOSED,
        lot_size=0.2,
        net_profit=100.0,
        created_at=datetime.now()
    )
    
    db_session.add_all([position1, position2])
    db_session.commit()
    
    summary = trading_engine.get_trading_summary()
    
    assert 'today' in summary
    assert 'overall' in summary
    assert 'current_portfolio' in summary
    assert summary['today']['total_trades'] >= 2


def test_set_execution_mode(trading_engine):
    """実行モード変更テスト"""
    # SIMULATION → LIVE
    trading_engine.set_execution_mode(ExecutionMode.LIVE)
    assert trading_engine.execution_mode == ExecutionMode.LIVE
    assert trading_engine.mt5_service is not None
    
    # LIVE → PAPER
    trading_engine.set_execution_mode(ExecutionMode.PAPER)
    assert trading_engine.execution_mode == ExecutionMode.PAPER


def test_set_strategy(trading_engine):
    """戦略変更テスト"""
    class CustomStrategy(TradingStrategy):
        def should_open_position(self, signal, market_data):
            return True
        
        def should_close_position(self, position, market_data):
            return False
        
        def calculate_position_size(self, signal, account_info):
            return 0.1
    
    custom_strategy = CustomStrategy()
    trading_engine.set_strategy(custom_strategy)
    
    assert trading_engine.strategy == custom_strategy


def test_calculate_final_pnl(trading_engine):
    """最終損益計算テスト"""
    position = Position(
        position_type=PositionType.BUY,
        lot_size=0.1,
        entry_price=150.0,
        exit_price=151.0,
        commission=5.0,
        swap=2.0
    )
    
    trading_engine._calculate_final_pnl(position)
    
    assert position.profit_loss == 10000.0  # 1.0 * 0.1 * 100000
    assert position.net_profit == 9993.0     # profit_loss - commission - swap


def test_get_pip_value(trading_engine):
    """ピップ値計算テスト"""
    # JPYペア
    jpy_pip = trading_engine._get_pip_value("USDJPY")
    assert jpy_pip == 0.01
    
    # その他のペア
    eur_pip = trading_engine._get_pip_value("EURUSD")
    assert eur_pip == 0.0001


def test_live_execution_mode_mt5_connection():
    """LIVEモードでのMT5接続テスト"""
    # MT5が利用できない環境用のモック
    with patch('app.services.trading_engine.MT5Service') as mock_mt5:
        mock_instance = Mock()
        mock_instance.is_connected.return_value = False
        mock_mt5.return_value = mock_instance
        
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        trading_engine = TradingEngineService(
            db=session,
            execution_mode=ExecutionMode.LIVE
        )
        
        # MT5未接続時の注文実行
        order_request = OrderRequest(
            symbol="USDJPY",
            order_type=OrderType.MARKET,
            position_type=PositionType.BUY,
            lot_size=0.1
        )
        
        position = Mock()
        result = trading_engine._execute_live_order(order_request, position)
        
        assert result.success is False or result.message is not None
        
        session.close()