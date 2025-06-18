import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from enum import Enum

from app.services.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestResult, BacktestStatus,
    BacktestMetrics, BacktestPosition
)
from app.services.trading_engine import ExecutionMode
from app.models.positions import PositionType
from app.models.price_data import PriceData, Base

# ExitReasonクラスを定義
class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    TIMEOUT = "timeout"


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
def mock_signal_generator():
    """モックシグナル生成器"""
    with patch('app.services.signal_generator.SignalGeneratorService') as mock:
        instance = mock.return_value
        instance.generate_signals.return_value = pd.DataFrame([
            {
                'datetime': datetime.now(),
                'signal_type': 'BUY',
                'confidence': 0.8,
                'entry_price': 150.0,
                'stop_loss': 149.0,
                'take_profit': 152.0
            }
        ])
        yield instance


@pytest.fixture
def backtest_engine(db_session, mock_signal_generator):
    """バックテストエンジン"""
    with patch('app.services.backtest_engine.SignalGeneratorService') as mock_sg:
        with patch('app.services.backtest_engine.RiskManagerService') as mock_rm:
            mock_sg.return_value = mock_signal_generator
            mock_rm.return_value = Mock()
            return BacktestEngine(db_session)


@pytest.fixture
def sample_config():
    """サンプル設定"""
    return BacktestConfig(
        symbol="USDJPY",
        timeframe="H1",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_balance=100000.0,
        risk_per_trade=0.02,
        commission_per_lot=5.0,
        spread_pips=2.0,
        min_signal_confidence=0.7,
        min_risk_reward=1.5,
        max_positions=3
    )


@pytest.fixture
def sample_price_data():
    """サンプル価格データ"""
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    data = []
    
    for i, date in enumerate(dates):
        price = 150.0 + (i % 20) * 0.1
        data.append({
            'symbol': 'USDJPY',
            'timeframe': 'H1',
            'datetime': date,
            'open': price,
            'high': price + 0.05,
            'low': price - 0.05,
            'close': price + 0.02,
            'volume': 1000 + i
        })
    
    return pd.DataFrame(data)


def test_backtest_engine_initialization(backtest_engine):
    """バックテストエンジン初期化テスト"""
    assert backtest_engine.db is not None
    assert backtest_engine.signal_generator is not None
    assert backtest_engine.risk_manager is not None
    assert backtest_engine.current_balance == 0.0
    assert backtest_engine.equity_history == []
    assert backtest_engine.positions == []
    assert backtest_engine.position_counter == 0


def test_backtest_config_validation(sample_config):
    """バックテスト設定検証テスト"""
    # 正常設定
    assert sample_config.symbol == "USDJPY"
    assert sample_config.initial_balance > 0
    assert 0 < sample_config.risk_per_trade < 1
    assert sample_config.start_date < sample_config.end_date
    
    # 異常設定テスト
    with pytest.raises(ValueError):
        BacktestConfig(
            symbol="INVALID",
            timeframe="H1",
            start_date=datetime.now(),
            end_date=datetime.now() - timedelta(days=1),  # 終了日が開始日より前
            initial_balance=100000.0
        )


def test_load_price_data(backtest_engine, sample_price_data, sample_config):
    """価格データ読み込みテスト"""
    # テストデータをDBに保存
    for _, row in sample_price_data.iterrows():
        price_data = PriceData(**row.to_dict())
        backtest_engine.db.add(price_data)
    backtest_engine.db.commit()
    
    # データ読み込み
    loaded_data = backtest_engine._load_price_data(sample_config)
    
    assert not loaded_data.empty
    assert len(loaded_data) > 0
    assert 'datetime' in loaded_data.columns
    assert loaded_data['symbol'].iloc[0] == sample_config.symbol
    assert loaded_data['timeframe'].iloc[0] == sample_config.timeframe


def test_load_price_data_no_data(backtest_engine, sample_config):
    """価格データなし読み込みテスト"""
    loaded_data = backtest_engine._load_price_data(sample_config)
    assert loaded_data.empty


def test_validate_data_sufficiency(backtest_engine, sample_price_data, sample_config):
    """データ充足性検証テスト"""
    # 十分なデータ
    is_sufficient = backtest_engine._validate_data_sufficiency(sample_price_data, sample_config)
    assert is_sufficient
    
    # 不十分なデータ
    insufficient_data = sample_price_data.head(5)  # 5行のみ
    is_sufficient = backtest_engine._validate_data_sufficiency(insufficient_data, sample_config)
    assert not is_sufficient


def test_calculate_position_size(backtest_engine, sample_config):
    """ポジションサイズ計算テスト"""
    current_balance = 100000.0
    entry_price = 150.0
    stop_loss = 149.0
    
    position_size = backtest_engine._calculate_position_size(
        current_balance, entry_price, stop_loss, sample_config.risk_per_trade
    )
    
    assert position_size > 0
    assert position_size <= current_balance * sample_config.risk_per_trade


def test_calculate_position_size_edge_cases(backtest_engine, sample_config):
    """ポジションサイズ計算エッジケース"""
    # ストップロスが0の場合
    position_size = backtest_engine._calculate_position_size(
        100000.0, 150.0, 0.0, sample_config.risk_per_trade
    )
    assert position_size == 0
    
    # エントリー価格とストップロスが同じ場合
    position_size = backtest_engine._calculate_position_size(
        100000.0, 150.0, 150.0, sample_config.risk_per_trade
    )
    assert position_size == 0


def test_execute_trade_simulation_mode(backtest_engine, sample_config):
    """取引実行テスト - シミュレーションモード"""
    signal = {
        'datetime': datetime.now(),
        'signal_type': 'BUY',
        'confidence': 0.8,
        'entry_price': 150.0,
        'stop_loss': 149.0,
        'take_profit': 152.0
    }
    
    current_balance = 100000.0
    position = backtest_engine._execute_trade(signal, current_balance, sample_config)
    
    assert position is not None
    assert position.symbol == sample_config.symbol
    assert position.position_type == PositionType.BUY
    assert position.entry_price == signal['entry_price']
    assert position.stop_loss == signal['stop_loss']
    assert position.take_profit == signal['take_profit']
    assert position.lot_size > 0


def test_execute_trade_insufficient_confidence(backtest_engine, sample_config):
    """取引実行テスト - 信頼度不足"""
    signal = {
        'datetime': datetime.now(),
        'signal_type': 'BUY',
        'confidence': 0.5,  # 設定の最小信頼度(0.7)未満
        'entry_price': 150.0,
        'stop_loss': 149.0,
        'take_profit': 152.0
    }
    
    current_balance = 100000.0
    position = backtest_engine._execute_trade(signal, current_balance, sample_config)
    
    assert position is None  # 信頼度不足で取引実行されない


def test_execute_trade_poor_risk_reward(backtest_engine, sample_config):
    """取引実行テスト - リスクリワード不足"""
    signal = {
        'datetime': datetime.now(),
        'signal_type': 'BUY',
        'confidence': 0.8,
        'entry_price': 150.0,
        'stop_loss': 149.0,
        'take_profit': 150.5  # リスクリワード比が低い
    }
    
    current_balance = 100000.0
    position = backtest_engine._execute_trade(signal, current_balance, sample_config)
    
    assert position is None  # リスクリワード不足で取引実行されない


def test_update_positions_stop_loss(backtest_engine, sample_config):
    """ポジション更新テスト - ストップロス"""
    position = BacktestPosition(
        id="test_1",
        symbol="USDJPY",
        position_type=PositionType.BUY,
        entry_time=datetime.now() - timedelta(hours=1),
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=152.0,
        lot_size=10000
    )
    
    current_price = {
        'datetime': datetime.now(),
        'open': 148.5,
        'high': 149.2,
        'low': 148.0,
        'close': 148.8
    }
    
    updated_position = backtest_engine._update_position(position, current_price, sample_config)
    
    assert updated_position.exit_time is not None
    assert updated_position.exit_reason == ExitReason.STOP_LOSS
    assert updated_position.exit_price <= position.stop_loss


def test_update_positions_take_profit(backtest_engine, sample_config):
    """ポジション更新テスト - テイクプロフィット"""
    position = BacktestPosition(
        id="test_1",
        symbol="USDJPY",
        position_type=PositionType.BUY,
        entry_time=datetime.now() - timedelta(hours=1),
        entry_price=150.0,
        stop_loss=149.0,
        take_profit=152.0,
        lot_size=10000
    )
    
    current_price = {
        'datetime': datetime.now(),
        'open': 151.8,
        'high': 152.5,
        'low': 151.5,
        'close': 152.2
    }
    
    updated_position = backtest_engine._update_position(position, current_price, sample_config)
    
    assert updated_position.exit_time is not None
    assert updated_position.exit_reason == ExitReason.TAKE_PROFIT
    assert updated_position.exit_price >= position.take_profit


def test_calculate_trade_metrics(backtest_engine):
    """取引メトリクス計算テスト"""
    positions = [
        BacktestPosition(
            id="test_1",
            symbol="USDJPY",
            position_type=PositionType.BUY,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now() - timedelta(hours=1),
            entry_price=150.0,
            exit_price=151.0,
            lot_size=10000,
            net_profit=100.0,
                    ),
        BacktestPosition(
            id="test_2",
            symbol="USDJPY",
            position_type=PositionType.SELL,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=149.5,
            lot_size=10000,
            net_profit=50.0,
                    )
    ]
    
    initial_balance = 100000.0
    final_balance = 100150.0
    
    metrics = backtest_engine._calculate_metrics(positions, initial_balance, final_balance)
    
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 0
    assert metrics.win_rate == 1.0
    assert metrics.total_profit == 150.0
    assert metrics.gross_profit == 150.0
    assert metrics.gross_loss == 0.0
    assert metrics.profit_factor > 0


def test_calculate_trade_metrics_with_losses(backtest_engine):
    """取引メトリクス計算テスト - 損失含む"""
    positions = [
        BacktestPosition(
            id="test_1",
            symbol="USDJPY",
            position_type=PositionType.BUY,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now() - timedelta(hours=1),
            entry_price=150.0,
            exit_price=151.0,
            lot_size=10000,
            net_profit=100.0,
                    ),
        BacktestPosition(
            id="test_2",
            symbol="USDJPY",
            position_type=PositionType.BUY,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=149.0,
            lot_size=10000,
            net_profit=-100.0,
                    )
    ]
    
    initial_balance = 100000.0
    final_balance = 100000.0
    
    metrics = backtest_engine._calculate_metrics(positions, initial_balance, final_balance)
    
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 0.5
    assert metrics.total_profit == 0.0
    assert metrics.gross_profit == 100.0
    assert metrics.gross_loss == -100.0
    assert metrics.profit_factor == 1.0


def test_generate_equity_curve(backtest_engine):
    """エクイティカーブ生成テスト"""
    positions = [
        BacktestPosition(
            id="test_1",
            symbol="USDJPY",
            position_type=PositionType.BUY,
            exit_time=datetime.now() - timedelta(hours=2),
            net_profit=100.0,
                    ),
        BacktestPosition(
            id="test_2",
            symbol="USDJPY",
            position_type=PositionType.BUY,
            exit_time=datetime.now() - timedelta(hours=1),
            net_profit=-50.0,
                    )
    ]
    
    initial_balance = 100000.0
    equity_curve = backtest_engine._generate_equity_curve(positions, initial_balance)
    
    assert isinstance(equity_curve, pd.DataFrame)
    assert not equity_curve.empty
    assert 'datetime' in equity_curve.columns
    assert 'balance' in equity_curve.columns
    assert len(equity_curve) == len(positions) + 1  # 初期残高 + 各取引後


def test_run_backtest_success(backtest_engine, sample_config, sample_price_data):
    """バックテスト実行成功テスト"""
    # テストデータをDBに保存
    for _, row in sample_price_data.iterrows():
        price_data = PriceData(**row.to_dict())
        backtest_engine.db.add(price_data)
    backtest_engine.db.commit()
    
    # バックテスト実行
    result = backtest_engine.run_backtest(sample_config)
    
    assert isinstance(result, BacktestResult)
    assert result.config == sample_config
    assert result.status == BacktestStatus.COMPLETED
    assert result.metrics is not None
    assert result.execution_time > 0
    assert result.positions is not None
    assert result.equity_curve is not None


def test_run_backtest_no_data(backtest_engine, sample_config):
    """バックテスト実行 - データなし"""
    result = backtest_engine.run_backtest(sample_config)
    
    assert result.status == BacktestStatus.FAILED
    assert "データが不足しています" in result.error_message


def test_run_backtest_invalid_config(backtest_engine):
    """バックテスト実行 - 無効設定"""
    invalid_config = BacktestConfig(
        symbol="INVALID",
        timeframe="H1",
        start_date=datetime.now(),
        end_date=datetime.now() - timedelta(days=1),  # 無効な日付範囲
        initial_balance=100000.0
    )
    
    result = backtest_engine.run_backtest(invalid_config)
    
    assert result.status == BacktestStatus.FAILED
    assert result.error_message is not None


def test_run_optimization(backtest_engine, sample_config, sample_price_data):
    """最適化実行テスト"""
    # テストデータをDBに保存
    for _, row in sample_price_data.iterrows():
        price_data = PriceData(**row.to_dict())
        backtest_engine.db.add(price_data)
    backtest_engine.db.commit()
    
    # パラメータ範囲
    parameter_ranges = {
        'risk_per_trade': [0.01, 0.02, 0.03],
        'min_signal_confidence': [0.6, 0.7, 0.8]
    }
    
    # 最適化実行（簡易版）
    results = backtest_engine.run_optimization(sample_config, parameter_ranges)
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    for result in results:
        assert isinstance(result, BacktestResult)
        assert result.status == BacktestStatus.COMPLETED


def test_position_manager_functions(backtest_engine):
    """ポジション管理機能テスト"""
    position = BacktestPosition(
        id="test_1",
        symbol="USDJPY",
        position_type=PositionType.BUY,
        entry_time=datetime.now(),
        entry_price=150.0,
        lot_size=10000
    )
    
    # ポジション追加
    backtest_engine.position_manager.add_position(position)
    open_positions = backtest_engine.position_manager.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].id == position.id
    
    # ポジション数確認
    position_count = backtest_engine.position_manager.get_position_count()
    assert position_count == 1
    
    # ポジションクローズ
    position.exit_time = datetime.now()
    position.exit_time = datetime.now()
    position.exit_price = 151.0
    
    backtest_engine.position_manager.close_position(position)
    open_positions = backtest_engine.position_manager.get_open_positions()
    assert len(open_positions) == 0


def test_execution_modes(backtest_engine, sample_config, sample_price_data):
    """実行モードテスト"""
    # テストデータをDBに保存
    for _, row in sample_price_data.iterrows():
        price_data = PriceData(**row.to_dict())
        backtest_engine.db.add(price_data)
    backtest_engine.db.commit()
    
    # シミュレーションモード
    sample_config.execution_mode = ExecutionMode.SIMULATION
    result_sim = backtest_engine.run_backtest(sample_config)
    assert result_sim.status == BacktestStatus.COMPLETED
    
    # ペーパートレードモード
    sample_config.execution_mode = ExecutionMode.PAPER
    result_paper = backtest_engine.run_backtest(sample_config)
    assert result_paper.status == BacktestStatus.COMPLETED
    
    # 結果の基本的な一貫性確認
    if result_sim.metrics.total_trades > 0 and result_paper.metrics.total_trades > 0:
        # 同じデータ・設定なので基本的な結果は似ているはず
        assert abs(result_sim.metrics.total_trades - result_paper.metrics.total_trades) <= 5


def test_error_handling_edge_cases(backtest_engine, sample_config):
    """エラーハンドリング・エッジケース"""
    # 空の価格データ
    empty_data = pd.DataFrame()
    is_sufficient = backtest_engine._validate_data_sufficiency(empty_data, sample_config)
    assert not is_sufficient
    
    # 不正な価格データ
    invalid_data = pd.DataFrame([{'invalid': 'data'}])
    is_sufficient = backtest_engine._validate_data_sufficiency(invalid_data, sample_config)
    assert not is_sufficient


def test_performance_under_load(backtest_engine, sample_config):
    """負荷下でのパフォーマンステスト"""
    # 大量の価格データ作成（1000件）
    dates = [datetime.now() - timedelta(hours=i) for i in range(1000, 0, -1)]
    large_data = []
    
    for i, date in enumerate(dates):
        price = 150.0 + (i % 100) * 0.01
        large_data.append({
            'symbol': 'USDJPY',
            'timeframe': 'H1',
            'datetime': date,
            'open': price,
            'high': price + 0.05,
            'low': price - 0.05,
            'close': price + 0.02,
            'volume': 1000 + i
        })
    
    # テストデータをDBに保存
    for data_point in large_data:
        price_data = PriceData(**data_point)
        backtest_engine.db.add(price_data)
    backtest_engine.db.commit()
    
    # バックテスト実行
    result = backtest_engine.run_backtest(sample_config)
    
    assert result.status == BacktestStatus.COMPLETED
    assert result.execution_time > 0
    assert result.metrics is not None