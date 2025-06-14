import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services.signal_generator import (
    SignalGeneratorService, DowElliottCombinedStrategy,
    SignalType, SignalStrength, TradingSignal
)
from app.services.dow_theory import TrendDirection, TrendAnalysis, SwingPoint
from app.services.elliott_wave import WavePattern, WaveType, WaveLabel, ElliottWave


@pytest.fixture
def sample_price_data():
    """サンプル価格データ"""
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    np.random.seed(42)
    
    data = []
    base_price = 110.0
    
    for i, date in enumerate(dates):
        # トレンド + ノイズ
        trend = 0.1 * i
        noise = np.random.normal(0, 0.5)
        
        close = base_price + trend + noise
        high = close + abs(np.random.normal(0, 0.3))
        low = close - abs(np.random.normal(0, 0.3))
        open_price = low + (high - low) * np.random.random()
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(np.random.normal(1000, 200))
        })
    
    return pd.DataFrame(data)


def test_signal_generator_service_initialization():
    """SignalGeneratorServiceの初期化テスト"""
    service = SignalGeneratorService()
    assert service.strategy is not None
    assert isinstance(service.strategy, DowElliottCombinedStrategy)


def test_dow_elliott_combined_strategy_initialization():
    """DowElliottCombinedStrategyの初期化テスト"""
    strategy = DowElliottCombinedStrategy(
        min_confidence=0.7,
        risk_reward_min=3.0
    )
    
    assert strategy.min_confidence == 0.7
    assert strategy.risk_reward_min == 3.0
    assert strategy.dow_service is not None
    assert strategy.elliott_service is not None
    assert strategy.rsi_indicator is not None


def test_generate_signal_with_sample_data(sample_price_data):
    """サンプルデータでのシグナル生成テスト"""
    strategy = DowElliottCombinedStrategy()
    signal = strategy.generate_signal(sample_price_data)
    
    assert isinstance(signal, TradingSignal)
    assert signal.signal_type in [e for e in SignalType]
    assert signal.strength in [e for e in SignalStrength]
    assert 0 <= signal.confidence <= 1
    assert isinstance(signal.reasoning, list)
    assert len(signal.reasoning) > 0
    assert signal.timestamp is not None
    assert isinstance(signal.metadata, dict)


def test_insufficient_data_signal():
    """データ不足時のシグナル生成テスト"""
    # 少量のデータ
    small_data = pd.DataFrame({
        'datetime': [datetime.now()],
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100],
        'volume': [1000]
    })
    
    strategy = DowElliottCombinedStrategy()
    signal = strategy.generate_signal(small_data)
    
    assert signal.signal_type == SignalType.HOLD
    assert signal.strength == SignalStrength.WEAK
    assert signal.confidence == 0.0
    assert "データ不足" in signal.reasoning[0]


@patch('app.services.signal_generator.DowTheoryService')
@patch('app.services.signal_generator.ElliottWaveService')
def test_buy_signal_generation(mock_elliott_service, mock_dow_service, sample_price_data):
    """買いシグナル生成のテスト"""
    # ダウ理論のモック - 上昇トレンド
    mock_dow_analysis = Mock(spec=TrendAnalysis)
    mock_dow_analysis.trend_direction = TrendDirection.UPTREND
    mock_dow_analysis.confidence = 0.8
    mock_dow_analysis.swing_points = [
        SwingPoint(0, pd.Timestamp.now(), 100, 'low'),
        SwingPoint(1, pd.Timestamp.now(), 110, 'high')
    ]
    mock_dow_analysis.higher_highs = [SwingPoint(1, pd.Timestamp.now(), 110, 'high')]
    mock_dow_analysis.higher_lows = [SwingPoint(2, pd.Timestamp.now(), 105, 'low')]
    mock_dow_analysis.lower_highs = []
    mock_dow_analysis.lower_lows = []
    
    mock_dow_service.return_value.analyze.return_value = mock_dow_analysis
    
    # エリオット波動のモック - 第3波エントリーポイント
    mock_elliott_wave = Mock(spec=ElliottWave)
    mock_elliott_wave.wave_label = WaveLabel.WAVE_2
    mock_elliott_wave.confidence = 0.8
    
    mock_elliott_pattern = Mock(spec=WavePattern)
    mock_elliott_pattern.pattern_type = WaveType.IMPULSE
    mock_elliott_pattern.waves = [mock_elliott_wave]
    mock_elliott_pattern.completion_percentage = 80
    mock_elliott_pattern.invalidation_level = 100
    
    mock_elliott_service.return_value.analyze.return_value = mock_elliott_pattern
    
    strategy = DowElliottCombinedStrategy()
    signal = strategy.generate_signal(sample_price_data)
    
    # 買いシグナルが生成されるべき
    assert signal.signal_type == SignalType.BUY
    assert signal.confidence > 0.6
    assert signal.stop_loss is not None
    assert signal.take_profit is not None


@patch('app.services.signal_generator.DowTheoryService')
@patch('app.services.signal_generator.ElliottWaveService')
def test_sell_signal_generation(mock_elliott_service, mock_dow_service, sample_price_data):
    """売りシグナル生成のテスト"""
    # ダウ理論のモック - 下降トレンド
    mock_dow_analysis = Mock(spec=TrendAnalysis)
    mock_dow_analysis.trend_direction = TrendDirection.DOWNTREND
    mock_dow_analysis.confidence = 0.8
    mock_dow_analysis.swing_points = [
        SwingPoint(0, pd.Timestamp.now(), 120, 'high'),
        SwingPoint(1, pd.Timestamp.now(), 110, 'low')
    ]
    mock_dow_analysis.higher_highs = []
    mock_dow_analysis.higher_lows = []
    mock_dow_analysis.lower_highs = [SwingPoint(1, pd.Timestamp.now(), 115, 'high')]
    mock_dow_analysis.lower_lows = [SwingPoint(2, pd.Timestamp.now(), 105, 'low')]
    
    mock_dow_service.return_value.analyze.return_value = mock_dow_analysis
    
    # エリオット波動のモック - 第5波完了
    mock_elliott_wave = Mock(spec=ElliottWave)
    mock_elliott_wave.wave_label = WaveLabel.WAVE_5
    mock_elliott_wave.confidence = 0.8
    
    mock_elliott_pattern = Mock(spec=WavePattern)
    mock_elliott_pattern.pattern_type = WaveType.IMPULSE
    mock_elliott_pattern.waves = [mock_elliott_wave]
    mock_elliott_pattern.completion_percentage = 95
    mock_elliott_pattern.invalidation_level = 120
    
    mock_elliott_service.return_value.analyze.return_value = mock_elliott_pattern
    
    strategy = DowElliottCombinedStrategy()
    signal = strategy.generate_signal(sample_price_data)
    
    # 売りシグナルが生成されるべき
    assert signal.signal_type == SignalType.SELL
    assert signal.confidence > 0.6


def test_rsi_signal_evaluation():
    """RSIシグナル評価のテスト"""
    strategy = DowElliottCombinedStrategy()
    
    # 過買い状態
    rsi_signal = strategy._evaluate_rsi_signal(75)
    assert rsi_signal['type'] == SignalType.SELL
    assert "過買い" in rsi_signal['reasoning'][0]
    
    # 過売り状態
    rsi_signal = strategy._evaluate_rsi_signal(25)
    assert rsi_signal['type'] == SignalType.BUY
    assert "過売り" in rsi_signal['reasoning'][0]
    
    # 中立状態
    rsi_signal = strategy._evaluate_rsi_signal(50)
    assert rsi_signal['type'] == SignalType.HOLD
    assert "中立" in rsi_signal['reasoning'][0]


def test_trade_levels_calculation():
    """取引レベル計算のテスト"""
    strategy = DowElliottCombinedStrategy(risk_reward_min=2.0)
    
    current_price = 110.0
    
    # エリオット波動のモック
    mock_elliott_pattern = Mock(spec=WavePattern)
    mock_elliott_pattern.invalidation_level = 105.0
    
    # ダウ理論のモック
    mock_dow_analysis = Mock(spec=TrendAnalysis)
    mock_dow_analysis.swing_points = [
        SwingPoint(0, pd.Timestamp.now(), 108, 'low'),
        SwingPoint(1, pd.Timestamp.now(), 112, 'high')
    ]
    
    # 買いシグナルの場合
    entry, stop, target, rr = strategy._calculate_trade_levels(
        SignalType.BUY, current_price, mock_elliott_pattern, mock_dow_analysis
    )
    
    assert entry == current_price
    assert stop is not None
    assert stop < current_price  # 買いの場合、ストップは下
    assert target is not None
    assert target > current_price  # 買いの場合、ターゲットは上
    assert rr is not None
    assert rr >= 2.0  # 最小リスクリワード比


def test_signal_summary():
    """シグナルサマリーのテスト"""
    service = SignalGeneratorService()
    
    # サンプルシグナル作成
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.85,
        entry_price=110.0,
        stop_loss=105.0,
        take_profit=120.0,
        risk_reward_ratio=2.0,
        reasoning=["テスト理由"],
        timestamp=datetime.now(),
        metadata={"test": "data"}
    )
    
    summary = service.get_signal_summary(signal)
    
    assert isinstance(summary, dict)
    assert summary['signal_type'] == 'buy'
    assert summary['strength'] == 'strong'
    assert summary['confidence'] == 0.85
    assert summary['entry_price'] == 110.0
    assert summary['stop_loss'] == 105.0
    assert summary['take_profit'] == 120.0
    assert summary['risk_reward_ratio'] == 2.0
    assert 'reasoning' in summary
    assert 'timestamp' in summary
    assert 'metadata' in summary


def test_strategy_change():
    """戦略変更のテスト"""
    service = SignalGeneratorService()
    
    # 新しい戦略
    new_strategy = DowElliottCombinedStrategy(min_confidence=0.8)
    service.set_strategy(new_strategy)
    
    assert service.strategy.min_confidence == 0.8


def test_hold_signal_creation():
    """ホールドシグナル作成のテスト"""
    strategy = DowElliottCombinedStrategy()
    
    data = pd.DataFrame({
        'datetime': [datetime.now()],
        'close': [110.0],
        'high': [111.0],
        'low': [109.0],
        'open': [110.0],
        'volume': [1000]
    })
    
    signal = strategy._create_hold_signal("テスト理由", data)
    
    assert signal.signal_type == SignalType.HOLD
    assert signal.strength == SignalStrength.WEAK
    assert signal.confidence == 0.0
    assert signal.entry_price == 110.0
    assert "テスト理由" in signal.reasoning


def test_signal_strength_determination():
    """シグナル強度判定のテスト"""
    strategy = DowElliottCombinedStrategy()
    
    # モックデータ作成
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=60),
        'open': range(100, 160),
        'high': range(101, 161),
        'low': range(99, 159),
        'close': range(100, 160),
        'volume': [1000] * 60
    })
    
    with patch.object(strategy.dow_service, 'analyze') as mock_dow, \
         patch.object(strategy.elliott_service, 'analyze') as mock_elliott:
        
        # 高信頼度のモック
        mock_dow.return_value.trend_direction = TrendDirection.UPTREND
        mock_dow.return_value.confidence = 0.9
        mock_dow.return_value.swing_points = []
        mock_dow.return_value.higher_highs = []
        mock_dow.return_value.higher_lows = []
        mock_dow.return_value.lower_highs = []
        mock_dow.return_value.lower_lows = []
        
        mock_elliott_wave = Mock()
        mock_elliott_wave.wave_label = WaveLabel.WAVE_2
        mock_elliott_pattern = Mock()
        mock_elliott_pattern.pattern_type = WaveType.IMPULSE
        mock_elliott_pattern.waves = [mock_elliott_wave]
        mock_elliott_pattern.completion_percentage = 80
        mock_elliott_pattern.invalidation_level = 100
        mock_elliott.return_value = mock_elliott_pattern
        
        signal = strategy.generate_signal(data)
        
        # 高信頼度の場合は強いシグナル
        if signal.signal_type != SignalType.HOLD:
            assert signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]