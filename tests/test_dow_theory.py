import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.dow_theory import (
    DowTheoryService, ClassicDowTheory, TrendDirection, 
    TrendAnalysis, SwingPoint
)


@pytest.fixture
def uptrend_data():
    """上昇トレンドのサンプルデータ"""
    dates = [datetime.now() - timedelta(days=i) for i in range(15, 0, -1)]
    
    # 明確な上昇トレンドパターン（鋸歯状で高値と安値が切り上がり）
    prices = [
        100, 102, 101, 104, 103,  # HL, HH, HL
        106, 105, 108, 107, 110,  # HH, HL, HH, HL, HH  
        109, 112, 111, 114, 113   # HL, HH, HL, HH, HL
    ]
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + 0.5
        low = price - 0.5
        
        data.append({
            'datetime': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def downtrend_data():
    """下降トレンドのサンプルデータ"""
    dates = [datetime.now() - timedelta(days=i) for i in range(15, 0, -1)]
    
    # 明確な下降トレンドパターン（鋸歯状で高値と安値が切り下がり）
    prices = [
        120, 118, 119, 116, 117,  # LH, LL, LH
        114, 115, 112, 113, 110,  # LL, LH, LL, LH, LL
        111, 108, 109, 106, 107   # LH, LL, LH, LL, LH
    ]
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + 0.5
        low = price - 0.5
        
        data.append({
            'datetime': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sideways_data():
    """横這いトレンドのサンプルデータ"""
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # レンジ相場を作成
    base_price = 110.0
    data = []
    
    for i, date in enumerate(dates):
        # サイン波 + ノイズ
        price = base_price + 2 * np.sin(i * 0.3) + np.random.normal(0, 0.3)
        high = price + abs(np.random.normal(0, 0.3))
        low = price - abs(np.random.normal(0, 0.3))
        
        data.append({
            'datetime': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000
        })
    
    return pd.DataFrame(data)


def test_dow_theory_service_initialization():
    """DowTheoryServiceの初期化テスト"""
    service = DowTheoryService()
    assert service.strategy is not None
    assert isinstance(service.strategy, ClassicDowTheory)


def test_classic_dow_theory_uptrend(uptrend_data):
    """上昇トレンド検出のテスト"""
    strategy = ClassicDowTheory(zigzag_deviation=1.0)
    analysis = strategy.analyze_trend(uptrend_data)
    
    assert isinstance(analysis, TrendAnalysis)
    
    # 上昇トレンドが検出されるべき
    assert analysis.trend_direction in [TrendDirection.UPTREND, TrendDirection.SIDEWAYS]
    
    # スイングポイントが存在
    assert len(analysis.swing_points) > 0
    
    # 信頼度チェック
    assert 0 <= analysis.confidence <= 1
    
    # 高値更新または安値切り上げが存在すべき
    total_positive_swings = len(analysis.higher_highs) + len(analysis.higher_lows)
    assert total_positive_swings > 0


def test_classic_dow_theory_downtrend(downtrend_data):
    """下降トレンド検出のテスト"""
    strategy = ClassicDowTheory(zigzag_deviation=1.0)
    analysis = strategy.analyze_trend(downtrend_data)
    
    assert isinstance(analysis, TrendAnalysis)
    
    # 下降トレンドが検出されるべき
    assert analysis.trend_direction in [TrendDirection.DOWNTREND, TrendDirection.SIDEWAYS]
    
    # 安値更新または高値切り下げが存在すべき
    total_negative_swings = len(analysis.lower_highs) + len(analysis.lower_lows)
    assert total_negative_swings > 0


def test_classic_dow_theory_sideways(sideways_data):
    """横這いトレンド検出のテスト"""
    strategy = ClassicDowTheory(zigzag_deviation=3.0)
    analysis = strategy.analyze_trend(sideways_data)
    
    assert isinstance(analysis, TrendAnalysis)
    
    # 横這いまたは不明が検出されるべき
    assert analysis.trend_direction in [TrendDirection.SIDEWAYS, TrendDirection.UNKNOWN]


def test_insufficient_data():
    """データ不足時のテスト"""
    # 少量のデータ
    small_data = pd.DataFrame({
        'datetime': [datetime.now()],
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100],
        'volume': [1000]
    })
    
    strategy = ClassicDowTheory()
    analysis = strategy.analyze_trend(small_data)
    
    assert analysis.trend_direction == TrendDirection.UNKNOWN
    assert analysis.confidence == 0.0
    assert len(analysis.swing_points) == 0


def test_swing_point_creation():
    """スイングポイント作成のテスト"""
    # 明確なスイングを持つデータ
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=7),
        'open': [100, 102, 104, 102, 106, 104, 108],
        'high': [101, 103, 105, 103, 107, 105, 109],
        'low': [99, 101, 103, 101, 105, 103, 107],
        'close': [100, 102, 104, 102, 106, 104, 108],
        'volume': [1000] * 7
    })
    
    strategy = ClassicDowTheory(zigzag_deviation=1.0)
    analysis = strategy.analyze_trend(data)
    
    # スイングポイントが検出される
    assert len(analysis.swing_points) > 0
    
    # SwingPointオブジェクトの検証
    for swing in analysis.swing_points:
        assert isinstance(swing, SwingPoint)
        assert swing.swing_type in ['high', 'low']
        assert swing.price > 0
        assert isinstance(swing.datetime, pd.Timestamp)


def test_trend_confidence_calculation():
    """トレンド信頼度計算のテスト"""
    strategy = ClassicDowTheory()
    
    # モックデータでテスト
    categorized_swings = {
        'higher_highs': [SwingPoint(0, pd.Timestamp.now(), 100, 'high')],
        'higher_lows': [SwingPoint(1, pd.Timestamp.now(), 99, 'low')],
        'lower_highs': [],
        'lower_lows': []
    }
    
    # 上昇トレンドの信頼度
    confidence = strategy._calculate_confidence(categorized_swings, TrendDirection.UPTREND)
    assert confidence == 1.0  # 100%上昇トレンド
    
    # 混合パターン
    categorized_swings['lower_highs'].append(SwingPoint(2, pd.Timestamp.now(), 98, 'high'))
    confidence = strategy._calculate_confidence(categorized_swings, TrendDirection.UPTREND)
    assert confidence == 2/3  # 2/3が上昇


def test_dow_theory_service_methods():
    """DowTheoryServiceのメソッドテスト"""
    service = DowTheoryService()
    
    # 簡単なデータで分析
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10),
        'open': range(100, 110),
        'high': range(101, 111),
        'low': range(99, 109),
        'close': range(100, 110),
        'volume': [1000] * 10
    })
    
    analysis = service.analyze(data)
    assert isinstance(analysis, TrendAnalysis)
    
    # サマリー取得
    summary = service.get_trend_summary(analysis)
    assert isinstance(summary, dict)
    assert 'trend_direction' in summary
    assert 'confidence' in summary
    assert 'swing_points_count' in summary
    
    # 戦略変更
    new_strategy = ClassicDowTheory(zigzag_deviation=5.0)
    service.set_strategy(new_strategy)
    assert service.strategy.zigzag_deviation == 5.0


def test_empty_dataframe():
    """空のDataFrameのテスト"""
    empty_data = pd.DataFrame()
    
    strategy = ClassicDowTheory()
    analysis = strategy.analyze_trend(empty_data)
    
    assert analysis.trend_direction == TrendDirection.UNKNOWN
    assert analysis.confidence == 0.0


def test_zigzag_deviation_impact():
    """ZigZag偏差値の影響テスト"""
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=20),
        'open': [100 + i + 0.5*np.sin(i) for i in range(20)],
        'high': [101 + i + 0.5*np.sin(i) for i in range(20)],
        'low': [99 + i + 0.5*np.sin(i) for i in range(20)],
        'close': [100 + i + 0.5*np.sin(i) for i in range(20)],
        'volume': [1000] * 20
    })
    
    # 低い偏差値 -> より多くのスイング
    strategy_low = ClassicDowTheory(zigzag_deviation=1.0)
    analysis_low = strategy_low.analyze_trend(data)
    
    # 高い偏差値 -> より少ないスイング
    strategy_high = ClassicDowTheory(zigzag_deviation=10.0)
    analysis_high = strategy_high.analyze_trend(data)
    
    # 低い偏差値の方がスイングポイントが多いはず
    assert len(analysis_low.swing_points) >= len(analysis_high.swing_points)