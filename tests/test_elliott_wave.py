import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.elliott_wave import (
    ElliottWaveService, ClassicElliottWave, WaveType, WaveLabel,
    WavePattern, ElliottWave, FibonacciLevel
)
from app.services.dow_theory import SwingPoint


@pytest.fixture
def impulse_wave_data():
    """推進波のサンプルデータ（5波パターン）"""
    # 明確な5波パターンを作成
    prices = [
        100,  # 開始
        110,  # 1波頂点
        105,  # 2波底
        125,  # 3波頂点 (最長)
        115,  # 4波底
        135   # 5波頂点
    ]
    
    dates = [datetime.now() - timedelta(days=i) for i in range(len(prices)-1, -1, -1)]
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + 1
        low = price - 1
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
def corrective_wave_data():
    """修正波のサンプルデータ（ABC波）"""
    # ABC修正波パターン
    prices = [
        120,  # 開始
        110,  # A波底
        118,  # B波頂
        105   # C波底
    ]
    
    dates = [datetime.now() - timedelta(days=i) for i in range(len(prices)-1, -1, -1)]
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + 1
        low = price - 1
        data.append({
            'datetime': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000
        })
    
    return pd.DataFrame(data)


def test_elliott_wave_service_initialization():
    """ElliottWaveServiceの初期化テスト"""
    service = ElliottWaveService()
    assert service.strategy is not None
    assert isinstance(service.strategy, ClassicElliottWave)


def test_classic_elliott_wave_impulse_detection(impulse_wave_data):
    """推進波検出のテスト"""
    strategy = ClassicElliottWave(zigzag_deviation=3.0)
    pattern = strategy.analyze_waves(impulse_wave_data)
    
    assert isinstance(pattern, WavePattern)
    
    # パターンタイプチェック
    if pattern.pattern_type != WaveType.UNKNOWN:
        assert pattern.pattern_type in [WaveType.IMPULSE, WaveType.CORRECTIVE]
    
    # 波動が検出された場合
    if pattern.waves:
        assert len(pattern.waves) > 0
        
        # 各波動の検証
        for wave in pattern.waves:
            assert isinstance(wave, ElliottWave)
            assert isinstance(wave.start_point, SwingPoint)
            assert isinstance(wave.end_point, SwingPoint)
            assert wave.wave_label in [e for e in WaveLabel]
            assert wave.price_length > 0
            assert wave.time_length >= 0
            assert 0 <= wave.confidence <= 1


def test_classic_elliott_wave_corrective_detection(corrective_wave_data):
    """修正波検出のテスト"""
    strategy = ClassicElliottWave(zigzag_deviation=3.0)
    pattern = strategy.analyze_waves(corrective_wave_data)
    
    assert isinstance(pattern, WavePattern)
    
    # 修正波が検出された場合
    if pattern.pattern_type == WaveType.CORRECTIVE:
        assert len(pattern.waves) <= 3  # ABC波
        
        # ラベルチェック
        labels = [wave.wave_label for wave in pattern.waves]
        corrective_labels = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        for label in labels:
            assert label in corrective_labels


def test_fibonacci_level_calculation():
    """フィボナッチレベル計算のテスト"""
    strategy = ClassicElliottWave()
    
    start_point = SwingPoint(0, pd.Timestamp.now(), 100.0, 'low')
    end_point = SwingPoint(1, pd.Timestamp.now(), 110.0, 'high')
    
    # 2波（リトレースメント）のフィボナッチレベル
    fib_levels = strategy._calculate_fibonacci_levels(start_point, end_point, WaveLabel.WAVE_2)
    
    assert len(fib_levels) > 0
    
    for level in fib_levels:
        assert isinstance(level, FibonacciLevel)
        assert level.ratio > 0
        assert level.price > 0
        assert level.level_type in ['support', 'resistance', 'target']
    
    # 主要フィボナッチ比率の存在確認
    ratios = [level.ratio for level in fib_levels]
    assert 0.382 in ratios  # 38.2%
    assert 0.618 in ratios  # 61.8%


def test_impulse_wave_rules_validation():
    """推進波ルール検証のテスト"""
    strategy = ClassicElliottWave()
    
    # 正しい推進波パターン (上昇)
    valid_points = [
        SwingPoint(0, pd.Timestamp.now(), 100, 'low'),   # 開始
        SwingPoint(1, pd.Timestamp.now(), 110, 'high'),  # 1波頂点
        SwingPoint(2, pd.Timestamp.now(), 105, 'low'),   # 2波底 (100を下回らない)
        SwingPoint(3, pd.Timestamp.now(), 125, 'high'),  # 3波頂点 (最長)
        SwingPoint(4, pd.Timestamp.now(), 115, 'low'),   # 4波底 (110を下回らない)
        SwingPoint(5, pd.Timestamp.now(), 135, 'high')   # 5波頂点
    ]
    
    # 上昇推進波ルール検証
    is_valid = strategy._validate_impulse_rules(valid_points, 'up')
    assert is_valid
    
    # 無効なパターン: 2波が1波開始点を下回る
    invalid_points = valid_points.copy()
    invalid_points[2] = SwingPoint(2, pd.Timestamp.now(), 95, 'low')  # 100を下回る
    
    is_valid = strategy._validate_impulse_rules(invalid_points, 'up')
    assert not is_valid


def test_wave_pattern_creation():
    """波動パターン作成のテスト"""
    strategy = ClassicElliottWave()
    
    points = [
        SwingPoint(0, pd.Timestamp.now(), 100, 'low'),
        SwingPoint(1, pd.Timestamp.now(), 110, 'high'),
        SwingPoint(2, pd.Timestamp.now(), 105, 'low'),
        SwingPoint(3, pd.Timestamp.now(), 125, 'high'),
        SwingPoint(4, pd.Timestamp.now(), 115, 'low'),
        SwingPoint(5, pd.Timestamp.now(), 135, 'high')
    ]
    
    pattern = strategy._create_impulse_pattern(points, 'up')
    
    assert isinstance(pattern, WavePattern)
    assert pattern.pattern_type == WaveType.IMPULSE
    assert len(pattern.waves) == 5
    assert pattern.completion_percentage == 100.0
    
    # 波動ラベルの確認
    expected_labels = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, WaveLabel.WAVE_4, WaveLabel.WAVE_5]
    actual_labels = [wave.wave_label for wave in pattern.waves]
    assert actual_labels == expected_labels


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
    
    strategy = ClassicElliottWave()
    pattern = strategy.analyze_waves(small_data)
    
    assert pattern.pattern_type == WaveType.UNKNOWN
    assert len(pattern.waves) == 0
    assert pattern.completion_percentage == 0.0


def test_elliott_wave_service_methods():
    """ElliottWaveServiceのメソッドテスト"""
    service = ElliottWaveService()
    
    # 簡単なデータで分析
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10),
        'open': [100, 105, 102, 110, 108, 115, 112, 120, 118, 125],
        'high': [101, 106, 103, 111, 109, 116, 113, 121, 119, 126],
        'low': [99, 104, 101, 109, 107, 114, 111, 119, 117, 124],
        'close': [100, 105, 102, 110, 108, 115, 112, 120, 118, 125],
        'volume': [1000] * 10
    })
    
    pattern = service.analyze(data)
    assert isinstance(pattern, WavePattern)
    
    # サマリー取得
    summary = service.get_wave_summary(pattern)
    assert isinstance(summary, dict)
    assert 'pattern_type' in summary
    assert 'waves_count' in summary
    assert 'completion_percentage' in summary
    
    # 戦略変更
    new_strategy = ClassicElliottWave(zigzag_deviation=8.0)
    service.set_strategy(new_strategy)
    assert service.strategy.zigzag_deviation == 8.0


def test_fibonacci_ratios_configuration():
    """フィボナッチ比率設定のテスト"""
    strategy = ClassicElliottWave()
    
    # デフォルトのフィボナッチ比率確認
    assert 'retracement' in strategy.fib_ratios
    assert 'extension' in strategy.fib_ratios
    assert 'projection' in strategy.fib_ratios
    
    # 重要な比率が含まれているか
    assert 0.618 in strategy.fib_ratios['retracement']
    assert 1.618 in strategy.fib_ratios['extension']


def test_invalidation_level_calculation():
    """無効化レベル計算のテスト"""
    strategy = ClassicElliottWave()
    
    waves = [
        ElliottWave(
            start_point=SwingPoint(0, pd.Timestamp.now(), 100, 'low'),
            end_point=SwingPoint(1, pd.Timestamp.now(), 110, 'high'),
            wave_label=WaveLabel.WAVE_1,
            wave_type=WaveType.IMPULSE,
            price_length=10,
            time_length=1,
            fibonacci_levels=[],
            confidence=0.8
        )
    ]
    
    # 上昇波の無効化レベル
    invalidation = strategy._calculate_invalidation_level(waves, 'up')
    assert invalidation == 100  # 1波開始点
    
    # 下降波の無効化レベル
    invalidation = strategy._calculate_invalidation_level(waves, 'down')
    assert invalidation == 100  # 1波開始点


def test_next_target_calculation():
    """次のターゲット計算のテスト"""
    strategy = ClassicElliottWave()
    
    # 5波完了の場合
    wave5 = ElliottWave(
        start_point=SwingPoint(4, pd.Timestamp.now(), 115, 'low'),
        end_point=SwingPoint(5, pd.Timestamp.now(), 135, 'high'),
        wave_label=WaveLabel.WAVE_5,
        wave_type=WaveType.IMPULSE,
        price_length=20,
        time_length=1,
        fibonacci_levels=[],
        confidence=0.8
    )
    
    target = strategy._calculate_next_target(wave5, 'up')
    assert target is not None
    assert isinstance(target, FibonacciLevel)
    assert target.level_type == 'target'


def test_empty_pattern_creation():
    """空パターン作成のテスト"""
    strategy = ClassicElliottWave()
    empty_pattern = strategy._create_empty_pattern()
    
    assert empty_pattern.pattern_type == WaveType.UNKNOWN
    assert len(empty_pattern.waves) == 0
    assert empty_pattern.completion_percentage == 0.0
    assert empty_pattern.next_target is None
    assert empty_pattern.invalidation_level is None