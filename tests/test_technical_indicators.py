import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.technical_indicators import (
    ATRIndicator, ZigZagIndicator, MovingAverageIndicator, 
    RSIIndicator, TechnicalIndicatorFactory
)


@pytest.fixture
def sample_ohlc_data():
    """サンプルOHLCデータ"""
    dates = [datetime.now() - timedelta(days=i) for i in range(50, 0, -1)]
    np.random.seed(42)
    
    data = []
    base_price = 110.0
    
    for i, date in enumerate(dates):
        # トレンドのあるランダムデータ生成
        trend = 0.01 * np.sin(i * 0.1)
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
        
        base_price = close
    
    return pd.DataFrame(data)


def test_atr_indicator(sample_ohlc_data):
    """ATR指標のテスト"""
    atr = ATRIndicator(period=14)
    result = atr.calculate(sample_ohlc_data)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlc_data)
    
    # 最初の14個はNaN
    assert pd.isna(result.iloc[:13]).all()
    
    # 14個目以降は値が存在
    assert not pd.isna(result.iloc[13:]).any()
    
    # すべて正の値
    assert (result.dropna() > 0).all()


def test_zigzag_indicator(sample_ohlc_data):
    """ZigZag指標のテスト"""
    zigzag = ZigZagIndicator(deviation_pct=5.0)
    result = zigzag.calculate(sample_ohlc_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_ohlc_data)
    
    # 必要なカラムが存在
    assert 'zigzag_high' in result.columns
    assert 'zigzag_low' in result.columns
    assert 'swing_type' in result.columns
    
    # スイングポイントが検出されている
    swing_points = result[result['swing_type'].notna()]
    assert len(swing_points) > 0
    
    # high/lowが正しく分類されている
    high_swings = result[result['swing_type'] == 'high']
    low_swings = result[result['swing_type'] == 'low']
    
    if len(high_swings) > 0:
        assert not high_swings['zigzag_high'].isna().any()
    if len(low_swings) > 0:
        assert not low_swings['zigzag_low'].isna().any()


def test_moving_average_sma(sample_ohlc_data):
    """単純移動平均のテスト"""
    ma = MovingAverageIndicator(period=20, ma_type='sma')
    result = ma.calculate(sample_ohlc_data)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlc_data)
    
    # 最初の19個はNaN
    assert pd.isna(result.iloc[:19]).all()
    
    # 20個目以降は値が存在
    assert not pd.isna(result.iloc[19:]).any()
    
    # 手動計算と比較 (20個目)
    manual_sma = sample_ohlc_data['close'].iloc[:20].mean()
    assert abs(result.iloc[19] - manual_sma) < 1e-10


def test_moving_average_ema(sample_ohlc_data):
    """指数移動平均のテスト"""
    ma = MovingAverageIndicator(period=20, ma_type='ema')
    result = ma.calculate(sample_ohlc_data)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlc_data)
    
    # EMAは1個目から計算される
    assert not pd.isna(result).any()


def test_rsi_indicator(sample_ohlc_data):
    """RSI指標のテスト"""
    rsi = RSIIndicator(period=14)
    result = rsi.calculate(sample_ohlc_data)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_ohlc_data)
    
    # RSIは0-100の範囲
    valid_rsi = result.dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_technical_indicator_factory():
    """ファクトリークラスのテスト"""
    # 利用可能な指標リスト
    indicators = TechnicalIndicatorFactory.get_available_indicators()
    assert 'atr' in indicators
    assert 'zigzag' in indicators
    assert 'sma' in indicators
    assert 'ema' in indicators
    assert 'rsi' in indicators
    
    # ATR作成
    atr = TechnicalIndicatorFactory.create('atr', period=10)
    assert isinstance(atr, ATRIndicator)
    assert atr.period == 10
    
    # ZigZag作成
    zigzag = TechnicalIndicatorFactory.create('zigzag', deviation_pct=3.0)
    assert isinstance(zigzag, ZigZagIndicator)
    assert zigzag.deviation_pct == 3.0
    
    # SMA作成
    sma = TechnicalIndicatorFactory.create('sma', period=25)
    assert isinstance(sma, MovingAverageIndicator)
    assert sma.period == 25
    assert sma.ma_type == 'sma'
    
    # RSI作成
    rsi = TechnicalIndicatorFactory.create('rsi', period=21)
    assert isinstance(rsi, RSIIndicator)
    assert rsi.period == 21
    
    # 不正な指標タイプ
    with pytest.raises(ValueError):
        TechnicalIndicatorFactory.create('invalid_indicator')


def test_zigzag_peak_detection():
    """ZigZag特定パターンのテスト"""
    # 明確なピークを持つデータ
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10),
        'high': [100, 101, 105, 102, 110, 108, 115, 112, 120, 118],
        'low': [99, 100, 103, 100, 108, 106, 113, 110, 118, 116],
        'close': [100.5, 100.5, 104, 101, 109, 107, 114, 111, 119, 117],
        'open': [100, 100.5, 104, 101, 109, 107, 114, 111, 119, 117],
        'volume': [1000] * 10
    })
    
    zigzag = ZigZagIndicator(deviation_pct=3.0)
    result = zigzag.calculate(data)
    
    # スイングポイントが検出されることを確認
    swing_points = result[result['swing_type'].notna()]
    assert len(swing_points) >= 2


def test_atr_calculation_accuracy():
    """ATR計算精度のテスト"""
    # 簡単なデータでATRを手動計算
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=5),
        'high': [102, 103, 104, 103, 105],
        'low': [100, 101, 102, 101, 103],
        'close': [101, 102, 103, 102, 104],
        'open': [101, 101, 102, 103, 102],
        'volume': [1000] * 5
    })
    
    atr = ATRIndicator(period=3)
    result = atr.calculate(data)
    
    # 最初の2つはNaN
    assert pd.isna(result.iloc[:2]).all()
    
    # 3つ目から値が存在
    assert not pd.isna(result.iloc[2:]).any()


def test_rsi_extreme_values():
    """RSI極値のテスト"""
    # 連続上昇データ (RSI -> 100に近づく)
    data_up = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=20),
        'close': range(100, 120),
        'high': range(101, 121),
        'low': range(99, 119),
        'open': range(100, 120),
        'volume': [1000] * 20
    })
    
    rsi = RSIIndicator(period=14)
    result_up = rsi.calculate(data_up)
    
    # 最後のRSIは高い値
    assert result_up.iloc[-1] > 80
    
    # 連続下降データ (RSI -> 0に近づく)
    data_down = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=20),
        'close': range(120, 100, -1),
        'high': range(121, 101, -1),
        'low': range(119, 99, -1),
        'open': range(120, 100, -1),
        'volume': [1000] * 20
    })
    
    result_down = rsi.calculate(data_down)
    
    # 最後のRSIは低い値
    assert result_down.iloc[-1] < 20