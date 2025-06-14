import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class TechnicalIndicator(ABC):
    """テクニカル指標の基底クラス"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """指標を計算する"""
        pass


class ATRIndicator(TechnicalIndicator):
    """Average True Range (平均真の値幅)"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=self.period).mean()


class ZigZagIndicator(TechnicalIndicator):
    """ZigZagインジケーター（スイングポイント検出用）"""
    
    def __init__(self, deviation_pct: float = 5.0):
        self.deviation_pct = deviation_pct
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        high = data['high'].values
        low = data['low'].values
        
        peaks = self._find_peaks(high, low)
        return self._create_zigzag_dataframe(data, peaks)
    
    def _find_peaks(self, high: np.ndarray, low: np.ndarray) -> List[Tuple[int, float, str]]:
        peaks = []
        if len(high) < 3:
            return peaks
        
        # 最初のピークを決定（最初の高値から開始）
        peaks.append((0, high[0], 'high'))
        last_peak_idx = 0
        last_peak_value = high[0]
        last_peak_type = 'high'
        
        for i in range(1, len(high)):
            current_high = high[i]
            current_low = low[i]
            
            if last_peak_type == 'high':
                # 高値からの下降を探す
                if current_low < last_peak_value * (1 - self.deviation_pct / 100):
                    peaks.append((i, current_low, 'low'))
                    last_peak_idx = i
                    last_peak_value = current_low
                    last_peak_type = 'low'
                elif current_high > last_peak_value:
                    # より高い高値で更新
                    peaks[-1] = (i, current_high, 'high')
                    last_peak_value = current_high
                    last_peak_idx = i
            
            else:  # last_peak_type == 'low'
                # 安値からの上昇を探す
                if current_high > last_peak_value * (1 + self.deviation_pct / 100):
                    peaks.append((i, current_high, 'high'))
                    last_peak_idx = i
                    last_peak_value = current_high
                    last_peak_type = 'high'
                elif current_low < last_peak_value:
                    # より安い安値で更新
                    peaks[-1] = (i, current_low, 'low')
                    last_peak_value = current_low
                    last_peak_idx = i
        
        return peaks
    
    def _create_zigzag_dataframe(self, data: pd.DataFrame, peaks: List[Tuple[int, float, str]]) -> pd.DataFrame:
        result = data.copy()
        result['zigzag_high'] = np.nan
        result['zigzag_low'] = np.nan
        result['swing_type'] = None
        
        for idx, value, peak_type in peaks:
            if peak_type == 'high':
                result.iloc[idx, result.columns.get_loc('zigzag_high')] = value
                result.iloc[idx, result.columns.get_loc('swing_type')] = 'high'
            else:
                result.iloc[idx, result.columns.get_loc('zigzag_low')] = value
                result.iloc[idx, result.columns.get_loc('swing_type')] = 'low'
        
        return result


class MovingAverageIndicator(TechnicalIndicator):
    """移動平均線"""
    
    def __init__(self, period: int = 20, ma_type: str = 'sma'):
        self.period = period
        self.ma_type = ma_type
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        
        if self.ma_type == 'sma':
            return close.rolling(window=self.period).mean()
        elif self.ma_type == 'ema':
            return close.ewm(span=self.period).mean()
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")


class RSIIndicator(TechnicalIndicator):
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class TechnicalIndicatorFactory:
    """テクニカル指標のファクトリークラス"""
    
    _indicators = {
        'atr': ATRIndicator,
        'zigzag': ZigZagIndicator,
        'sma': lambda period: MovingAverageIndicator(period, 'sma'),
        'ema': lambda period: MovingAverageIndicator(period, 'ema'),
        'rsi': RSIIndicator
    }
    
    @classmethod
    def create(cls, indicator_type: str, **kwargs) -> TechnicalIndicator:
        if indicator_type not in cls._indicators:
            raise ValueError(f"Unknown indicator type: {indicator_type}")
        
        indicator_class = cls._indicators[indicator_type]
        return indicator_class(**kwargs)
    
    @classmethod
    def get_available_indicators(cls) -> List[str]:
        return list(cls._indicators.keys())