import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from app.services.technical_indicators import ZigZagIndicator
from app.services.dow_theory import SwingPoint


class WaveType(Enum):
    """波動タイプ"""
    IMPULSE = "impulse"  # 推進波
    CORRECTIVE = "corrective"  # 修正波
    UNKNOWN = "unknown"


class WaveLabel(Enum):
    """波動ラベル"""
    WAVE_1 = "1"
    WAVE_2 = "2"
    WAVE_3 = "3"
    WAVE_4 = "4"
    WAVE_5 = "5"
    WAVE_A = "A"
    WAVE_B = "B"
    WAVE_C = "C"
    UNKNOWN = "unknown"


@dataclass
class FibonacciLevel:
    """フィボナッチレベル"""
    ratio: float
    price: float
    level_type: str  # 'support', 'resistance', 'target'


@dataclass
class ElliottWave:
    """エリオット波動"""
    start_point: SwingPoint
    end_point: SwingPoint
    wave_label: WaveLabel
    wave_type: WaveType
    price_length: float
    time_length: int
    fibonacci_levels: List[FibonacciLevel]
    confidence: float


@dataclass
class WavePattern:
    """波動パターン"""
    waves: List[ElliottWave]
    pattern_type: WaveType
    completion_percentage: float
    next_target: Optional[FibonacciLevel]
    invalidation_level: Optional[float]


class ElliottWaveStrategy(ABC):
    """エリオット波動戦略の基底クラス"""
    
    @abstractmethod
    def analyze_waves(self, data: pd.DataFrame) -> WavePattern:
        """波動分析を実行"""
        pass


class ClassicElliottWave(ElliottWaveStrategy):
    """クラシックなエリオット波動実装"""
    
    def __init__(self, 
                 zigzag_deviation: float = 5.0,
                 fibonacci_tolerance: float = 0.1):
        self.zigzag_deviation = zigzag_deviation
        self.fibonacci_tolerance = fibonacci_tolerance
        self.zigzag = ZigZagIndicator(zigzag_deviation)
        
        # フィボナッチ比率
        self.fib_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.618],
            'projection': [0.618, 1.0, 1.618, 2.618]
        }
    
    def analyze_waves(self, data: pd.DataFrame) -> WavePattern:
        """メインの波動分析"""
        if len(data) < 20:
            return self._create_empty_pattern()
        
        # ZigZag計算とスイングポイント抽出
        zigzag_data = self.zigzag.calculate(data)
        swing_points = self._extract_swing_points(zigzag_data)
        
        if len(swing_points) < 5:
            return self._create_empty_pattern()
        
        # 推進波パターンを検索
        impulse_pattern = self._find_impulse_pattern(swing_points)
        if impulse_pattern:
            return impulse_pattern
        
        # 修正波パターンを検索
        corrective_pattern = self._find_corrective_pattern(swing_points)
        if corrective_pattern:
            return corrective_pattern
        
        return self._create_empty_pattern()
    
    def _extract_swing_points(self, zigzag_data: pd.DataFrame) -> List[SwingPoint]:
        """ZigZagデータからスイングポイントを抽出"""
        swing_points = []
        
        for idx, row in zigzag_data.iterrows():
            if pd.notna(row['zigzag_high']):
                swing_points.append(SwingPoint(
                    index=idx,
                    datetime=row['datetime'],
                    price=row['zigzag_high'],
                    swing_type='high',
                    volume=row.get('volume', 0)
                ))
            elif pd.notna(row['zigzag_low']):
                swing_points.append(SwingPoint(
                    index=idx,
                    datetime=row['datetime'],
                    price=row['zigzag_low'],
                    swing_type='low',
                    volume=row.get('volume', 0)
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def _find_impulse_pattern(self, swing_points: List[SwingPoint]) -> Optional[WavePattern]:
        """5波の推進波パターンを検索"""
        if len(swing_points) < 6:  # 推進波には最低6つのポイントが必要
            return None
        
        # 最新の6つのポイントで5波パターンをチェック
        recent_points = swing_points[-6:]
        
        # 上昇推進波の場合: 安値→高値→安値→高値→安値→高値
        if (recent_points[0].swing_type == 'low' and
            recent_points[1].swing_type == 'high' and
            recent_points[2].swing_type == 'low' and
            recent_points[3].swing_type == 'high' and
            recent_points[4].swing_type == 'low' and
            recent_points[5].swing_type == 'high'):
            
            if self._validate_impulse_rules(recent_points, direction='up'):
                return self._create_impulse_pattern(recent_points, direction='up')
        
        # 下降推進波の場合: 高値→安値→高値→安値→高値→安値
        elif (recent_points[0].swing_type == 'high' and
              recent_points[1].swing_type == 'low' and
              recent_points[2].swing_type == 'high' and
              recent_points[3].swing_type == 'low' and
              recent_points[4].swing_type == 'high' and
              recent_points[5].swing_type == 'low'):
            
            if self._validate_impulse_rules(recent_points, direction='down'):
                return self._create_impulse_pattern(recent_points, direction='down')
        
        return None
    
    def _validate_impulse_rules(self, points: List[SwingPoint], direction: str) -> bool:
        """推進波の基本ルールを検証"""
        if direction == 'up':
            # 波2は波1の開始点を下回らない
            if points[2].price <= points[0].price:
                return False
            
            # 波4は波1の価格領域に重複しない
            if points[4].price <= points[1].price:
                return False
            
            # 波3は最短波動ではない
            wave1_length = points[1].price - points[0].price
            wave3_length = points[3].price - points[2].price
            wave5_length = points[5].price - points[4].price
            
            if wave3_length <= wave1_length and wave3_length <= wave5_length:
                return False
        
        else:  # direction == 'down'
            # 波2は波1の開始点を上回らない
            if points[2].price >= points[0].price:
                return False
            
            # 波4は波1の価格領域に重複しない
            if points[4].price >= points[1].price:
                return False
            
            # 波3は最短波動ではない
            wave1_length = points[0].price - points[1].price
            wave3_length = points[2].price - points[3].price
            wave5_length = points[4].price - points[5].price
            
            if wave3_length <= wave1_length and wave3_length <= wave5_length:
                return False
        
        return True
    
    def _create_impulse_pattern(self, points: List[SwingPoint], direction: str) -> WavePattern:
        """推進波パターンを作成"""
        waves = []
        labels = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        
        for i in range(5):
            start_point = points[i]
            end_point = points[i + 1]
            
            price_length = abs(end_point.price - start_point.price)
            time_length = end_point.index - start_point.index
            
            # フィボナッチレベル計算
            fibonacci_levels = self._calculate_fibonacci_levels(start_point, end_point, labels[i])
            
            wave = ElliottWave(
                start_point=start_point,
                end_point=end_point,
                wave_label=labels[i],
                wave_type=WaveType.IMPULSE,
                price_length=price_length,
                time_length=time_length,
                fibonacci_levels=fibonacci_levels,
                confidence=0.8  # 基本的な信頼度
            )
            waves.append(wave)
        
        # 次のターゲット計算
        next_target = self._calculate_next_target(waves[-1], direction)
        
        # 無効化レベル計算
        invalidation_level = self._calculate_invalidation_level(waves, direction)
        
        return WavePattern(
            waves=waves,
            pattern_type=WaveType.IMPULSE,
            completion_percentage=100.0,  # 5波完了
            next_target=next_target,
            invalidation_level=invalidation_level
        )
    
    def _find_corrective_pattern(self, swing_points: List[SwingPoint]) -> Optional[WavePattern]:
        """修正波パターンを検索（ABC波）"""
        if len(swing_points) < 4:
            return None
        
        # 最新の4つのポイントでABC修正波をチェック
        recent_points = swing_points[-4:]
        
        # 上昇修正波の場合
        if (recent_points[0].swing_type == 'low' and
            recent_points[1].swing_type == 'high' and
            recent_points[2].swing_type == 'low' and
            recent_points[3].swing_type == 'high'):
            
            return self._create_corrective_pattern(recent_points, direction='up')
        
        # 下降修正波の場合
        elif (recent_points[0].swing_type == 'high' and
              recent_points[1].swing_type == 'low' and
              recent_points[2].swing_type == 'high' and
              recent_points[3].swing_type == 'low'):
            
            return self._create_corrective_pattern(recent_points, direction='down')
        
        return None
    
    def _create_corrective_pattern(self, points: List[SwingPoint], direction: str) -> WavePattern:
        """修正波パターンを作成"""
        waves = []
        labels = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        
        for i in range(3):
            start_point = points[i]
            end_point = points[i + 1]
            
            price_length = abs(end_point.price - start_point.price)
            time_length = end_point.index - start_point.index
            
            fibonacci_levels = self._calculate_fibonacci_levels(start_point, end_point, labels[i])
            
            wave = ElliottWave(
                start_point=start_point,
                end_point=end_point,
                wave_label=labels[i],
                wave_type=WaveType.CORRECTIVE,
                price_length=price_length,
                time_length=time_length,
                fibonacci_levels=fibonacci_levels,
                confidence=0.7  # 修正波の信頼度
            )
            waves.append(wave)
        
        return WavePattern(
            waves=waves,
            pattern_type=WaveType.CORRECTIVE,
            completion_percentage=100.0,
            next_target=None,
            invalidation_level=None
        )
    
    def _calculate_fibonacci_levels(self, start_point: SwingPoint, end_point: SwingPoint, 
                                  wave_label: WaveLabel) -> List[FibonacciLevel]:
        """フィボナッチレベルを計算"""
        levels = []
        price_range = abs(end_point.price - start_point.price)
        
        if wave_label in [WaveLabel.WAVE_2, WaveLabel.WAVE_4]:
            # リトレースメントレベル
            for ratio in self.fib_ratios['retracement']:
                if start_point.price < end_point.price:  # 上昇波の押し戻し
                    level_price = end_point.price - (price_range * ratio)
                else:  # 下降波の戻し
                    level_price = end_point.price + (price_range * ratio)
                
                levels.append(FibonacciLevel(
                    ratio=ratio,
                    price=level_price,
                    level_type='support' if start_point.price < end_point.price else 'resistance'
                ))
        
        elif wave_label in [WaveLabel.WAVE_3, WaveLabel.WAVE_5]:
            # エクステンションレベル
            for ratio in self.fib_ratios['extension']:
                if start_point.price < end_point.price:  # 上昇波
                    level_price = start_point.price + (price_range * ratio)
                else:  # 下降波
                    level_price = start_point.price - (price_range * ratio)
                
                levels.append(FibonacciLevel(
                    ratio=ratio,
                    price=level_price,
                    level_type='target'
                ))
        
        return levels
    
    def _calculate_next_target(self, last_wave: ElliottWave, direction: str) -> Optional[FibonacciLevel]:
        """次のターゲットを計算"""
        if last_wave.wave_label == WaveLabel.WAVE_5:
            # 5波完了後は修正波の可能性
            if direction == 'up':
                target_price = last_wave.end_point.price * 0.618  # 61.8%戻し
                return FibonacciLevel(0.618, target_price, 'target')
            else:
                target_price = last_wave.end_point.price * 1.382  # 38.2%戻し
                return FibonacciLevel(0.618, target_price, 'target')
        
        return None
    
    def _calculate_invalidation_level(self, waves: List[ElliottWave], direction: str) -> Optional[float]:
        """無効化レベルを計算"""
        if len(waves) >= 1:
            first_wave = waves[0]
            if direction == 'up':
                return first_wave.start_point.price  # 1波開始点を下回ると無効
            else:
                return first_wave.start_point.price  # 1波開始点を上回ると無効
        
        return None
    
    def _create_empty_pattern(self) -> WavePattern:
        """空のパターンを作成"""
        return WavePattern(
            waves=[],
            pattern_type=WaveType.UNKNOWN,
            completion_percentage=0.0,
            next_target=None,
            invalidation_level=None
        )


class ElliottWaveService:
    """エリオット波動分析サービス"""
    
    def __init__(self, strategy: ElliottWaveStrategy = None):
        self.strategy = strategy or ClassicElliottWave()
    
    def analyze(self, data: pd.DataFrame) -> WavePattern:
        """波動分析を実行"""
        return self.strategy.analyze_waves(data)
    
    def set_strategy(self, strategy: ElliottWaveStrategy):
        """戦略を変更"""
        self.strategy = strategy
    
    def get_wave_summary(self, pattern: WavePattern) -> Dict:
        """波動パターンのサマリーを取得"""
        return {
            'pattern_type': pattern.pattern_type.value,
            'waves_count': len(pattern.waves),
            'completion_percentage': pattern.completion_percentage,
            'current_wave': pattern.waves[-1].wave_label.value if pattern.waves else None,
            'next_target': {
                'ratio': pattern.next_target.ratio,
                'price': pattern.next_target.price,
                'type': pattern.next_target.level_type
            } if pattern.next_target else None,
            'invalidation_level': pattern.invalidation_level
        }