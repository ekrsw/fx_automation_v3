import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from app.services.technical_indicators import ZigZagIndicator, ATRIndicator


class TrendDirection(Enum):
    """トレンド方向"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass
class SwingPoint:
    """スイングポイント"""
    index: int
    datetime: pd.Timestamp
    price: float
    swing_type: str  # 'high' or 'low'
    volume: int = 0


@dataclass
class TrendAnalysis:
    """トレンド分析結果"""
    trend_direction: TrendDirection
    confidence: float  # 0-1の信頼度
    swing_points: List[SwingPoint]
    higher_highs: List[SwingPoint]
    higher_lows: List[SwingPoint]
    lower_highs: List[SwingPoint]
    lower_lows: List[SwingPoint]
    last_updated: pd.Timestamp


class DowTheoryStrategy(ABC):
    """ダウ理論戦略の基底クラス"""
    
    @abstractmethod
    def analyze_trend(self, data: pd.DataFrame) -> TrendAnalysis:
        """トレンド分析を実行"""
        pass


class ClassicDowTheory(DowTheoryStrategy):
    """クラシックなダウ理論実装"""
    
    def __init__(self, 
                 zigzag_deviation: float = 3.0,
                 min_swing_period: int = 5,
                 confirmation_threshold: float = 0.7):
        self.zigzag_deviation = zigzag_deviation
        self.min_swing_period = min_swing_period
        self.confirmation_threshold = confirmation_threshold
        self.zigzag = ZigZagIndicator(zigzag_deviation)
        self.atr = ATRIndicator()
    
    def analyze_trend(self, data: pd.DataFrame) -> TrendAnalysis:
        """メインのトレンド分析"""
        if len(data) < 3:
            return self._create_empty_analysis(data)
        
        # ZigZag計算
        zigzag_data = self.zigzag.calculate(data)
        swing_points = self._extract_swing_points(zigzag_data)
        
        if len(swing_points) < 2:
            return self._create_empty_analysis(data)
        
        # スイングポイント分類
        categorized_swings = self._categorize_swing_points(swing_points)
        
        # トレンド判定
        trend_direction = self._determine_trend_direction(categorized_swings)
        confidence = self._calculate_confidence(categorized_swings, trend_direction)
        
        return TrendAnalysis(
            trend_direction=trend_direction,
            confidence=confidence,
            swing_points=swing_points,
            higher_highs=categorized_swings['higher_highs'],
            higher_lows=categorized_swings['higher_lows'],
            lower_highs=categorized_swings['lower_highs'],
            lower_lows=categorized_swings['lower_lows'],
            last_updated=data.iloc[-1]['datetime']
        )
    
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
            if pd.notna(row['zigzag_low']):
                swing_points.append(SwingPoint(
                    index=idx,
                    datetime=row['datetime'],
                    price=row['zigzag_low'],
                    swing_type='low',
                    volume=row.get('volume', 0)
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def _categorize_swing_points(self, swing_points: List[SwingPoint]) -> Dict[str, List[SwingPoint]]:
        """スイングポイントを高値・安値別に分類"""
        highs = [sp for sp in swing_points if sp.swing_type == 'high']
        lows = [sp for sp in swing_points if sp.swing_type == 'low']
        
        categorized = {
            'higher_highs': [],
            'higher_lows': [],
            'lower_highs': [],
            'lower_lows': []
        }
        
        # 高値の分類
        for i in range(1, len(highs)):
            current = highs[i]
            previous = highs[i-1]
            
            if current.price > previous.price:
                categorized['higher_highs'].append(current)
            else:
                categorized['lower_highs'].append(current)
        
        # 安値の分類
        for i in range(1, len(lows)):
            current = lows[i]
            previous = lows[i-1]
            
            if current.price > previous.price:
                categorized['higher_lows'].append(current)
            else:
                categorized['lower_lows'].append(current)
        
        return categorized
    
    def _determine_trend_direction(self, categorized_swings: Dict[str, List[SwingPoint]]) -> TrendDirection:
        """トレンド方向を判定"""
        hh_count = len(categorized_swings['higher_highs'])
        hl_count = len(categorized_swings['higher_lows'])
        lh_count = len(categorized_swings['lower_highs'])
        ll_count = len(categorized_swings['lower_lows'])
        
        # 上昇トレンド判定: HH と HL が存在
        if hh_count >= 1 and hl_count >= 1:
            uptrend_strength = (hh_count + hl_count) / max(lh_count + ll_count, 1)
            if uptrend_strength > 1.5:
                return TrendDirection.UPTREND
        
        # 下降トレンド判定: LH と LL が存在
        if lh_count >= 1 and ll_count >= 1:
            downtrend_strength = (lh_count + ll_count) / max(hh_count + hl_count, 1)
            if downtrend_strength > 1.5:
                return TrendDirection.DOWNTREND
        
        # レンジ相場判定
        total_swings = hh_count + hl_count + lh_count + ll_count
        if total_swings >= 4:
            return TrendDirection.SIDEWAYS
        
        return TrendDirection.UNKNOWN
    
    def _calculate_confidence(self, categorized_swings: Dict[str, List[SwingPoint]], 
                            trend_direction: TrendDirection) -> float:
        """信頼度を計算"""
        if trend_direction == TrendDirection.UNKNOWN:
            return 0.0
        
        hh_count = len(categorized_swings['higher_highs'])
        hl_count = len(categorized_swings['higher_lows'])
        lh_count = len(categorized_swings['lower_highs'])
        ll_count = len(categorized_swings['lower_lows'])
        
        total_swings = hh_count + hl_count + lh_count + ll_count
        if total_swings == 0:
            return 0.0
        
        if trend_direction == TrendDirection.UPTREND:
            trend_swings = hh_count + hl_count
        elif trend_direction == TrendDirection.DOWNTREND:
            trend_swings = lh_count + ll_count
        else:
            return 0.5  # SIDEWAYS
        
        confidence = trend_swings / total_swings
        return min(confidence, 1.0)
    
    def _create_empty_analysis(self, data: pd.DataFrame) -> TrendAnalysis:
        """空の分析結果を作成"""
        last_timestamp = data.iloc[-1]['datetime'] if len(data) > 0 else pd.Timestamp.now()
        
        return TrendAnalysis(
            trend_direction=TrendDirection.UNKNOWN,
            confidence=0.0,
            swing_points=[],
            higher_highs=[],
            higher_lows=[],
            lower_highs=[],
            lower_lows=[],
            last_updated=last_timestamp
        )


class DowTheoryService:
    """ダウ理論分析サービス"""
    
    def __init__(self, strategy: DowTheoryStrategy = None):
        self.strategy = strategy or ClassicDowTheory()
    
    def analyze(self, data: pd.DataFrame) -> TrendAnalysis:
        """トレンド分析を実行"""
        return self.strategy.analyze_trend(data)
    
    def set_strategy(self, strategy: DowTheoryStrategy):
        """戦略を変更"""
        self.strategy = strategy
    
    def get_trend_summary(self, analysis: TrendAnalysis) -> Dict:
        """分析結果のサマリーを取得"""
        return {
            'trend_direction': analysis.trend_direction.value,
            'confidence': round(analysis.confidence, 3),
            'swing_points_count': len(analysis.swing_points),
            'higher_highs_count': len(analysis.higher_highs),
            'higher_lows_count': len(analysis.higher_lows),
            'lower_highs_count': len(analysis.lower_highs),
            'lower_lows_count': len(analysis.lower_lows),
            'last_updated': analysis.last_updated.isoformat() if analysis.last_updated else None
        }