import pandas as pd
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

from app.services.dow_theory import DowTheoryService, TrendDirection, TrendAnalysis
from app.services.elliott_wave import ElliottWaveService, WavePattern, WaveLabel, WaveType
from app.services.technical_indicators import TechnicalIndicatorFactory


class SignalType(Enum):
    """シグナルタイプ"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"


class SignalStrength(Enum):
    """シグナル強度"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """取引シグナル"""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    reasoning: List[str]  # シグナル根拠
    timestamp: datetime
    metadata: Dict[str, Any]


class SignalStrategy(ABC):
    """シグナル戦略の基底クラス"""
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """シグナルを生成"""
        pass


class DowElliottCombinedStrategy(SignalStrategy):
    """ダウ理論とエリオット波動を組み合わせた戦略"""
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 risk_reward_min: float = 2.0):
        self.min_confidence = min_confidence
        self.risk_reward_min = risk_reward_min
        self.dow_service = DowTheoryService()
        self.elliott_service = ElliottWaveService()
        self.rsi_indicator = TechnicalIndicatorFactory.create('rsi', period=14)
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """メインのシグナル生成ロジック"""
        if len(data) < 50:
            return self._create_hold_signal("データ不足", data)
        
        # 各分析を実行
        dow_analysis = self.dow_service.analyze(data)
        elliott_pattern = self.elliott_service.analyze(data)
        rsi_values = self.rsi_indicator.calculate(data)
        current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50
        
        # シグナル判定
        signal = self._evaluate_combined_signal(
            data, dow_analysis, elliott_pattern, current_rsi
        )
        
        return signal
    
    def _evaluate_combined_signal(self, 
                                data: pd.DataFrame,
                                dow_analysis: TrendAnalysis,
                                elliott_pattern: WavePattern,
                                current_rsi: float) -> TradingSignal:
        """総合的なシグナル評価"""
        
        current_price = data.iloc[-1]['close']
        reasoning = []
        signal_type = SignalType.HOLD
        strength = SignalStrength.WEAK
        confidence = 0.0
        
        # ダウ理論によるトレンド確認
        dow_signal, dow_confidence = self._evaluate_dow_signal(dow_analysis)
        reasoning.extend(dow_signal['reasoning'])
        
        # エリオット波動による位置確認
        elliott_signal, elliott_confidence = self._evaluate_elliott_signal(elliott_pattern)
        reasoning.extend(elliott_signal['reasoning'])
        
        # RSIによる過熱感確認
        rsi_signal = self._evaluate_rsi_signal(current_rsi)
        reasoning.extend(rsi_signal['reasoning'])
        
        # 総合判定
        if (dow_signal['type'] == SignalType.BUY and 
            elliott_signal['type'] == SignalType.BUY and
            rsi_signal['type'] != SignalType.SELL):
            
            signal_type = SignalType.BUY
            confidence = (dow_confidence + elliott_confidence) / 2
            
            if confidence > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif confidence > 0.7:
                strength = SignalStrength.STRONG
            elif confidence > 0.6:
                strength = SignalStrength.MODERATE
            
            reasoning.append("ダウ理論とエリオット波動が買いシグナルで一致")
        
        elif (dow_signal['type'] == SignalType.SELL and 
              elliott_signal['type'] == SignalType.SELL and
              rsi_signal['type'] != SignalType.BUY):
            
            signal_type = SignalType.SELL
            confidence = (dow_confidence + elliott_confidence) / 2
            
            if confidence > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif confidence > 0.7:
                strength = SignalStrength.STRONG
            elif confidence > 0.6:
                strength = SignalStrength.MODERATE
            
            reasoning.append("ダウ理論とエリオット波動が売りシグナルで一致")
        
        # エントリー価格とリスク管理レベル計算
        entry_price, stop_loss, take_profit, risk_reward = self._calculate_trade_levels(
            signal_type, current_price, elliott_pattern, dow_analysis
        )
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata={
                'dow_trend': dow_analysis.trend_direction.value,
                'dow_confidence': dow_analysis.confidence,
                'elliott_pattern': elliott_pattern.pattern_type.value,
                'elliott_completion': elliott_pattern.completion_percentage,
                'current_rsi': current_rsi
            }
        )
    
    def _evaluate_dow_signal(self, analysis: TrendAnalysis) -> tuple:
        """ダウ理論シグナル評価"""
        reasoning = []
        signal_type = SignalType.HOLD
        
        if analysis.trend_direction == TrendDirection.UPTREND:
            if analysis.confidence > self.min_confidence:
                signal_type = SignalType.BUY
                reasoning.append(f"ダウ理論上昇トレンド確認 (信頼度: {analysis.confidence:.2f})")
                reasoning.append(f"高値更新: {len(analysis.higher_highs)}回, 安値切り上げ: {len(analysis.higher_lows)}回")
        
        elif analysis.trend_direction == TrendDirection.DOWNTREND:
            if analysis.confidence > self.min_confidence:
                signal_type = SignalType.SELL
                reasoning.append(f"ダウ理論下降トレンド確認 (信頼度: {analysis.confidence:.2f})")
                reasoning.append(f"安値更新: {len(analysis.lower_lows)}回, 高値切り下げ: {len(analysis.lower_highs)}回")
        
        else:
            reasoning.append(f"ダウ理論: {analysis.trend_direction.value} (信頼度不足)")
        
        return {'type': signal_type, 'reasoning': reasoning}, analysis.confidence
    
    def _evaluate_elliott_signal(self, pattern: WavePattern) -> tuple:
        """エリオット波動シグナル評価"""
        reasoning = []
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        if not pattern.waves:
            reasoning.append("エリオット波動: パターン未検出")
            return {'type': signal_type, 'reasoning': reasoning}, confidence
        
        current_wave = pattern.waves[-1]
        
        # 推進波の場合
        if pattern.pattern_type == WaveType.IMPULSE:
            if current_wave.wave_label == WaveLabel.WAVE_2:
                signal_type = SignalType.BUY
                reasoning.append("エリオット波動: 第3波エントリーポイント (第2波完了)")
                confidence = 0.8
            
            elif current_wave.wave_label == WaveLabel.WAVE_4:
                signal_type = SignalType.BUY
                reasoning.append("エリオット波動: 第5波エントリーポイント (第4波完了)")
                confidence = 0.7
            
            elif current_wave.wave_label == WaveLabel.WAVE_5:
                if pattern.completion_percentage > 90:
                    signal_type = SignalType.SELL
                    reasoning.append("エリオット波動: 第5波完了間近 (利確/転換)")
                    confidence = 0.6
        
        # 修正波の場合
        elif pattern.pattern_type == WaveType.CORRECTIVE:
            if current_wave.wave_label == WaveLabel.WAVE_C:
                if pattern.completion_percentage > 80:
                    # 修正波終了の可能性
                    reasoning.append("エリオット波動: 修正波C波完了間近")
                    confidence = 0.5
        
        reasoning.append(f"現在波動: {current_wave.wave_label.value} ({pattern.pattern_type.value})")
        
        return {'type': signal_type, 'reasoning': reasoning}, confidence
    
    def _evaluate_rsi_signal(self, rsi_value: float) -> dict:
        """RSIシグナル評価"""
        reasoning = []
        signal_type = SignalType.HOLD
        
        if rsi_value > 70:
            signal_type = SignalType.SELL
            reasoning.append(f"RSI過買い状態 ({rsi_value:.1f})")
        elif rsi_value < 30:
            signal_type = SignalType.BUY
            reasoning.append(f"RSI過売り状態 ({rsi_value:.1f})")
        else:
            reasoning.append(f"RSI中立 ({rsi_value:.1f})")
        
        return {'type': signal_type, 'reasoning': reasoning}
    
    def _calculate_trade_levels(self, 
                               signal_type: SignalType,
                               current_price: float,
                               elliott_pattern: WavePattern,
                               dow_analysis: TrendAnalysis) -> tuple:
        """エントリー価格とリスク管理レベルを計算"""
        
        if signal_type == SignalType.HOLD:
            return None, None, None, None
        
        entry_price = current_price
        stop_loss = None
        take_profit = None
        risk_reward = None
        
        # エリオット波動からのレベル計算
        if elliott_pattern.invalidation_level:
            if signal_type == SignalType.BUY:
                stop_loss = elliott_pattern.invalidation_level
            else:
                stop_loss = elliott_pattern.invalidation_level
        
        # ダウ理論からのサポート/レジスタンス
        if dow_analysis.swing_points:
            recent_swings = dow_analysis.swing_points[-3:]
            if signal_type == SignalType.BUY:
                # 最近の安値をストップロスに
                recent_lows = [sp.price for sp in recent_swings if sp.swing_type == 'low']
                if recent_lows and (stop_loss is None or min(recent_lows) > stop_loss):
                    stop_loss = min(recent_lows) * 0.995  # 少しマージンを取る
            else:
                # 最近の高値をストップロスに
                recent_highs = [sp.price for sp in recent_swings if sp.swing_type == 'high']
                if recent_highs and (stop_loss is None or max(recent_highs) < stop_loss):
                    stop_loss = max(recent_highs) * 1.005
        
        # デフォルトストップロス (2% ATR)
        if stop_loss is None:
            atr_buffer = current_price * 0.02  # 2%をATRの代わりに使用
            if signal_type == SignalType.BUY:
                stop_loss = current_price - atr_buffer
            else:
                stop_loss = current_price + atr_buffer
        
        # 利益確定レベル
        if stop_loss:
            risk = abs(entry_price - stop_loss)
            if signal_type == SignalType.BUY:
                take_profit = entry_price + (risk * self.risk_reward_min)
            else:
                take_profit = entry_price - (risk * self.risk_reward_min)
            
            risk_reward = abs(take_profit - entry_price) / risk if risk > 0 else None
        
        return entry_price, stop_loss, take_profit, risk_reward
    
    def _create_hold_signal(self, reason: str, data: pd.DataFrame) -> TradingSignal:
        """ホールドシグナルを作成"""
        current_price = data.iloc[-1]['close'] if len(data) > 0 else 0.0
        
        return TradingSignal(
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=0.0,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            reasoning=[reason],
            timestamp=datetime.now(),
            metadata={}
        )


class SignalGeneratorService:
    """シグナル生成サービス"""
    
    def __init__(self, strategy: SignalStrategy = None):
        self.strategy = strategy or DowElliottCombinedStrategy()
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """シグナルを生成"""
        return self.strategy.generate_signal(data)
    
    def set_strategy(self, strategy: SignalStrategy):
        """戦略を変更"""
        self.strategy = strategy
    
    def get_signal_summary(self, signal: TradingSignal) -> Dict:
        """シグナルサマリーを取得"""
        return {
            'signal_type': signal.signal_type.value,
            'strength': signal.strength.value,
            'confidence': round(signal.confidence, 3),
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'risk_reward_ratio': round(signal.risk_reward_ratio, 2) if signal.risk_reward_ratio else None,
            'reasoning': signal.reasoning,
            'timestamp': signal.timestamp.isoformat(),
            'metadata': signal.metadata
        }