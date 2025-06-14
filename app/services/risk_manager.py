from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from sqlalchemy.orm import Session

from app.models.positions import Position, PositionType
from app.services.data_service import DataService


class RiskLevel(str, Enum):
    """リスクレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """リスク評価結果"""
    risk_level: RiskLevel
    risk_score: float  # 0-100
    max_position_size: float
    recommended_position_size: float
    warnings: List[str]
    risk_factors: Dict[str, float]
    confidence: float


@dataclass
class PositionSizeCalculation:
    """ポジションサイズ計算結果"""
    recommended_size: float
    max_size: float
    risk_amount: float
    risk_percentage: float
    reasoning: List[str]


class RiskStrategy(ABC):
    """リスク管理戦略の基底クラス"""
    
    @abstractmethod
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str,
                              **kwargs) -> PositionSizeCalculation:
        """ポジションサイズを計算"""
        pass
    
    @abstractmethod
    def assess_risk(self, 
                   positions: List[Position],
                   market_data: Dict[str, Any],
                   **kwargs) -> RiskAssessment:
        """リスク評価を実行"""
        pass


class FixedRiskStrategy(RiskStrategy):
    """固定リスク戦略（資金の1-2%）"""
    
    def __init__(self, 
                 risk_percentage: float = 0.02,  # 2%
                 max_risk_percentage: float = 0.05,  # 5%
                 max_positions: int = 5):
        self.risk_percentage = risk_percentage
        self.max_risk_percentage = max_risk_percentage
        self.max_positions = max_positions
    
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str,
                              **kwargs) -> PositionSizeCalculation:
        """固定リスクに基づくポジションサイズ計算"""
        
        reasoning = []
        
        # リスク金額計算
        risk_amount = account_balance * self.risk_percentage
        max_risk_amount = account_balance * self.max_risk_percentage
        
        # ピップ値計算（簡易版）
        pip_value = self._calculate_pip_value(symbol, entry_price)
        
        # ストップロス幅（pips）
        stop_loss_pips = abs(entry_price - stop_loss) / pip_value
        
        if stop_loss_pips == 0:
            return PositionSizeCalculation(
                recommended_size=0.0,
                max_size=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                reasoning=["ストップロスが設定されていません"]
            )
        
        # ポジションサイズ計算（ロット単位）
        # リスク金額 = ポジションサイズ × ストップロス幅(pips) × pip_value
        recommended_size = risk_amount / (stop_loss_pips * pip_value)
        max_size = max_risk_amount / (stop_loss_pips * pip_value)
        
        # ロット単位に調整（0.01単位）
        recommended_size = round(recommended_size, 2)
        max_size = round(max_size, 2)
        
        reasoning.append(f"リスク許容額: {risk_amount:.2f} ({self.risk_percentage*100:.1f}%)")
        reasoning.append(f"ストップロス幅: {stop_loss_pips:.1f} pips")
        reasoning.append(f"推奨ポジションサイズ: {recommended_size:.2f} ロット")
        
        return PositionSizeCalculation(
            recommended_size=recommended_size,
            max_size=max_size,
            risk_amount=risk_amount,
            risk_percentage=self.risk_percentage,
            reasoning=reasoning
        )
    
    def assess_risk(self, 
                   positions: List[Position],
                   market_data: Dict[str, Any],
                   account_balance: float = 100000,
                   **kwargs) -> RiskAssessment:
        """リスク評価"""
        
        warnings = []
        risk_factors = {}
        
        # オープンポジション分析
        open_positions = [p for p in positions if p.is_open]
        total_risk = sum(p.risk_amount or 0 for p in open_positions)
        
        # ポジション数リスク
        position_count_risk = len(open_positions) / self.max_positions
        risk_factors['position_count'] = position_count_risk
        
        if len(open_positions) >= self.max_positions:
            warnings.append(f"最大ポジション数に達しています ({len(open_positions)}/{self.max_positions})")
        
        # 資金リスク
        total_risk_percentage = total_risk / account_balance if account_balance > 0 else 0
        risk_factors['total_risk'] = total_risk_percentage
        
        if total_risk_percentage > self.max_risk_percentage:
            warnings.append(f"総リスクが制限を超えています ({total_risk_percentage*100:.1f}% > {self.max_risk_percentage*100:.1f}%)")
        
        # 通貨ペア集中リスク
        symbol_exposure = {}
        for pos in open_positions:
            base_currency = pos.symbol[:3]
            quote_currency = pos.symbol[3:6]
            
            symbol_exposure[base_currency] = symbol_exposure.get(base_currency, 0) + abs(pos.lot_size or 0)
            symbol_exposure[quote_currency] = symbol_exposure.get(quote_currency, 0) + abs(pos.lot_size or 0)
        
        max_exposure = max(symbol_exposure.values()) if symbol_exposure else 0
        concentration_risk = max_exposure / sum(abs(p.lot_size or 0) for p in open_positions) if open_positions else 0
        risk_factors['concentration'] = concentration_risk
        
        if concentration_risk > 0.6:  # 60%以上の集中
            warnings.append(f"通貨集中リスクが高すぎます ({concentration_risk*100:.1f}%)")
        
        # 総合リスクスコア計算
        risk_score = (
            position_count_risk * 30 +
            total_risk_percentage * 100 * 40 +
            concentration_risk * 30
        )
        
        # リスクレベル判定
        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 30:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 最大ポジションサイズ計算
        remaining_risk = max(0, account_balance * self.max_risk_percentage - total_risk)
        max_position_size = remaining_risk / (account_balance * self.risk_percentage) if account_balance > 0 else 0
        
        # 推奨ポジションサイズ（より保守的）
        recommended_position_size = max_position_size * 0.7
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            max_position_size=max_position_size,
            recommended_position_size=recommended_position_size,
            warnings=warnings,
            risk_factors=risk_factors,
            confidence=0.8
        )
    
    def _calculate_pip_value(self, symbol: str, price: float) -> float:
        """ピップ値計算（簡易版）"""
        # 主要通貨ペアの簡易計算
        if symbol.endswith('JPY'):
            return 0.01  # JPYペアは小数点2桁
        else:
            return 0.0001  # その他は小数点4桁


class RiskManagerService:
    """リスク管理サービス"""
    
    def __init__(self, db: Session, strategy: RiskStrategy = None):
        self.db = db
        self.data_service = DataService(db)
        self.strategy = strategy or FixedRiskStrategy()
    
    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_loss: float,
                              account_balance: float = 100000,
                              **kwargs) -> PositionSizeCalculation:
        """ポジションサイズ計算"""
        return self.strategy.calculate_position_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            symbol=symbol,
            **kwargs
        )
    
    def assess_portfolio_risk(self,
                            account_balance: float = 100000,
                            **kwargs) -> RiskAssessment:
        """ポートフォリオリスク評価"""
        from app.models.positions import Position
        
        # 現在のポジション取得
        positions = self.db.query(Position).filter(
            Position.status == "open"
        ).all()
        
        # 市場データ取得（簡易版）
        market_data = self._get_market_data()
        
        return self.strategy.assess_risk(
            positions=positions,
            market_data=market_data,
            account_balance=account_balance,
            **kwargs
        )
    
    def validate_new_position(self,
                            symbol: str,
                            position_type: str,
                            lot_size: float,
                            entry_price: float,
                            stop_loss: float,
                            account_balance: float = 100000) -> Dict[str, Any]:
        """新規ポジションの妥当性検証"""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'risk_assessment': None,
            'position_size_calc': None
        }
        
        try:
            # ポジションサイズ計算
            pos_calc = self.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                account_balance=account_balance
            )
            validation_result['position_size_calc'] = pos_calc
            
            # サイズ検証
            if lot_size > pos_calc.max_size:
                validation_result['errors'].append(
                    f"ポジションサイズが大きすぎます (要求: {lot_size}, 最大: {pos_calc.max_size})"
                )
                validation_result['is_valid'] = False
            
            if lot_size > pos_calc.recommended_size * 1.5:
                validation_result['warnings'].append(
                    f"推奨サイズを大幅に超えています (要求: {lot_size}, 推奨: {pos_calc.recommended_size})"
                )
            
            # ポートフォリオリスク評価
            risk_assessment = self.assess_portfolio_risk(account_balance=account_balance)
            validation_result['risk_assessment'] = risk_assessment
            
            if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                validation_result['warnings'].append(
                    f"ポートフォリオリスクが高レベルです ({risk_assessment.risk_level.value})"
                )
            
            # 価格検証
            if entry_price <= 0 or stop_loss <= 0:
                validation_result['errors'].append("価格は正の値である必要があります")
                validation_result['is_valid'] = False
            
            if abs(entry_price - stop_loss) / entry_price < 0.001:  # 0.1%未満
                validation_result['warnings'].append("ストップロスが小さすぎる可能性があります")
            
        except Exception as e:
            validation_result['errors'].append(f"検証エラー: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def _get_market_data(self) -> Dict[str, Any]:
        """市場データ取得（簡易版）"""
        # 実装では実際の市場データを取得
        return {
            'volatility': 0.02,  # 2%
            'trend_strength': 0.6,
            'market_hours': True
        }
    
    def set_strategy(self, strategy: RiskStrategy):
        """リスク管理戦略を変更"""
        self.strategy = strategy
    
    def get_risk_summary(self, account_balance: float = 100000) -> Dict[str, Any]:
        """リスク管理サマリー"""
        risk_assessment = self.assess_portfolio_risk(account_balance=account_balance)
        
        from app.models.positions import Position
        positions = self.db.query(Position).filter(
            Position.status == "open"
        ).all()
        
        return {
            'total_positions': len(positions),
            'total_risk_amount': sum(p.risk_amount or 0 for p in positions),
            'risk_level': risk_assessment.risk_level.value,
            'risk_score': risk_assessment.risk_score,
            'warnings': risk_assessment.warnings,
            'max_new_position_size': risk_assessment.max_position_size,
            'risk_factors': risk_assessment.risk_factors
        }