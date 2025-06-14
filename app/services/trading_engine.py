from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from sqlalchemy.orm import Session

from app.models.positions import Position, PositionType, PositionStatus, PositionHistory
from app.models.system_status import SystemStatus
from app.services.risk_manager import RiskManagerService
from app.services.signal_generator import TradingSignal, SignalType
from app.services.mt5_service import MT5Service


class OrderType(str, Enum):
    """注文タイプ"""
    MARKET = "market"      # 成行注文
    PENDING = "pending"    # 指値・逆指値注文
    STOP = "stop"         # ストップ注文
    LIMIT = "limit"       # リミット注文


class ExecutionMode(str, Enum):
    """実行モード"""
    LIVE = "live"         # 実取引
    SIMULATION = "simulation"  # シミュレーション
    PAPER = "paper"       # ペーパートレード


@dataclass
class OrderRequest:
    """注文リクエスト"""
    symbol: str
    order_type: OrderType
    position_type: PositionType
    lot_size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None
    strategy_name: Optional[str] = None
    signal_data: Optional[Dict[str, Any]] = None


@dataclass
class OrderResult:
    """注文結果"""
    success: bool
    position_id: Optional[int] = None
    ticket: Optional[int] = None
    message: str = ""
    error_code: Optional[int] = None
    execution_price: Optional[float] = None


class TradingStrategy(ABC):
    """取引戦略の基底クラス"""
    
    @abstractmethod
    def should_open_position(self, signal: TradingSignal, market_data: Dict[str, Any]) -> bool:
        """ポジションを開くべきかの判定"""
        pass
    
    @abstractmethod
    def should_close_position(self, position: Position, market_data: Dict[str, Any]) -> bool:
        """ポジションを閉じるべきかの判定"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, account_info: Dict[str, Any]) -> float:
        """ポジションサイズ計算"""
        pass


class AutomatedTradingStrategy(TradingStrategy):
    """自動取引戦略"""
    
    def __init__(self, 
                 min_signal_confidence: float = 0.7,
                 max_daily_trades: int = 10,
                 max_drawdown_threshold: float = 0.1):  # 10%
        self.min_signal_confidence = min_signal_confidence
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_threshold = max_drawdown_threshold
    
    def should_open_position(self, signal: TradingSignal, market_data: Dict[str, Any]) -> bool:
        """ポジションオープン判定"""
        
        # シグナル信頼度チェック
        if signal.confidence < self.min_signal_confidence:
            return False
        
        # ホールドシグナルは除外
        if signal.signal_type == SignalType.HOLD:
            return False
        
        # 市場時間チェック
        if not market_data.get('market_open', True):
            return False
        
        # リスク・リワード比チェック
        if signal.risk_reward_ratio and signal.risk_reward_ratio < 2.0:
            return False
        
        return True
    
    def should_close_position(self, position: Position, market_data: Dict[str, Any]) -> bool:
        """ポジションクローズ判定"""
        
        # ストップロス・テイクプロフィットは MT5 で自動執行
        # ここでは追加的な条件をチェック
        
        current_price = market_data.get('current_price')
        if not current_price:
            return False
        
        # 緊急時クローズ（大きな含み損）
        if position.entry_price and current_price:
            unrealized_pnl = position.unrealized_pnl
            if unrealized_pnl < -position.risk_amount * 2:  # リスク額の2倍の損失
                return True
        
        # 利益確定の判定（利益が一定以上で市場が逆転しそうな場合）
        if position.is_profitable and market_data.get('trend_reversal_signal', False):
            return True
        
        return False
    
    def calculate_position_size(self, signal: TradingSignal, account_info: Dict[str, Any]) -> float:
        """ポジションサイズ計算"""
        # リスク管理サービスで計算された推奨サイズを使用
        return account_info.get('recommended_position_size', 0.01)


class TradingEngineService:
    """取引エンジンサービス"""
    
    def __init__(self, 
                 db: Session,
                 execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
                 strategy: TradingStrategy = None):
        self.db = db
        self.execution_mode = execution_mode
        self.strategy = strategy or AutomatedTradingStrategy()
        self.risk_manager = RiskManagerService(db)
        self.mt5_service = MT5Service() if execution_mode == ExecutionMode.LIVE else None
        
        # 設定
        self.config = {
            'max_slippage': 3,      # 最大スリッページ (pips)
            'retry_attempts': 3,     # リトライ回数
            'timeout_seconds': 30,   # タイムアウト
            'enable_trailing_stop': False,  # トレーリングストップ
            'max_positions_per_symbol': 1   # シンボルあたり最大ポジション数
        }
    
    def execute_signal(self, 
                      signal: TradingSignal,
                      account_balance: float = 100000) -> OrderResult:
        """シグナルに基づく取引実行"""
        
        try:
            # 市場データ取得
            market_data = self._get_current_market_data(signal.entry_price)
            
            # 取引戦略による判定
            if not self.strategy.should_open_position(signal, market_data):
                return OrderResult(
                    success=False,
                    message="取引戦略により実行見送り"
                )
            
            # リスク管理検証
            validation = self._validate_signal(signal, account_balance)
            if not validation['is_valid']:
                return OrderResult(
                    success=False,
                    message=f"リスク管理検証失敗: {', '.join(validation['errors'])}"
                )
            
            # ポジションサイズ計算
            position_size_calc = validation['position_size_calc']
            lot_size = position_size_calc.recommended_size
            
            # 注文リクエスト作成
            order_request = OrderRequest(
                symbol="USDJPY",  # 実装では signal から取得
                order_type=OrderType.MARKET,
                position_type=PositionType.BUY if signal.signal_type == SignalType.BUY else PositionType.SELL,
                lot_size=lot_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"Auto trade - {signal.signal_type.value}",
                strategy_name="AutomatedTradingStrategy",
                signal_data={
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'risk_reward_ratio': signal.risk_reward_ratio
                }
            )
            
            # 注文実行
            return self._execute_order(order_request)
            
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"実行エラー: {str(e)}"
            )
    
    def _execute_order(self, order_request: OrderRequest) -> OrderResult:
        """注文実行"""
        
        try:
            # ポジションレコード作成
            position = Position(
                symbol=order_request.symbol,
                position_type=order_request.position_type,
                status=PositionStatus.PENDING,
                lot_size=order_request.lot_size,
                entry_price=order_request.price,
                stop_loss=order_request.stop_loss,
                take_profit=order_request.take_profit,
                strategy_name=order_request.strategy_name,
                analysis_data=json.dumps(order_request.signal_data) if order_request.signal_data else None,
                comments=order_request.comment
            )
            
            self.db.add(position)
            self.db.flush()  # IDを取得
            
            # 実行モードに応じた処理
            if self.execution_mode == ExecutionMode.LIVE:
                # 実取引実行
                result = self._execute_live_order(order_request, position)
            elif self.execution_mode == ExecutionMode.SIMULATION:
                # シミュレーション実行
                result = self._execute_simulation_order(order_request, position)
            else:  # PAPER
                # ペーパートレード実行
                result = self._execute_paper_order(order_request, position)
            
            # 結果に応じてポジション更新
            if result.success:
                position.status = PositionStatus.OPEN
                position.mt5_ticket = result.ticket
                position.entry_price = result.execution_price or order_request.price
                position.opened_at = datetime.now()
                
                # リスク金額計算
                if position.entry_price and position.stop_loss:
                    risk_pips = abs(position.entry_price - position.stop_loss) / self._get_pip_value(position.symbol)
                    position.risk_amount = position.lot_size * risk_pips * self._get_pip_value(position.symbol) * 100000
                
                result.position_id = position.id
            else:
                position.status = PositionStatus.CANCELLED
                position.comments = f"実行失敗: {result.message}"
            
            # 履歴記録
            self._record_position_history(position, "create", result.message)
            
            self.db.commit()
            return result
            
        except Exception as e:
            self.db.rollback()
            return OrderResult(
                success=False,
                message=f"注文実行エラー: {str(e)}"
            )
    
    def _execute_live_order(self, order_request: OrderRequest, position: Position) -> OrderResult:
        """実取引実行"""
        if not self.mt5_service or not self.mt5_service.is_connected():
            return OrderResult(
                success=False,
                message="MT5接続エラー"
            )
        
        # MT5での注文実行実装
        # 実装では MT5Service を拡張して取引機能を追加
        return OrderResult(
            success=True,
            ticket=12345,  # 仮のチケット番号
            execution_price=order_request.price,
            message="実取引実行成功（実装中）"
        )
    
    def _execute_simulation_order(self, order_request: OrderRequest, position: Position) -> OrderResult:
        """シミュレーション実行"""
        # シミュレーションでは即座に実行成功とする
        return OrderResult(
            success=True,
            ticket=position.id + 10000,  # 仮のチケット番号
            execution_price=order_request.price,
            message="シミュレーション実行成功"
        )
    
    def _execute_paper_order(self, order_request: OrderRequest, position: Position) -> OrderResult:
        """ペーパートレード実行"""
        return OrderResult(
            success=True,
            ticket=position.id + 20000,  # 仮のチケット番号
            execution_price=order_request.price,
            message="ペーパートレード実行成功"
        )
    
    def close_position(self, position_id: int, reason: str = "Manual close") -> OrderResult:
        """ポジションクローズ"""
        
        try:
            position = self.db.query(Position).filter(Position.id == position_id).first()
            if not position:
                return OrderResult(
                    success=False,
                    message="ポジションが見つかりません"
                )
            
            if not position.is_open:
                return OrderResult(
                    success=False,
                    message="ポジションがオープン状態ではありません"
                )
            
            # 現在価格取得
            current_price = self._get_current_price(position.symbol)
            
            # 実行モードに応じたクローズ処理
            if self.execution_mode == ExecutionMode.LIVE:
                # 実取引クローズ
                close_result = self._close_live_position(position, current_price)
            else:
                # シミュレーション/ペーパートレードクローズ
                close_result = self._close_simulation_position(position, current_price)
            
            if close_result.success:
                # ポジション更新
                position.status = PositionStatus.CLOSED
                position.exit_price = current_price
                position.closed_at = datetime.now()
                
                # 損益計算
                self._calculate_final_pnl(position)
                
                # 履歴記録
                self._record_position_history(position, "close", reason)
                
                self.db.commit()
            
            return close_result
            
        except Exception as e:
            self.db.rollback()
            return OrderResult(
                success=False,
                message=f"クローズエラー: {str(e)}"
            )
    
    def monitor_positions(self) -> Dict[str, Any]:
        """ポジション監視"""
        
        open_positions = self.db.query(Position).filter(
            Position.status == PositionStatus.OPEN
        ).all()
        
        monitoring_result = {
            'total_positions': len(open_positions),
            'actions_taken': [],
            'warnings': [],
            'summary': {}
        }
        
        for position in open_positions:
            try:
                # 現在価格更新
                current_price = self._get_current_price(position.symbol)
                position.current_price = current_price
                
                # 市場データ取得
                market_data = self._get_current_market_data(current_price)
                
                # クローズ判定
                if self.strategy.should_close_position(position, market_data):
                    close_result = self.close_position(position.id, "Strategy signal")
                    if close_result.success:
                        monitoring_result['actions_taken'].append(
                            f"ポジション {position.id} をクローズしました"
                        )
                
                # リスク警告
                unrealized_pnl = position.unrealized_pnl
                if unrealized_pnl < -position.risk_amount * 1.5:
                    monitoring_result['warnings'].append(
                        f"ポジション {position.id} の含み損が大きくなっています"
                    )
                
            except Exception as e:
                monitoring_result['warnings'].append(
                    f"ポジション {position.id} の監視エラー: {str(e)}"
                )
        
        # サマリー更新
        self.db.commit()
        
        monitoring_result['summary'] = self._get_portfolio_summary()
        return monitoring_result
    
    def _validate_signal(self, signal: TradingSignal, account_balance: float) -> Dict[str, Any]:
        """シグナル検証"""
        return self.risk_manager.validate_new_position(
            symbol="USDJPY",  # 実装では signal から取得
            position_type=signal.signal_type.value,
            lot_size=0.1,  # 仮の値、実際は計算
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            account_balance=account_balance
        )
    
    def _get_current_market_data(self, current_price: float) -> Dict[str, Any]:
        """現在の市場データ取得"""
        return {
            'current_price': current_price,
            'market_open': True,
            'volatility': 0.02,
            'trend_reversal_signal': False
        }
    
    def _get_current_price(self, symbol: str) -> float:
        """現在価格取得（シミュレーション用）"""
        # 実装では実際の価格を取得
        return 150.0  # 仮の価格
    
    def _get_pip_value(self, symbol: str) -> float:
        """ピップ値取得"""
        if symbol.endswith('JPY'):
            return 0.01
        else:
            return 0.0001
    
    def _calculate_final_pnl(self, position: Position):
        """最終損益計算"""
        if not all([position.entry_price, position.exit_price, position.lot_size]):
            return
        
        price_diff = position.exit_price - position.entry_price
        if position.position_type == PositionType.SELL:
            price_diff = -price_diff
        
        # 簡易的な損益計算
        position.profit_loss = price_diff * position.lot_size * 100000
        position.net_profit = position.profit_loss - position.commission - position.swap
    
    def _close_live_position(self, position: Position, current_price: float) -> OrderResult:
        """実取引ポジションクローズ"""
        # 実装では MT5Service を使用
        return OrderResult(
            success=True,
            message="実取引クローズ成功（実装中）"
        )
    
    def _close_simulation_position(self, position: Position, current_price: float) -> OrderResult:
        """シミュレーションポジションクローズ"""
        return OrderResult(
            success=True,
            message="シミュレーションクローズ成功"
        )
    
    def _record_position_history(self, position: Position, action: str, reason: str):
        """ポジション履歴記録"""
        history = PositionHistory(
            position_id=position.id,
            action=action,
            price=position.current_price,
            volume=position.lot_size,
            reason=reason,
            comments=f"{action} - {reason}"
        )
        self.db.add(history)
    
    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリー"""
        open_positions = self.db.query(Position).filter(
            Position.status == PositionStatus.OPEN
        ).all()
        
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_risk = sum(p.risk_amount or 0 for p in open_positions)
        
        return {
            'open_positions': len(open_positions),
            'total_unrealized_pnl': total_unrealized,
            'total_risk_amount': total_risk,
            'profit_positions': len([p for p in open_positions if p.unrealized_pnl > 0]),
            'loss_positions': len([p for p in open_positions if p.unrealized_pnl < 0])
        }
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """取引サマリー"""
        # 本日の取引
        today_positions = self.db.query(Position).filter(
            Position.created_at >= datetime.now().date()
        ).all()
        
        # 全期間の統計
        all_closed_positions = self.db.query(Position).filter(
            Position.status == PositionStatus.CLOSED
        ).all()
        
        return {
            'today': {
                'total_trades': len(today_positions),
                'open_trades': len([p for p in today_positions if p.is_open]),
                'closed_trades': len([p for p in today_positions if p.is_closed])
            },
            'overall': {
                'total_closed': len(all_closed_positions),
                'profitable_trades': len([p for p in all_closed_positions if p.is_profitable]),
                'total_profit': sum(p.net_profit or 0 for p in all_closed_positions),
                'win_rate': len([p for p in all_closed_positions if p.is_profitable]) / len(all_closed_positions) if all_closed_positions else 0
            },
            'current_portfolio': self._get_portfolio_summary()
        }
    
    def set_execution_mode(self, mode: ExecutionMode):
        """実行モード変更"""
        self.execution_mode = mode
        if mode == ExecutionMode.LIVE and not self.mt5_service:
            self.mt5_service = MT5Service()
    
    def set_strategy(self, strategy: TradingStrategy):
        """取引戦略変更"""
        self.strategy = strategy