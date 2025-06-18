import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging

from app.models.price_data import PriceData
from app.services.signal_generator import SignalGeneratorService, TradingSignal, SignalType
from app.services.risk_manager import RiskManagerService, PositionSizeCalculation
from app.services.trading_engine import OrderType, ExecutionMode

logger = logging.getLogger(__name__)


class BacktestStatus(str, Enum):
    """バックテストステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestPosition:
    """バックテスト用ポジション"""
    id: int
    symbol: str
    position_type: str  # "buy" or "sell"
    entry_time: datetime
    entry_price: float
    lot_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    profit_loss: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_profit: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    signal_data: Optional[Dict[str, Any]] = None


@dataclass 
class BacktestMetrics:
    """バックテストメトリクス"""
    # 基本統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # 損益統計
    total_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    
    # リスク統計
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    recovery_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # 期間統計
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: int = 0
    trading_days: int = 0
    
    # 追加統計
    average_trade_duration: float = 0.0  # 時間
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    
    # リスク・リワード
    average_risk_reward: float = 0.0
    best_risk_reward: float = 0.0
    worst_risk_reward: float = 0.0


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    symbol: str = "USDJPY"
    timeframe: str = "H1"
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = field(default_factory=datetime.now)
    initial_balance: float = 100000.0
    commission_per_lot: float = 5.0
    spread_pips: float = 2.0
    slippage_pips: float = 1.0
    
    # リスク管理設定
    risk_per_trade: float = 0.02  # 2%
    max_risk_per_day: float = 0.06  # 6%
    max_positions: int = 3
    
    # 戦略設定
    min_signal_confidence: float = 0.7
    min_risk_reward: float = 1.5
    
    # 詳細設定
    enable_compound_interest: bool = True
    enable_swap_calculation: bool = False
    market_hours_only: bool = False
    
    def __post_init__(self):
        """初期化後のバリデーション"""
        if self.start_date >= self.end_date:
            raise ValueError("終了日は開始日より後である必要があります")
        if self.initial_balance <= 0:
            raise ValueError("初期残高は正の値である必要があります")
        if not (0 < self.risk_per_trade < 1):
            raise ValueError("リスク率は0-1の範囲である必要があります")
        if self.max_positions <= 0:
            raise ValueError("最大ポジション数は正の値である必要があります")
        if self.min_signal_confidence < 0 or self.min_signal_confidence > 1:
            raise ValueError("最小信頼度は0-1の範囲である必要があります")


@dataclass
class BacktestResult:
    """バックテスト結果"""
    config: BacktestConfig
    status: BacktestStatus
    metrics: BacktestMetrics
    positions: List[BacktestPosition]
    equity_curve: pd.DataFrame
    daily_returns: pd.DataFrame
    execution_time: float
    error_message: Optional[str] = None
    
    # 分析結果
    monthly_returns: Optional[pd.DataFrame] = None
    drawdown_periods: Optional[List[Dict[str, Any]]] = None
    performance_summary: Optional[Dict[str, Any]] = None


class BacktestEngine:
    """バックテストエンジン"""
    
    def __init__(self, db: Session):
        self.db = db
        self.signal_generator = SignalGeneratorService()
        self.risk_manager = RiskManagerService(db)
        
        # パフォーマンス追跡
        self.current_balance = 0.0
        self.equity_history = []
        self.positions = []
        self.position_counter = 0
        
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """バックテスト実行"""
        
        start_time = datetime.now()
        logger.info(f"バックテスト開始: {config.symbol} {config.timeframe} ({config.start_date.date()} - {config.end_date.date()})")
        
        try:
            # 初期化
            self._initialize_backtest(config)
            
            # 履歴データ取得
            price_data = self._load_price_data(config)
            if len(price_data) == 0:
                raise ValueError("価格データが見つかりません")
            
            # バックテスト実行
            result = self._execute_backtest(config, price_data)
            
            # メトリクス計算
            metrics = self._calculate_metrics(result.positions, config)
            
            # 分析結果生成
            equity_curve, daily_returns = self._generate_equity_curve(result.positions, config)
            
            # 結果作成
            execution_time = (datetime.now() - start_time).total_seconds()
            
            backtest_result = BacktestResult(
                config=config,
                status=BacktestStatus.COMPLETED,
                metrics=metrics,
                positions=result.positions,
                equity_curve=equity_curve,
                daily_returns=daily_returns,
                execution_time=execution_time
            )
            
            # 追加分析
            backtest_result.monthly_returns = self._calculate_monthly_returns(daily_returns)
            backtest_result.drawdown_periods = self._analyze_drawdown_periods(equity_curve)
            backtest_result.performance_summary = self._create_performance_summary(metrics)
            
            logger.info(f"バックテスト完了: {metrics.total_trades}取引, 勝率{metrics.win_rate:.1%}, 総利益{metrics.total_profit:.2f}")
            return backtest_result
            
        except Exception as e:
            logger.error(f"バックテストエラー: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return BacktestResult(
                config=config,
                status=BacktestStatus.FAILED,
                metrics=BacktestMetrics(),
                positions=[],
                equity_curve=pd.DataFrame(),
                daily_returns=pd.DataFrame(),
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _initialize_backtest(self, config: BacktestConfig):
        """バックテスト初期化"""
        self.current_balance = config.initial_balance
        self.equity_history = []
        self.positions = []
        self.position_counter = 0
        
        logger.debug(f"バックテスト初期化完了: 初期資金 {config.initial_balance}")
    
    def _load_price_data(self, config: BacktestConfig) -> pd.DataFrame:
        """価格データ読み込み"""
        
        try:
            # データベースからデータ取得
            price_records = self.db.query(PriceData).filter(
                and_(
                    PriceData.symbol == config.symbol,
                    PriceData.timeframe == config.timeframe,
                    PriceData.datetime >= config.start_date,
                    PriceData.datetime <= config.end_date
                )
            ).order_by(PriceData.datetime).all()
            
            if not price_records:
                logger.warning(f"価格データが見つかりません: {config.symbol} {config.timeframe}")
                return pd.DataFrame()
            
            # DataFrameに変換
            data = pd.DataFrame([
                {
                    'datetime': record.datetime,
                    'open': float(record.open),
                    'high': float(record.high),
                    'low': float(record.low),
                    'close': float(record.close),
                    'volume': int(record.volume) if record.volume else 0
                }
                for record in price_records
            ])
            
            # スプレッドを考慮した価格調整
            data['bid'] = data['close'] - (config.spread_pips * self._get_pip_value(config.symbol))
            data['ask'] = data['close'] + (config.spread_pips * self._get_pip_value(config.symbol))
            
            logger.info(f"価格データ読み込み完了: {len(data)}件 ({data['datetime'].min()} - {data['datetime'].max()})")
            return data
            
        except Exception as e:
            logger.error(f"価格データ読み込みエラー: {str(e)}")
            return pd.DataFrame()
    
    def _execute_backtest(self, config: BacktestConfig, price_data: pd.DataFrame) -> 'BacktestResult':
        """バックテスト実行コア"""
        
        positions = []
        open_positions = []
        
        for i, row in price_data.iterrows():
            current_time = row['datetime']
            current_data = price_data.iloc[:i+1]  # 現在時点までのデータ
            
            # 既存ポジションの更新とクローズ判定
            open_positions = self._update_open_positions(
                open_positions, row, config
            )
            
            # クローズされたポジションを結果に追加
            for pos in open_positions:
                if pos.exit_time is not None:
                    positions.append(pos)
            
            # まだオープンなポジションのみ残す
            open_positions = [pos for pos in open_positions if pos.exit_time is None]
            
            # 新規シグナル生成（十分なデータがある場合のみ）
            if len(current_data) >= 50:  # 最低50本のローソク足が必要
                signals = self._generate_signals(current_data, config)
                
                # シグナルに基づく新規ポジション
                for signal in signals:
                    if len(open_positions) < config.max_positions:
                        new_position = self._open_position(signal, row, config)
                        if new_position:
                            open_positions.append(new_position)
            
            # エクイティ記録
            self._record_equity(current_time, open_positions)
        
        # 残りのオープンポジションを強制クローズ
        final_row = price_data.iloc[-1]
        for pos in open_positions:
            pos.exit_time = final_row['datetime']
            pos.exit_price = final_row['close']
            pos.exit_reason = "End of backtest"
            self._calculate_position_pnl(pos, config)
            positions.append(pos)
        
        # 仮の結果オブジェクトを作成（metricsは後で計算）
        return type('Result', (), {'positions': positions})()
    
    def _generate_signals(self, data: pd.DataFrame, config: BacktestConfig) -> List[TradingSignal]:
        """シグナル生成"""
        
        try:
            # 最新データでシグナル生成
            signal = self.signal_generator.generate_signal(data)
            
            # 設定に基づくフィルタリング
            if (signal.confidence >= config.min_signal_confidence and
                signal.signal_type in [SignalType.BUY, SignalType.SELL] and
                signal.risk_reward_ratio and signal.risk_reward_ratio >= config.min_risk_reward):
                return [signal]
            
        except Exception as e:
            logger.debug(f"シグナル生成エラー: {str(e)}")
        
        return []
    
    def _open_position(self, signal: TradingSignal, price_row: pd.Series, config: BacktestConfig) -> Optional[BacktestPosition]:
        """ポジションオープン"""
        
        try:
            # ポジションサイズ計算
            position_size = self._calculate_position_size(signal, config)
            if position_size <= 0:
                return None
            
            # エントリー価格決定（スリッページ考慮）
            pip_value = self._get_pip_value(config.symbol)
            slippage = config.slippage_pips * pip_value
            
            if signal.signal_type == SignalType.BUY:
                entry_price = price_row['ask'] + slippage
                position_type = "buy"
            else:
                entry_price = price_row['bid'] - slippage
                position_type = "sell"
            
            # ポジション作成
            self.position_counter += 1
            position = BacktestPosition(
                id=self.position_counter,
                symbol=config.symbol,
                position_type=position_type,
                entry_time=price_row['datetime'],
                entry_price=entry_price,
                lot_size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                commission=position_size * config.commission_per_lot,
                signal_data={
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'risk_reward_ratio': signal.risk_reward_ratio
                }
            )
            
            logger.debug(f"ポジションオープン: {position.id} {position_type} {position.lot_size} @ {entry_price}")
            return position
            
        except Exception as e:
            logger.error(f"ポジションオープンエラー: {str(e)}")
            return None
    
    def _update_open_positions(self, positions: List[BacktestPosition], price_row: pd.Series, config: BacktestConfig) -> List[BacktestPosition]:
        """オープンポジションの更新"""
        
        for position in positions:
            if position.exit_time is not None:
                continue  # 既にクローズ済み
            
            current_price = price_row['close']
            
            # ストップロス・テイクプロフィットチェック
            should_close, exit_reason, exit_price = self._check_exit_conditions(
                position, price_row, config
            )
            
            if should_close:
                position.exit_time = price_row['datetime']
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                self._calculate_position_pnl(position, config)
                
                logger.debug(f"ポジションクローズ: {position.id} {exit_reason} @ {exit_price} P/L: {position.net_profit}")
            else:
                # 含み損益更新
                self._update_unrealized_pnl(position, current_price)
        
        return positions
    
    def _check_exit_conditions(self, position: BacktestPosition, price_row: pd.Series, config: BacktestConfig) -> Tuple[bool, str, float]:
        """エグジット条件チェック"""
        
        high = price_row['high']
        low = price_row['low']
        close = price_row['close']
        
        if position.position_type == "buy":
            # 買いポジション
            if position.stop_loss and low <= position.stop_loss:
                return True, "Stop Loss", position.stop_loss
            if position.take_profit and high >= position.take_profit:
                return True, "Take Profit", position.take_profit
        else:
            # 売りポジション  
            if position.stop_loss and high >= position.stop_loss:
                return True, "Stop Loss", position.stop_loss
            if position.take_profit and low <= position.take_profit:
                return True, "Take Profit", position.take_profit
        
        return False, "", close
    
    def _calculate_position_size(self, signal: TradingSignal, config: BacktestConfig) -> float:
        """ポジションサイズ計算"""
        
        try:
            if not signal.stop_loss:
                return 0.0
            
            # リスク金額
            risk_amount = self.current_balance * config.risk_per_trade
            
            # ストップロス幅（pips）
            pip_value = self._get_pip_value(config.symbol)
            stop_loss_pips = abs(signal.entry_price - signal.stop_loss) / pip_value
            
            if stop_loss_pips <= 0:
                return 0.0
            
            # ポジションサイズ計算
            position_size = risk_amount / (stop_loss_pips * pip_value * 100000)
            
            # 最小・最大制限
            position_size = max(0.01, min(position_size, 10.0))
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {str(e)}")
            return 0.0
    
    def _calculate_position_pnl(self, position: BacktestPosition, config: BacktestConfig):
        """ポジション損益計算"""
        
        if not position.exit_price:
            return
        
        # 価格差計算
        price_diff = position.exit_price - position.entry_price
        if position.position_type == "sell":
            price_diff = -price_diff
        
        # 総損益計算
        position.profit_loss = price_diff * position.lot_size * 100000
        
        # 手数料・スワップ考慮
        position.net_profit = position.profit_loss - position.commission - position.swap
        
        # 残高更新（複利効果）
        if config.enable_compound_interest:
            self.current_balance += position.net_profit
    
    def _update_unrealized_pnl(self, position: BacktestPosition, current_price: float):
        """含み損益更新"""
        
        price_diff = current_price - position.entry_price
        if position.position_type == "sell":
            price_diff = -price_diff
        
        unrealized_pnl = price_diff * position.lot_size * 100000
        
        # 最大利益・最大含み損更新
        if unrealized_pnl > position.max_profit:
            position.max_profit = unrealized_pnl
        if unrealized_pnl < position.max_drawdown:
            position.max_drawdown = unrealized_pnl
    
    def _record_equity(self, timestamp: datetime, open_positions: List[BacktestPosition]):
        """エクイティ記録"""
        
        unrealized_pnl = sum(
            self._get_current_unrealized_pnl(pos, 0.0) for pos in open_positions
        )
        
        equity = self.current_balance + unrealized_pnl
        
        self.equity_history.append({
            'datetime': timestamp,
            'balance': self.current_balance,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': len(open_positions)
        })
    
    def _get_current_unrealized_pnl(self, position: BacktestPosition, current_price: float) -> float:
        """現在の含み損益取得"""
        # 簡易実装（実際のcurrent_priceが必要）
        return 0.0
    
    def _calculate_metrics(self, positions: List[BacktestPosition], config: BacktestConfig) -> BacktestMetrics:
        """メトリクス計算"""
        
        if not positions:
            return BacktestMetrics()
        
        # 基本統計
        total_trades = len(positions)
        winning_trades = len([p for p in positions if p.net_profit > 0])
        losing_trades = len([p for p in positions if p.net_profit < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 損益統計
        profits = [p.net_profit for p in positions]
        wins = [p.net_profit for p in positions if p.net_profit > 0]
        losses = [p.net_profit for p in positions if p.net_profit < 0]
        
        total_profit = sum(profits)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        average_win = np.mean(wins) if wins else 0
        average_loss = abs(np.mean(losses)) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = abs(min(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 期間統計
        start_date = min(p.entry_time for p in positions)
        end_date = max(p.exit_time for p in positions if p.exit_time)
        total_days = (end_date - start_date).days if end_date else 0
        
        # 取引期間統計
        durations = []
        for p in positions:
            if p.exit_time:
                duration = (p.exit_time - p.entry_time).total_seconds() / 3600  # 時間単位
                durations.append(duration)
        
        average_trade_duration = np.mean(durations) if durations else 0
        
        # 連続勝敗統計
        consecutive_wins, consecutive_losses = self._calculate_consecutive_stats(positions)
        
        # 期待値
        expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
        
        # リスクリワード統計
        risk_rewards = []
        for p in positions:
            if p.signal_data and 'risk_reward_ratio' in p.signal_data:
                risk_rewards.append(p.signal_data['risk_reward_ratio'])
        
        average_risk_reward = np.mean(risk_rewards) if risk_rewards else 0
        best_risk_reward = max(risk_rewards) if risk_rewards else 0
        worst_risk_reward = min(risk_rewards) if risk_rewards else 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            average_trade_duration=average_trade_duration,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
            expectancy=expectancy,
            average_risk_reward=average_risk_reward,
            best_risk_reward=best_risk_reward,
            worst_risk_reward=worst_risk_reward
        )
    
    def _calculate_consecutive_stats(self, positions: List[BacktestPosition]) -> Tuple[int, int]:
        """連続勝敗統計計算"""
        
        if not positions:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for position in positions:
            if position.net_profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif position.net_profit < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _generate_equity_curve(self, positions: List[BacktestPosition], config: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """エクイティカーブ生成"""
        
        if not self.equity_history:
            return pd.DataFrame(), pd.DataFrame()
        
        # エクイティカーブ
        equity_df = pd.DataFrame(self.equity_history)
        equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
        equity_df.set_index('datetime', inplace=True)
        
        # 日次リターン
        daily_equity = equity_df['equity'].resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        
        daily_returns_df = pd.DataFrame({
            'date': daily_returns.index,
            'daily_return': daily_returns.values,
            'cumulative_return': (1 + daily_returns).cumprod() - 1
        })
        
        return equity_df, daily_returns_df
    
    def _calculate_monthly_returns(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """月次リターン計算"""
        
        if daily_returns.empty:
            return pd.DataFrame()
        
        daily_returns_series = pd.Series(
            daily_returns['daily_return'].values,
            index=pd.to_datetime(daily_returns['date'])
        )
        
        monthly_returns = (1 + daily_returns_series).resample('M').prod() - 1
        
        return pd.DataFrame({
            'month': monthly_returns.index.strftime('%Y-%m'),
            'monthly_return': monthly_returns.values
        })
    
    def _analyze_drawdown_periods(self, equity_curve: pd.DataFrame) -> List[Dict[str, Any]]:
        """ドローダウン期間分析"""
        
        if equity_curve.empty:
            return []
        
        equity = equity_curve['equity']
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        
        # ドローダウン期間の特定
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # 1%以上のドローダウン開始
                in_drawdown = True
                start_date = date
            elif dd >= -0.01 and in_drawdown:  # ドローダウン終了
                in_drawdown = False
                if start_date:
                    max_dd = drawdown[start_date:date].min()
                    duration = (date - start_date).days
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': date,
                        'duration_days': duration,
                        'max_drawdown': max_dd,
                        'recovery_time': duration
                    })
        
        return drawdown_periods
    
    def _create_performance_summary(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """パフォーマンスサマリー作成"""
        
        return {
            'overview': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.1%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'total_profit': f"{metrics.total_profit:.2f}",
                'expectancy': f"{metrics.expectancy:.2f}"
            },
            'profitability': {
                'gross_profit': metrics.gross_profit,
                'gross_loss': metrics.gross_loss,
                'average_win': metrics.average_win,
                'average_loss': metrics.average_loss,
                'largest_win': metrics.largest_win,
                'largest_loss': metrics.largest_loss
            },
            'consistency': {
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'max_consecutive_wins': metrics.max_consecutive_wins,
                'max_consecutive_losses': metrics.max_consecutive_losses,
                'average_trade_duration_hours': metrics.average_trade_duration
            },
            'risk_management': {
                'average_risk_reward': metrics.average_risk_reward,
                'best_risk_reward': metrics.best_risk_reward,
                'worst_risk_reward': metrics.worst_risk_reward
            }
        }
    
    def _get_pip_value(self, symbol: str) -> float:
        """ピップ値取得"""
        if symbol.endswith('JPY'):
            return 0.01
        else:
            return 0.0001
    
    def run_optimization(
        self,
        base_config: BacktestConfig,
        parameter_ranges: Dict[str, List[float]]
    ) -> List[BacktestResult]:
        """パラメータ最適化"""
        
        logger.info("パラメータ最適化開始")
        
        results = []
        total_combinations = 1
        for param_range in parameter_ranges.values():
            total_combinations *= len(param_range)
        
        logger.info(f"総組み合わせ数: {total_combinations}")
        
        # 全組み合わせでバックテスト実行
        combination_count = 0
        
        def generate_combinations(param_names, param_ranges, current_config, index=0):
            nonlocal combination_count
            
            if index == len(param_names):
                combination_count += 1
                logger.debug(f"最適化実行: {combination_count}/{total_combinations}")
                
                result = self.run_backtest(current_config)
                results.append(result)
                return
            
            param_name = param_names[index]
            for value in param_ranges[param_name]:
                new_config = type(current_config).__call__(**{
                    **current_config.__dict__,
                    param_name: value
                })
                generate_combinations(param_names, param_ranges, new_config, index + 1)
        
        param_names = list(parameter_ranges.keys())
        generate_combinations(param_names, parameter_ranges, base_config)
        
        logger.info(f"パラメータ最適化完了: {len(results)}結果")
        return results
    
    def _validate_data_sufficiency(self, price_data: pd.DataFrame, config: BacktestConfig) -> bool:
        """データ充足性検証"""
        if price_data.empty:
            return False
        
        # 最小データ数チェック
        min_required_records = 100  # 最低100レコード必要
        if len(price_data) < min_required_records:
            return False
        
        # 日付範囲チェック
        data_start = price_data['datetime'].min()
        data_end = price_data['datetime'].max()
        
        if data_start > config.start_date or data_end < config.end_date:
            return False
        
        return True
    
    def _get_pip_value(self, symbol: str) -> float:
        """通貨ペアのPIP値を取得"""
        if 'JPY' in symbol:
            return 0.01  # JPY pairs
        else:
            return 0.0001  # Major pairs
    
    def _calculate_position_size(self, price: float, stop_loss: float, risk_amount: float) -> float:
        """ポジションサイズ計算"""
        if stop_loss == 0 or price == stop_loss:
            return 0.0
        
        risk_per_pip = abs(price - stop_loss)
        if risk_per_pip == 0:
            return 0.0
        
        # 基本ロットサイズ計算
        position_size = risk_amount / risk_per_pip
        
        # 最小・最大制限
        min_lot = 1000  # 1000通貨単位
        max_lot = 1000000  # 100万通貨単位
        
        return max(min_lot, min(max_lot, position_size))