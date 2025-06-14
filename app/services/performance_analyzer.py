import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from app.services.backtest_engine import BacktestResult, BacktestMetrics, BacktestPosition

logger = logging.getLogger(__name__)


class PerformanceRating(str, Enum):
    """パフォーマンス評価"""
    EXCELLENT = "excellent"  # 90-100点
    GOOD = "good"           # 70-89点  
    AVERAGE = "average"     # 50-69点
    POOR = "poor"          # 30-49点
    TERRIBLE = "terrible"   # 0-29点


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    max_drawdown: float
    max_drawdown_duration: int  # 日数
    value_at_risk_95: float     # 95% VaR
    conditional_var_95: float   # 95% CVaR
    volatility: float           # 年率ボラティリティ
    downside_deviation: float
    beta: float                 # 市場ベータ（ベンチマーク対比）
    correlation_to_market: float


@dataclass
class ReturnMetrics:
    """リターンメトリクス"""
    total_return: float
    annualized_return: float
    monthly_return_avg: float
    monthly_return_std: float
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    monthly_win_rate: float


@dataclass
class RatioMetrics:
    """比率メトリクス"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float


@dataclass
class TradingMetrics:
    """取引メトリクス"""
    profit_factor: float
    recovery_factor: float
    payoff_ratio: float         # 平均利益/平均損失
    trade_efficiency: float     # 利益取引時間/全取引時間
    market_exposure: float      # 市場エクスポージャー率
    trade_frequency: float      # 年間取引頻度
    
    # 統計的指標
    trades_per_month: float
    average_holding_period: float  # 日数
    win_streak_analysis: Dict[str, int]
    loss_streak_analysis: Dict[str, int]


@dataclass
class PerformanceScore:
    """パフォーマンススコア"""
    overall_score: float        # 0-100
    rating: PerformanceRating
    
    # 分野別スコア
    profitability_score: float  # 収益性
    consistency_score: float    # 一貫性
    risk_management_score: float # リスク管理
    efficiency_score: float     # 効率性
    
    # 詳細評価
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


@dataclass
class ComprehensiveAnalysis:
    """包括的分析結果"""
    backtest_result: BacktestResult
    risk_metrics: RiskMetrics
    return_metrics: ReturnMetrics
    ratio_metrics: RatioMetrics
    trading_metrics: TradingMetrics
    performance_score: PerformanceScore
    
    # 分析チャート（Base64エンコード）
    equity_curve_chart: Optional[str] = None
    monthly_returns_chart: Optional[str] = None
    drawdown_chart: Optional[str] = None
    trade_distribution_chart: Optional[str] = None
    
    # 詳細レポート
    executive_summary: Dict[str, Any] = None
    detailed_analysis: Dict[str, Any] = None


class PerformanceAnalyzer:
    """パフォーマンス分析エンジン"""
    
    def __init__(self):
        # 評価基準設定
        self.scoring_weights = {
            'profitability': 0.35,    # 収益性35%
            'consistency': 0.25,      # 一貫性25%
            'risk_management': 0.25,  # リスク管理25%
            'efficiency': 0.15        # 効率性15%
        }
        
        # ベンチマーク設定（年率）
        self.benchmarks = {
            'risk_free_rate': 0.02,   # 無リスク金利2%
            'market_return': 0.08,    # 市場リターン8%
            'target_sharpe': 1.0,     # 目標シャープレシオ
            'target_drawdown': 0.10   # 目標最大ドローダウン10%
        }
    
    def analyze_performance(self, backtest_result: BacktestResult) -> ComprehensiveAnalysis:
        """包括的パフォーマンス分析"""
        
        logger.info("パフォーマンス分析開始")
        
        try:
            # 各種メトリクス計算
            risk_metrics = self._calculate_risk_metrics(backtest_result)
            return_metrics = self._calculate_return_metrics(backtest_result)
            ratio_metrics = self._calculate_ratio_metrics(backtest_result, risk_metrics, return_metrics)
            trading_metrics = self._calculate_trading_metrics(backtest_result)
            
            # パフォーマンススコア計算
            performance_score = self._calculate_performance_score(
                risk_metrics, return_metrics, ratio_metrics, trading_metrics
            )
            
            # チャート生成
            charts = self._generate_charts(backtest_result)
            
            # レポート生成
            executive_summary = self._create_executive_summary(
                backtest_result, performance_score, risk_metrics, return_metrics
            )
            detailed_analysis = self._create_detailed_analysis(
                risk_metrics, return_metrics, ratio_metrics, trading_metrics
            )
            
            analysis = ComprehensiveAnalysis(
                backtest_result=backtest_result,
                risk_metrics=risk_metrics,
                return_metrics=return_metrics,
                ratio_metrics=ratio_metrics,
                trading_metrics=trading_metrics,
                performance_score=performance_score,
                equity_curve_chart=charts.get('equity_curve'),
                monthly_returns_chart=charts.get('monthly_returns'),
                drawdown_chart=charts.get('drawdown'),
                trade_distribution_chart=charts.get('trade_distribution'),
                executive_summary=executive_summary,
                detailed_analysis=detailed_analysis
            )
            
            logger.info(f"パフォーマンス分析完了: 総合スコア {performance_score.overall_score:.1f}/100 ({performance_score.rating.value})")
            return analysis
            
        except Exception as e:
            logger.error(f"パフォーマンス分析エラー: {str(e)}")
            raise
    
    def _calculate_risk_metrics(self, backtest_result: BacktestResult) -> RiskMetrics:
        """リスクメトリクス計算"""
        
        if backtest_result.daily_returns.empty:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        daily_returns = backtest_result.daily_returns['daily_return']
        equity_curve = backtest_result.equity_curve['equity'] if not backtest_result.equity_curve.empty else pd.Series()
        
        # 最大ドローダウン
        if not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # ドローダウン期間
            in_drawdown = drawdown < -0.01
            if in_drawdown.any():
                drawdown_periods = []
                start = None
                for i, is_dd in enumerate(in_drawdown):
                    if is_dd and start is None:
                        start = i
                    elif not is_dd and start is not None:
                        drawdown_periods.append(i - start)
                        start = None
                max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            else:
                max_drawdown_duration = 0
        else:
            max_drawdown = 0
            max_drawdown_duration = 0
        
        # VaR・CVaR計算
        if len(daily_returns) > 0:
            var_95 = np.percentile(daily_returns, 5)  # 5%ile
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            
            # ボラティリティ（年率）
            volatility = daily_returns.std() * np.sqrt(252)
            
            # 下方偏差
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        else:
            var_95 = cvar_95 = volatility = downside_deviation = 0
        
        # ベータ・相関（簡易実装、実際は市場データが必要）
        beta = 1.0  # 仮設定
        correlation_to_market = 0.5  # 仮設定
        
        return RiskMetrics(
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            volatility=volatility,
            downside_deviation=downside_deviation,
            beta=beta,
            correlation_to_market=correlation_to_market
        )
    
    def _calculate_return_metrics(self, backtest_result: BacktestResult) -> ReturnMetrics:
        """リターンメトリクス計算"""
        
        metrics = backtest_result.metrics
        
        # 総リターン
        initial_balance = backtest_result.config.initial_balance
        total_return = metrics.total_profit / initial_balance
        
        # 年率リターン
        if metrics.total_days > 0:
            years = metrics.total_days / 365.25
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        else:
            annualized_return = 0
        
        # 月次リターン分析
        if backtest_result.monthly_returns is not None and not backtest_result.monthly_returns.empty:
            monthly_returns = backtest_result.monthly_returns['monthly_return']
            monthly_return_avg = monthly_returns.mean()
            monthly_return_std = monthly_returns.std()
            best_month = monthly_returns.max()
            worst_month = monthly_returns.min()
            positive_months = (monthly_returns > 0).sum()
            negative_months = (monthly_returns < 0).sum()
            total_months = len(monthly_returns)
            monthly_win_rate = positive_months / total_months if total_months > 0 else 0
        else:
            monthly_return_avg = monthly_return_std = best_month = worst_month = 0
            positive_months = negative_months = 0
            monthly_win_rate = 0
        
        return ReturnMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_return_avg=monthly_return_avg,
            monthly_return_std=monthly_return_std,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            negative_months=negative_months,
            monthly_win_rate=monthly_win_rate
        )
    
    def _calculate_ratio_metrics(self, backtest_result: BacktestResult, risk_metrics: RiskMetrics, return_metrics: ReturnMetrics) -> RatioMetrics:
        """比率メトリクス計算"""
        
        rf_rate = self.benchmarks['risk_free_rate']
        market_return = self.benchmarks['market_return']
        
        # シャープレシオ
        excess_return = return_metrics.annualized_return - rf_rate
        sharpe_ratio = excess_return / risk_metrics.volatility if risk_metrics.volatility > 0 else 0
        
        # ソルティノレシオ
        sortino_ratio = excess_return / risk_metrics.downside_deviation if risk_metrics.downside_deviation > 0 else 0
        
        # カルマレシオ
        calmar_ratio = return_metrics.annualized_return / risk_metrics.max_drawdown if risk_metrics.max_drawdown > 0 else 0
        
        # スターリングレシオ
        sterling_ratio = return_metrics.annualized_return / (risk_metrics.max_drawdown + 0.1) if risk_metrics.max_drawdown >= 0 else 0
        
        # 情報比率（簡易）
        information_ratio = sharpe_ratio  # 簡易実装
        
        # トレイナー比率
        treynor_ratio = excess_return / risk_metrics.beta if risk_metrics.beta > 0 else 0
        
        # ジェンセンアルファ
        jensen_alpha = return_metrics.annualized_return - (rf_rate + risk_metrics.beta * (market_return - rf_rate))
        
        return RatioMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha
        )
    
    def _calculate_trading_metrics(self, backtest_result: BacktestResult) -> TradingMetrics:
        """取引メトリクス計算"""
        
        metrics = backtest_result.metrics
        positions = backtest_result.positions
        
        # 基本比率
        profit_factor = metrics.profit_factor
        recovery_factor = abs(metrics.total_profit / metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
        payoff_ratio = metrics.average_win / abs(metrics.average_loss) if metrics.average_loss != 0 else 0
        
        # 取引効率
        if positions:
            total_trade_time = sum(
                (p.exit_time - p.entry_time).total_seconds() / 3600 
                for p in positions if p.exit_time
            )
            profitable_trade_time = sum(
                (p.exit_time - p.entry_time).total_seconds() / 3600 
                for p in positions if p.exit_time and p.net_profit > 0
            )
            trade_efficiency = profitable_trade_time / total_trade_time if total_trade_time > 0 else 0
            
            # 市場エクスポージャー
            if metrics.total_days > 0:
                market_exposure = total_trade_time / (metrics.total_days * 24)
            else:
                market_exposure = 0
            
            # 取引頻度
            if metrics.total_days > 0:
                trade_frequency = metrics.total_trades / (metrics.total_days / 365.25)
                trades_per_month = metrics.total_trades / (metrics.total_days / 30.44)
            else:
                trade_frequency = trades_per_month = 0
            
            # 平均保有期間
            average_holding_period = metrics.average_trade_duration / 24  # 日数に変換
            
        else:
            trade_efficiency = market_exposure = trade_frequency = 0
            trades_per_month = average_holding_period = 0
        
        # 連勝連敗分析
        win_streak_analysis = {
            'max_consecutive_wins': metrics.max_consecutive_wins,
            'avg_win_streak': self._calculate_average_streak(positions, True),
        }
        
        loss_streak_analysis = {
            'max_consecutive_losses': metrics.max_consecutive_losses,
            'avg_loss_streak': self._calculate_average_streak(positions, False),
        }
        
        return TradingMetrics(
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio,
            trade_efficiency=trade_efficiency,
            market_exposure=market_exposure,
            trade_frequency=trade_frequency,
            trades_per_month=trades_per_month,
            average_holding_period=average_holding_period,
            win_streak_analysis=win_streak_analysis,
            loss_streak_analysis=loss_streak_analysis
        )
    
    def _calculate_average_streak(self, positions: List[BacktestPosition], winning: bool) -> float:
        """平均連勝/連敗計算"""
        
        if not positions:
            return 0
        
        streaks = []
        current_streak = 0
        
        for pos in positions:
            is_winner = pos.net_profit > 0
            
            if is_winner == winning:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return np.mean(streaks) if streaks else 0
    
    def _calculate_performance_score(self, risk_metrics: RiskMetrics, return_metrics: ReturnMetrics, 
                                   ratio_metrics: RatioMetrics, trading_metrics: TradingMetrics) -> PerformanceScore:
        """パフォーマンススコア計算"""
        
        # 分野別スコア計算（0-100点）
        profitability_score = self._score_profitability(return_metrics, trading_metrics)
        consistency_score = self._score_consistency(return_metrics, risk_metrics, trading_metrics)
        risk_management_score = self._score_risk_management(risk_metrics, ratio_metrics)
        efficiency_score = self._score_efficiency(trading_metrics, ratio_metrics)
        
        # 総合スコア
        overall_score = (
            profitability_score * self.scoring_weights['profitability'] +
            consistency_score * self.scoring_weights['consistency'] +
            risk_management_score * self.scoring_weights['risk_management'] +
            efficiency_score * self.scoring_weights['efficiency']
        )
        
        # 評価判定
        if overall_score >= 90:
            rating = PerformanceRating.EXCELLENT
        elif overall_score >= 70:
            rating = PerformanceRating.GOOD
        elif overall_score >= 50:
            rating = PerformanceRating.AVERAGE
        elif overall_score >= 30:
            rating = PerformanceRating.POOR
        else:
            rating = PerformanceRating.TERRIBLE
        
        # 強み・弱み・推奨事項
        strengths, weaknesses, recommendations = self._analyze_strengths_weaknesses(
            profitability_score, consistency_score, risk_management_score, efficiency_score,
            risk_metrics, return_metrics, ratio_metrics, trading_metrics
        )
        
        return PerformanceScore(
            overall_score=overall_score,
            rating=rating,
            profitability_score=profitability_score,
            consistency_score=consistency_score,
            risk_management_score=risk_management_score,
            efficiency_score=efficiency_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _score_profitability(self, return_metrics: ReturnMetrics, trading_metrics: TradingMetrics) -> float:
        """収益性スコア"""
        
        # 年率リターン評価（20%で満点）
        return_score = min(return_metrics.annualized_return / 0.20 * 40, 40)
        
        # プロフィットファクター評価（2.0で満点）
        pf_score = min(trading_metrics.profit_factor / 2.0 * 30, 30)
        
        # 期待値評価
        expectancy_score = 15  # 簡易実装
        
        # 月次勝率評価
        monthly_win_score = return_metrics.monthly_win_rate * 15
        
        return max(0, min(100, return_score + pf_score + expectancy_score + monthly_win_score))
    
    def _score_consistency(self, return_metrics: ReturnMetrics, risk_metrics: RiskMetrics, trading_metrics: TradingMetrics) -> float:
        """一貫性スコア"""
        
        # 月次リターンの安定性
        if return_metrics.monthly_return_std > 0:
            stability_score = max(0, 40 - return_metrics.monthly_return_std * 200)
        else:
            stability_score = 40
        
        # 最悪月の評価
        worst_month_score = max(0, 30 + return_metrics.worst_month * 300)
        
        # 取引の一貫性
        win_rate_score = min(trading_metrics.profit_factor / 2.0 * 30, 30)
        
        return max(0, min(100, stability_score + worst_month_score + win_rate_score))
    
    def _score_risk_management(self, risk_metrics: RiskMetrics, ratio_metrics: RatioMetrics) -> float:
        """リスク管理スコア"""
        
        # 最大ドローダウン評価（10%以下で満点）
        dd_score = max(0, 40 - risk_metrics.max_drawdown * 400)
        
        # シャープレシオ評価（1.0以上で満点）
        sharpe_score = min(ratio_metrics.sharpe_ratio * 30, 30)
        
        # ソルティノレシオ評価
        sortino_score = min(ratio_metrics.sortino_ratio * 20, 20)
        
        # VaR評価
        var_score = max(0, 10 + risk_metrics.value_at_risk_95 * 100)
        
        return max(0, min(100, dd_score + sharpe_score + sortino_score + var_score))
    
    def _score_efficiency(self, trading_metrics: TradingMetrics, ratio_metrics: RatioMetrics) -> float:
        """効率性スコア"""
        
        # 取引効率
        efficiency_score = trading_metrics.trade_efficiency * 40
        
        # カルマレシオ
        calmar_score = min(ratio_metrics.calmar_ratio * 20, 30)
        
        # 市場エクスポージャー（適度なエクスポージャーが良い）
        exposure_score = 30 - abs(trading_metrics.market_exposure - 0.3) * 100
        
        return max(0, min(100, efficiency_score + calmar_score + exposure_score))
    
    def _analyze_strengths_weaknesses(self, prof_score: float, cons_score: float, risk_score: float, eff_score: float,
                                    risk_metrics: RiskMetrics, return_metrics: ReturnMetrics, 
                                    ratio_metrics: RatioMetrics, trading_metrics: TradingMetrics) -> Tuple[List[str], List[str], List[str]]:
        """強み・弱み・推奨事項分析"""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # 収益性分析
        if prof_score >= 70:
            strengths.append(f"優れた収益性（年率リターン: {return_metrics.annualized_return:.1%}）")
        elif prof_score < 40:
            weaknesses.append("収益性が低い")
            recommendations.append("戦略の見直しまたはリスク許容度の調整を検討")
        
        # 一貫性分析
        if cons_score >= 70:
            strengths.append("安定した取引パフォーマンス")
        elif cons_score < 40:
            weaknesses.append("パフォーマンスの変動が大きい")
            recommendations.append("ポジションサイズの調整またはリスク管理強化")
        
        # リスク管理分析
        if risk_score >= 70:
            strengths.append("効果的なリスク管理")
        elif risk_score < 40:
            weaknesses.append(f"リスク管理に課題（最大DD: {risk_metrics.max_drawdown:.1%}）")
            recommendations.append("ストップロス設定の見直しまたはポジションサイズ縮小")
        
        # 効率性分析
        if eff_score >= 70:
            strengths.append("高い取引効率")
        elif eff_score < 40:
            weaknesses.append("取引効率が低い")
            recommendations.append("エントリー・エグジット条件の最適化")
        
        # 具体的指標による分析
        if ratio_metrics.sharpe_ratio >= 1.5:
            strengths.append(f"優秀なシャープレシオ（{ratio_metrics.sharpe_ratio:.2f}）")
        elif ratio_metrics.sharpe_ratio < 0.5:
            weaknesses.append(f"低いシャープレシオ（{ratio_metrics.sharpe_ratio:.2f}）")
        
        if trading_metrics.profit_factor >= 1.5:
            strengths.append(f"良好なプロフィットファクター（{trading_metrics.profit_factor:.2f}）")
        elif trading_metrics.profit_factor < 1.2:
            weaknesses.append(f"低いプロフィットファクター（{trading_metrics.profit_factor:.2f}）")
        
        return strengths, weaknesses, recommendations
    
    def _generate_charts(self, backtest_result: BacktestResult) -> Dict[str, str]:
        """チャート生成"""
        
        charts = {}
        
        try:
            # エクイティカーブ
            if not backtest_result.equity_curve.empty:
                charts['equity_curve'] = self._create_equity_chart(backtest_result.equity_curve)
            
            # 月次リターン
            if backtest_result.monthly_returns is not None and not backtest_result.monthly_returns.empty:
                charts['monthly_returns'] = self._create_monthly_returns_chart(backtest_result.monthly_returns)
            
            # ドローダウン
            if not backtest_result.equity_curve.empty:
                charts['drawdown'] = self._create_drawdown_chart(backtest_result.equity_curve)
            
            # 取引分布
            if backtest_result.positions:
                charts['trade_distribution'] = self._create_trade_distribution_chart(backtest_result.positions)
        
        except Exception as e:
            logger.error(f"チャート生成エラー: {str(e)}")
        
        return charts
    
    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> str:
        """エクイティカーブチャート"""
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_monthly_returns_chart(self, monthly_returns: pd.DataFrame) -> str:
        """月次リターンチャート"""
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(monthly_returns)), monthly_returns['monthly_return'])
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return')
        plt.xticks(range(len(monthly_returns)), monthly_returns['month'], rotation=45)
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_drawdown_chart(self, equity_curve: pd.DataFrame) -> str:
        """ドローダウンチャート"""
        
        equity = equity_curve['equity']
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_curve.index, drawdown, 0, alpha=0.5, color='red')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_trade_distribution_chart(self, positions: List[BacktestPosition]) -> str:
        """取引分布チャート"""
        
        profits = [p.net_profit for p in positions]
        
        plt.figure(figsize=(12, 6))
        plt.hist(profits, bins=30, alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Trade P&L Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_executive_summary(self, backtest_result: BacktestResult, performance_score: PerformanceScore,
                                risk_metrics: RiskMetrics, return_metrics: ReturnMetrics) -> Dict[str, Any]:
        """エグゼクティブサマリー"""
        
        return {
            'overview': {
                'strategy_performance': performance_score.rating.value,
                'overall_score': f"{performance_score.overall_score:.1f}/100",
                'test_period': f"{backtest_result.config.start_date.date()} - {backtest_result.config.end_date.date()}",
                'total_trades': backtest_result.metrics.total_trades,
                'win_rate': f"{backtest_result.metrics.win_rate:.1%}"
            },
            'key_metrics': {
                'total_return': f"{return_metrics.total_return:.1%}",
                'annualized_return': f"{return_metrics.annualized_return:.1%}",
                'max_drawdown': f"{risk_metrics.max_drawdown:.1%}",
                'sharpe_ratio': f"{backtest_result.metrics.total_profit / risk_metrics.volatility if risk_metrics.volatility > 0 else 0:.2f}",
                'profit_factor': f"{backtest_result.metrics.profit_factor:.2f}"
            },
            'strengths': performance_score.strengths,
            'areas_for_improvement': performance_score.weaknesses,
            'recommendations': performance_score.recommendations[:3]  # 上位3つの推奨事項
        }
    
    def _create_detailed_analysis(self, risk_metrics: RiskMetrics, return_metrics: ReturnMetrics,
                                ratio_metrics: RatioMetrics, trading_metrics: TradingMetrics) -> Dict[str, Any]:
        """詳細分析"""
        
        return {
            'risk_analysis': {
                'max_drawdown': f"{risk_metrics.max_drawdown:.1%}",
                'volatility': f"{risk_metrics.volatility:.1%}",
                'var_95': f"{risk_metrics.value_at_risk_95:.1%}",
                'downside_deviation': f"{risk_metrics.downside_deviation:.1%}"
            },
            'return_analysis': {
                'total_return': f"{return_metrics.total_return:.1%}",
                'annualized_return': f"{return_metrics.annualized_return:.1%}",
                'best_month': f"{return_metrics.best_month:.1%}",
                'worst_month': f"{return_metrics.worst_month:.1%}",
                'monthly_win_rate': f"{return_metrics.monthly_win_rate:.1%}"
            },
            'ratio_analysis': {
                'sharpe_ratio': f"{ratio_metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{ratio_metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{ratio_metrics.calmar_ratio:.2f}",
                'information_ratio': f"{ratio_metrics.information_ratio:.2f}"
            },
            'trading_analysis': {
                'profit_factor': f"{trading_metrics.profit_factor:.2f}",
                'recovery_factor': f"{trading_metrics.recovery_factor:.2f}",
                'trade_efficiency': f"{trading_metrics.trade_efficiency:.1%}",
                'market_exposure': f"{trading_metrics.market_exposure:.1%}",
                'avg_holding_period': f"{trading_metrics.average_holding_period:.1f} days"
            }
        }
    
    def compare_strategies(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """戦略比較分析"""
        
        if len(results) < 2:
            raise ValueError("比較には最低2つの結果が必要です")
        
        comparison_metrics = []
        
        for i, result in enumerate(results):
            analysis = self.analyze_performance(result)
            
            comparison_metrics.append({
                'strategy_id': f"Strategy_{i+1}",
                'overall_score': analysis.performance_score.overall_score,
                'total_return': analysis.return_metrics.total_return,
                'max_drawdown': analysis.risk_metrics.max_drawdown,
                'sharpe_ratio': analysis.ratio_metrics.sharpe_ratio,
                'profit_factor': analysis.trading_metrics.profit_factor,
                'win_rate': result.metrics.win_rate,
                'total_trades': result.metrics.total_trades
            })
        
        # ランキング
        rankings = {
            'by_total_score': sorted(comparison_metrics, key=lambda x: x['overall_score'], reverse=True),
            'by_return': sorted(comparison_metrics, key=lambda x: x['total_return'], reverse=True),
            'by_sharpe': sorted(comparison_metrics, key=lambda x: x['sharpe_ratio'], reverse=True),
            'by_drawdown': sorted(comparison_metrics, key=lambda x: x['max_drawdown'])  # 昇順（小さい方が良い）
        }
        
        return {
            'strategy_comparison': comparison_metrics,
            'rankings': rankings,
            'best_overall': rankings['by_total_score'][0],
            'summary': {
                'total_strategies': len(results),
                'avg_score': np.mean([m['overall_score'] for m in comparison_metrics]),
                'score_range': [
                    min(m['overall_score'] for m in comparison_metrics),
                    max(m['overall_score'] for m in comparison_metrics)
                ]
            }
        }